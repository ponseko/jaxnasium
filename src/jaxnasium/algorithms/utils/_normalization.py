from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

import jaxnasium as jym

from ._transition import Transition

"""
A module that provides utilities for normalization in reinforcement learning algorithms.
To be used instead of Environment Wrappers.

Based on Brax's `RunningStatistics`.
https://github.com/google/brax/blob/241f9bc5bbd003f9cfc9ded7613388e2fe125af6/brax/training/acme/running_statistics.py

"""


class RunningStatisticsState(eqx.Module):
    """Full state of running statistics computation."""

    mean: Array
    std: Array
    count: Array
    summed_variance: Array

    center_mean: bool = eqx.field(static=True)

    def __init__(self, pytree_example, center_mean: bool = True):
        dtype: type = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32  # type: ignore

        self.count = jnp.zeros((), dtype=dtype)
        self.mean = optax.tree_utils.tree_zeros_like(pytree_example, dtype=dtype)
        self.summed_variance = optax.tree_utils.tree_zeros_like(
            pytree_example, dtype=dtype
        )
        self.std = optax.tree_utils.tree_ones_like(pytree_example, dtype=dtype)

        # Typically disabled for rewards, but enabled for observations.
        self.center_mean = center_mean

    def update(
        self,
        batch: Array,
        *,
        weights: Optional[jnp.ndarray] = None,
        std_min_value: float = 1e-6,
        std_max_value: float = 1e6,
        validate_shapes: bool = True,
    ) -> "RunningStatisticsState":
        """
        Update the running statistics with a new observation.

        NOTE from Brax: by default will use int32 for counts and float32 for accumulated
        variance. This results in an integer overflow after 2^31 data points and
        degrading precision after 2^24 batch updates or even earlier if variance
        updates have large dynamic range.
        """

        def _compute_node_statistics(
            mean: jnp.ndarray, summed_variance: jnp.ndarray, batch: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            assert isinstance(mean, jnp.ndarray), type(mean)
            assert isinstance(summed_variance, jnp.ndarray), type(summed_variance)
            # The mean and the sum of past variances are updated with Welford's
            # algorithm using batches (see https://stackoverflow.com/q/56402955).
            diff_to_old_mean = batch - mean

            mean_update = jnp.sum(diff_to_old_mean, axis=batch_axis) / count
            mean = mean + mean_update

            diff_to_new_mean = batch - mean
            variance_update = diff_to_old_mean * diff_to_new_mean
            variance_update = jnp.sum(variance_update, axis=batch_axis)
            summed_variance = summed_variance + variance_update
            return mean, summed_variance

        def compute_std(summed_variance: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
            assert isinstance(summed_variance, jnp.ndarray)
            # Summed variance can get negative due to rounding errors.
            summed_variance = jnp.maximum(summed_variance, 0)
            std = jnp.sqrt(summed_variance / count)
            std = jnp.clip(std, std_min_value, std_max_value)
            return std

        assert jax.tree.structure(batch) == jax.tree.structure(self.mean)
        batch_leaves = jax.tree.leaves(batch)

        if not batch_leaves:  # State and batch are both empty. Nothing to normalize.
            return self

        batch_shape = batch_leaves[0].shape
        # We assume the batch dimensions always go first.
        batch_dims = batch_shape[
            : len(batch_shape) - jax.tree.leaves(self.mean)[0].ndim
        ]
        batch_axis = range(len(batch_dims))
        if weights is None:
            step_increment = jnp.prod(jnp.array(batch_dims))
        else:
            step_increment = jnp.sum(weights)
        count = self.count + step_increment

        # # Validation is important. If the shapes don't match exactly, but are
        # # compatible, arrays will be silently broadcasted resulting in incorrect
        # # statistics.
        if validate_shapes:
            self._validate_batch_shapes(batch, self.mean, batch_dims)

        updated_stats = jax.tree.map(
            _compute_node_statistics, self.mean, self.summed_variance, batch
        )
        mean = jax.tree.map(lambda _, x: x[0], self.mean, updated_stats)
        summed_variance = jax.tree.map(lambda _, x: x[1], self.mean, updated_stats)
        std = jax.tree.map(compute_std, summed_variance, self.std)

        return eqx.tree_at(
            lambda x: (x.mean, x.std, x.count, x.summed_variance),
            self,
            (mean, std, count, summed_variance),
        )

    def normalize(self, batch: Array) -> Array:
        if self.center_mean:
            batch = optax.tree.sub(batch, self.mean)
        return jax.tree.map(
            lambda data, std: data / (std + 1e-8),
            batch,
            self.std,
        )

    @staticmethod
    def _validate_batch_shapes(
        batch: PyTree,
        reference_sample: PyTree,
        batch_dims: Tuple[int, ...],
    ) -> None:
        """Verifies shapes of the batch leaves against the reference sample.

        Checks that batch dimensions are the same in all leaves in the batch.
        Checks that non-batch dimensions for all leaves in the batch are the same
        as in the reference sample.

        Arguments:
            batch: the nested batch of data to be verified.
            reference_sample: the nested array to check non-batch dimensions.
            batch_dims: a Tuple of indices of batch dimensions in the batch shape.
        """

        def validate_node_shape(
            reference_sample: jnp.ndarray, batch: jnp.ndarray
        ) -> None:
            expected_shape = batch_dims + reference_sample.shape
            assert batch.shape == expected_shape, f"{batch.shape} != {expected_shape}"

        jax.tree.map(validate_node_shape, reference_sample, batch)


class Normalizer(eqx.Module):
    """A container for running statistics on Observations and Rewards."""

    obs: RunningStatisticsState | None
    reward: RunningStatisticsState | None

    returns: Optional[Array] = None  # for (discounted) reward normalization
    gamma: Optional[float] = None

    def __init__(
        self,
        dummy_obs: PyTree | None,
        *,
        normalize_obs: bool = True,
        normalize_rew: bool = True,
        gamma: float | None = 0.99,
        rew_shape: Tuple[int, ...] | None = (1,),
    ):
        self.obs = None
        self.reward = None

        if normalize_obs:
            assert dummy_obs is not None, (
                "When normalizing observations, a dummy observation must be provided."
            )
            if isinstance(dummy_obs, jym.AgentObservation):
                dummy_obs = dummy_obs.observation
            self.obs = RunningStatisticsState(dummy_obs)

        if normalize_rew:
            self.reward = RunningStatisticsState(jnp.zeros(()), center_mean=False)

        self.returns = None
        self.gamma = None
        if normalize_rew:
            assert gamma is not None and rew_shape is not None, (
                "Normalizer must be initialized with gamma and rew_shape when normalizing rewards."
            )
            self.returns = jnp.zeros(rew_shape, dtype=jnp.float32)
            self.gamma = gamma

    def update_obs(self, obs: PyTree) -> "Normalizer":
        if self.obs is None:
            return self
        if isinstance(obs, jym.AgentObservation):
            obs = obs.observation
        return eqx.tree_at(lambda x: x.obs, self, self.obs.update(obs))

    def update_reward(self, reward: Array, done: Array) -> "Normalizer":
        if self.reward is None:
            return self

        assert self.gamma is not None and self.returns is not None, (
            "Normalizer must be initialized with gamma and returns before updating rewards."
        )

        new_returns = self.returns * self.gamma + reward
        reward_normalizer = self.reward.update(new_returns)

        # If done is True, we reset the returns to zero for the next iteration.
        new_returns = jax.tree.map(lambda x: x * (1.0 - done), new_returns)

        return eqx.tree_at(
            lambda x: (x.reward, x.returns), self, (reward_normalizer, new_returns)
        )

    def update(self, batch: Transition) -> "Normalizer":
        """Updates the normalization state with a new Transition containing both rewards and observations."""
        _self = self.update_obs(batch.observation)
        done = jnp.logical_or(batch.terminated, batch.truncated)
        _self = _self.update_reward(batch.reward, done)
        return _self

    def normalize_obs(self, obs: PyTree) -> PyTree:
        """Normalizes the given batch of observations if normalization of observations is enabled."""
        if self.obs is None:
            return obs
        if isinstance(obs, jym.AgentObservation):
            return jym.AgentObservation(
                observation=self.obs.normalize(obs.observation),
                action_mask=obs.action_mask,
            )
        return self.obs.normalize(obs)

    def normalize_reward(self, reward: Array) -> Array:
        """Normalizes the given batch of rewards if normalization of rewards is enabled."""
        if self.reward is None:
            return reward
        return self.reward.normalize(reward)
