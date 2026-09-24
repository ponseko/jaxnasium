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
    variance: Array

    max_count: int | None = eqx.field(static=True)
    std_min_value: float = eqx.field(static=True)
    std_max_value: float = eqx.field(static=True)

    def __init__(
        self,
        pytree_example,
        max_count: int | None = None,
        std_min_value: float = 1e-4,
        std_max_value: float = 1e6,
    ):
        dtype: type = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32  # type: ignore
        count_dtype: type = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32  # type: ignore

        self.count = jnp.zeros((), dtype=count_dtype)
        self.mean = optax.tree_utils.tree_zeros_like(pytree_example, dtype=dtype)
        self.variance = optax.tree_utils.tree_zeros_like(pytree_example, dtype=dtype)
        self.std = optax.tree_utils.tree_ones_like(pytree_example, dtype=dtype)

        # Caps the sample count. None (original behavior and default) is an all-history average.
        # setting max count turns this into an EMA with an horizon of `max_count`.
        if max_count is not None and max_count < 1:
            raise ValueError(f"`max_count` must be at least 1, got {max_count}.")
        self.max_count = max_count

        if not 0.0 < std_min_value < std_max_value:
            raise ValueError(
                "Require 0 < std_min_value < std_max_value, got "
                f"{std_min_value}, {std_max_value}."
            )
        self.std_min_value = std_min_value
        self.std_max_value = std_max_value

    def update(
        self,
        batch: Array,
        *,
        mask: jnp.ndarray | None = None,
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
            mean: jnp.ndarray, variance: jnp.ndarray, batch: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            assert isinstance(mean, jnp.ndarray), type(mean)
            assert isinstance(variance, jnp.ndarray), type(variance)
            # The mean and the sum of past variances are updated with Welford's
            # algorithm using batches (see https://stackoverflow.com/q/56402955).
            # However, we now carry the variance rather than the summed variance to allow
            # us to floor the count
            weight = (
                True
                if mask is None
                else mask.reshape(mask.shape + (1,) * (batch.ndim - mask.ndim))
            )
            diff_to_old_mean = batch - mean

            mean_update = jnp.sum(diff_to_old_mean * weight, axis=batch_axis) / denom
            mean = mean + mean_update

            diff_to_new_mean = batch - mean
            variance_update = jnp.sum(
                diff_to_old_mean * diff_to_new_mean * weight, axis=batch_axis
            )
            variance = variance + (variance_update - step_increment * variance) / denom
            return mean, variance

        def compute_std(variance: jnp.ndarray) -> jnp.ndarray:
            assert isinstance(variance, jnp.ndarray)
            variance = jnp.maximum(
                variance, 0
            )  # in case var < 0 due to rounding errors.
            std = jnp.sqrt(variance)
            std = jnp.clip(std, self.std_min_value, self.std_max_value)
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
        if mask is None:
            step_increment = jnp.prod(jnp.array(batch_dims))
        else:
            assert mask.ndim == len(batch_dims) and mask.dtype == jnp.bool_, (
                f"`mask` must be a boolean mask (got dtype {mask.dtype}) and "
                f"must have one entry per sample, i.e. shape {batch_dims}, "
                f"got {mask.shape}."
            )
            step_increment = jnp.sum(mask)
        count = self.count + step_increment.astype(self.count.dtype)
        if self.max_count is not None:
            count = jnp.minimum(count, self.max_count).astype(self.count.dtype)
        # if all masks are 0 on the first update, denom becomes 0.
        denom = jnp.where(count > 0, count, 1)
        denom = jnp.maximum(denom, step_increment.astype(denom.dtype))

        # # Validation is important. If the shapes don't match exactly, but are
        # # compatible, arrays will be silently broadcasted resulting in incorrect
        # # statistics.
        if validate_shapes:
            self._validate_batch_shapes(batch, self.mean, batch_dims)

        updated_stats = jax.tree.map(
            _compute_node_statistics, self.mean, self.variance, batch
        )
        mean = jax.tree.map(lambda _, x: x[0], self.mean, updated_stats)
        variance = jax.tree.map(lambda _, x: x[1], self.mean, updated_stats)
        std = jax.tree.map(compute_std, variance)

        return eqx.tree_at(
            lambda x: (x.mean, x.std, x.count, x.variance),
            self,
            (mean, std, count, variance),
            is_leaf=lambda x: x is None,
        )

    @staticmethod
    def _validate_batch_shapes(
        batch: PyTree,
        reference_sample: PyTree,
        batch_dims: tuple[int, ...],
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

    returns: Array | None  # for (discounted) reward normalization
    returns_max: Array | None  # the running max discounted return
    gamma: float | None
    g_max: float | None = eqx.field(static=True)
    center_mean_obs: bool = eqx.field(static=True)
    center_mean_rew: bool = eqx.field(static=True)
    clip_value_obs: float | None = eqx.field(static=True)
    clip_value_rew: float | None = eqx.field(static=True)

    def __init__(
        self,
        dummy_obs: PyTree | None = None,
        *,
        obs_space: PyTree[jym.Space] | None = None,
        normalize_obs: bool = True,
        normalize_rew: bool = True,
        gamma: float | None = 0.99,
        rew_shape: tuple[int, ...] | None = (1,),
        g_max: float | None = None,
        center_mean_obs: bool = True,
        center_mean_rew: bool = False,
        clip_value_obs: float | None = 100.0,
        clip_value_rew: float | None = 100.0,
    ):
        """
        **Arguments:**

        - `dummy_obs`: a representative observation to initialize the running statistics.
        - `obs_space`: an observation space to sample a representative observation from.
            Dummy observation or obs_space must be provided if `normalize_obs` is True.
        - `normalize_obs`: whether to normalize observations.
        - `normalize_rew`: whether to normalize rewards.
        - `gamma`: discount factor for computing discounted returns. Must be provided if `normalize_rew` is True.
        - `rew_shape`: shape of a **single env step's** reward, i.e. `(num_envs,)`.
        - `g_max`: if not None, bounds the normalized returns between ±g_max (SimBaV2 trick).
        - `center_mean_obs`: subtract the running observation mean from the observations.
        - `center_mean_rew`: subtract the running return mean from the rewards.
        - `clip_value_obs` / `clip_value_rew`: when not None, clips normalized values to `±clip_value`.
        """
        self.obs = None
        self.reward = None
        self.center_mean_obs = center_mean_obs
        self.center_mean_rew = center_mean_rew
        self.clip_value_obs = clip_value_obs
        self.clip_value_rew = clip_value_rew

        if normalize_obs:
            assert dummy_obs is not None or obs_space is not None, (
                "When normalizing observations, a dummy observation or observation space must be provided."
            )
            if dummy_obs is None:
                dummy_obs = jax.tree.map(
                    lambda space: space.sample(jax.random.PRNGKey(0)), obs_space
                )
            if isinstance(dummy_obs, jym.AgentObservation):
                dummy_obs = dummy_obs.replace(action_mask=None)
            self.obs = RunningStatisticsState(dummy_obs)

        if normalize_rew:
            self.reward = RunningStatisticsState(jnp.zeros(()))

        self.returns = None
        self.returns_max = None
        self.gamma = None
        self.g_max = None
        if normalize_rew:
            assert gamma is not None and rew_shape is not None, (
                "Normalizer must be initialized with gamma and rew_shape when normalizing rewards."
            )
            if g_max is not None and g_max <= 0:
                raise ValueError(f"`g_max` must be positive, got {g_max}.")
            dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
            self.returns = jnp.zeros(rew_shape, dtype=dtype)
            self.returns_max = jnp.zeros((), dtype=dtype)
            self.gamma = gamma
            self.g_max = g_max

    def update_obs(self, obs: PyTree, mask: Array | None = None) -> "Normalizer":
        if self.obs is None:
            return self
        if isinstance(obs, jym.AgentObservation):
            obs = obs.replace(action_mask=None)
        return eqx.tree_at(
            lambda x: x.obs,
            self,
            self.obs.update(obs, mask=mask),
            is_leaf=lambda x: x is None,
        )

    def update_reward(self, reward: Array, done: Array) -> "Normalizer":
        if self.reward is None:
            return self

        assert self.gamma is not None and self.returns is not None, (
            "Normalizer must be initialized with gamma and returns before updating rewards."
        )

        if jnp.shape(reward) == jnp.shape(self.returns):
            # Single un-stacked step; add a leading time axis of length 1.
            reward, done = reward[None], done[None]

        def _accumulate(returns, step):
            reward, done = step
            returns = returns * self.gamma + reward
            # Emit the pre-reset return, then reset where the episode ended.
            return returns * (1.0 - done), returns

        new_returns, discounted_returns = jax.lax.scan(
            _accumulate, self.returns, (reward, done)
        )
        reward_normalizer = self.reward.update(discounted_returns)

        assert self.returns_max is not None
        new_returns_max = jnp.maximum(
            self.returns_max, jnp.max(jnp.abs(discounted_returns))
        )

        return eqx.tree_at(
            lambda x: (x.reward, x.returns, x.returns_max),
            self,
            (reward_normalizer, new_returns, new_returns_max),
            is_leaf=lambda x: x is None,
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

        def _normalize(
            batch: PyTree[Array], mean: PyTree[Array], std: PyTree[Array]
        ) -> PyTree[Array]:
            if self.center_mean_obs:
                batch = optax.tree.sub(batch, mean)
            normalized = jax.tree.map(lambda data, s: data / (s + 1e-8), batch, std)
            if self.clip_value_obs is None:
                return normalized
            return jym.tree.clip(normalized, -self.clip_value_obs, self.clip_value_obs)

        if isinstance(obs, jym.AgentObservation):
            action_mask = obs.action_mask
            obs = obs.replace(action_mask=None)
            normalized = _normalize(obs, self.obs.mean, self.obs.std)
            return normalized.replace(action_mask=action_mask)
        return _normalize(obs, self.obs.mean, self.obs.std)

    def normalize_reward(self, reward: Array) -> Array:
        """Normalizes the given batch of rewards if normalization of rewards is enabled."""
        if self.reward is None:
            return reward

        std = self.reward.std
        if self.g_max is not None:
            assert self.returns_max is not None
            std = jnp.maximum(std, self.returns_max / self.g_max)

        if self.center_mean_rew:
            reward = optax.tree.sub(reward, self.reward.mean)

        normalized = jax.tree.map(lambda data, s: data / (s + 1e-8), reward, std)
        if self.clip_value_rew is None:
            return normalized
        return jym.tree.clip(normalized, -self.clip_value_rew, self.clip_value_rew)
