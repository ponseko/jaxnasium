import logging
import warnings
from collections.abc import Callable
from typing import Any

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

logger = logging.getLogger(__name__)


def make_independent(dist: distrax.Distribution) -> distrax.Distribution:
    """Wraps a distrax distribution in an Independent distribution if the
    output space is multi-dimensional and sets the event shape accordingly."""
    ndims = len(dist.batch_shape)
    if ndims == 0:
        return dist  # Discrete, MultiDiscrete([n]), scalar Box
    return distrax.Independent(dist, reinterpreted_batch_ndims=ndims)


def _transpose_tree_of_tuples(r, outer_treedef):
    """
    Some functions may return tuples. Rather than returning
    a pytree of tuples, we convert it to a tuple of pytrees
    using jax.tree.transpose.
    """
    flat = outer_treedef.flatten_up_to(r)
    if not flat or not isinstance(flat[0], tuple):
        return r
    inner_treedef = jax.tree.structure(tuple(range(len(flat[0]))))
    return jax.tree.transpose(outer_treedef, inner_treedef, r)


class DistraxContainer(eqx.Module):
    """Container for (possibly nested as PyTrees) distrax distributions."""

    distribution: distrax.Distribution | PyTree[distrax.Distribution]

    def __check_init__(self):
        warnings.warn(
            "DistraxContainer is deprecated in favor of "
            "distrax.Joints combined with distrax.Independent.",
            DeprecationWarning,
            stacklevel=2,
        )

    def __getattr__(self, name):
        if isinstance(self.distribution, distrax.Distribution):
            return getattr(self.distribution, name)

        # Check if the attribute is callable
        ref = jax.tree.leaves(
            self.distribution, is_leaf=lambda x: isinstance(x, distrax.Distribution)
        )[0]
        if not callable(getattr(ref, name)):
            return jax.tree.map(
                lambda x: getattr(x, name),
                self.distribution,
                is_leaf=lambda x: isinstance(x, distrax.Distribution),
            )

        # If callable, return a method that calls the attribute on each distribution
        def method_caller(*args, **kwargs):
            outer_treedef = jax.tree.structure(
                self.distribution,
                is_leaf=lambda x: isinstance(x, distrax.Distribution),
            )
            res = jax.tree.map(
                lambda dist: getattr(dist, name)(*args, **kwargs),
                self.distribution,
                is_leaf=lambda x: isinstance(x, distrax.Distribution),
            )
            return _transpose_tree_of_tuples(res, outer_treedef)

        return method_caller

    def sample(self, *, seed):
        if isinstance(self.distribution, distrax.Distribution):
            return self.distribution.sample(seed=seed)

        structure = jax.tree.structure(
            self.distribution, is_leaf=lambda x: isinstance(x, distrax.Distribution)
        )
        seeds = jax.random.split(seed, structure.num_leaves)
        seeds = jax.tree.unflatten(structure, seeds)
        return jax.tree.map(
            lambda dist, key: dist.sample(seed=key),
            self.distribution,
            seeds,
            is_leaf=lambda x: isinstance(x, distrax.Distribution),
        )

    def sample_and_log_prob(self, *, seed):
        if isinstance(self.distribution, distrax.Distribution):
            return self.distribution.sample_and_log_prob(seed=seed)

        structure = jax.tree.structure(
            self.distribution, is_leaf=lambda x: isinstance(x, distrax.Distribution)
        )
        seeds = jax.random.split(seed, structure.num_leaves)
        seeds = jax.tree.unflatten(structure, seeds)
        res = jax.tree.map(
            lambda dist, key: dist.sample_and_log_prob(seed=key),
            self.distribution,
            seeds,
            is_leaf=lambda x: isinstance(x, distrax.Distribution),
        )
        return _transpose_tree_of_tuples(res, structure)

    def log_prob(self, value):
        if isinstance(self.distribution, distrax.Distribution):
            return self.distribution.log_prob(value)

        return jax.tree.map(
            lambda dist, v: dist.log_prob(v),
            self.distribution,
            value,
            is_leaf=lambda x: isinstance(x, distrax.Distribution),
        )


class TanhNormal(distrax.Transformed):
    """Normal squashed through `tanh`, affinely rescaled to `[low, high]`.

    NOTE: obtain log-probs with `sample_and_log_prob`, instead of `sample` followed by `log_prob`.

    https://github.com/google-deepmind/distrax/issues/7
    https://github.com/google-deepmind/distrax/issues/216
    """

    def __init__(
        self, mean, std, shift: float | Array = 0.0, scale: float | Array = 1.0
    ):
        mean = jnp.asarray(mean)
        std = jnp.asarray(std)
        target_shape = jnp.shape(mean)
        shift = jnp.broadcast_to(jnp.asarray(shift), target_shape)
        scale = jnp.broadcast_to(jnp.asarray(scale), target_shape)

        dist = distrax.Normal(loc=mean, scale=std)
        tanh = distrax.Tanh()
        scaler = distrax.ScalarAffine(shift=shift, scale=scale)
        super().__init__(dist, distrax.Chain([scaler, tanh]))
        self._mean = mean
        self._std = std
        self._shift = shift
        self._scale = scale

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    def mode(self):
        return self._shift + self._scale * jnp.tanh(self._mean)


def TanhNormalFactory(low, high) -> Callable[..., TanhNormal]:
    scale = (high - low) / 2.0
    shift = (high + low) / 2.0

    return eqx.Partial(TanhNormal, shift=shift, scale=scale)


def _masked_epsilon_greedy(
    preferences: Array,
    epsilon: float,
    action_mask: Array,
    dtype: jnp.dtype | type[Any],
) -> distrax.Categorical:
    num_actions = preferences.shape[-1]
    mask = jnp.asarray(action_mask, dtype=bool)
    num_valid = jnp.count_nonzero(mask, axis=-1, keepdims=True)

    uniform = jnp.where(
        num_valid > 0,
        mask.astype(preferences.dtype) / jnp.maximum(num_valid, 1),
        1.0 / num_actions,
    )

    masked_preferences = jnp.where(mask, preferences, -jnp.inf)
    # splitting ties
    optimal = masked_preferences == masked_preferences.max(axis=-1, keepdims=True)
    greedy = optimal / optimal.sum(axis=-1, keepdims=True)

    probs = (1 - epsilon) * greedy + epsilon * uniform
    return distrax.Categorical(probs=probs, dtype=dtype)


class EpsilonGreedy(distrax.Joint):
    """Like `distrax.EpsilonGreedy` but wrapped in a `Joint` and `Independent`
    for PyTree support. Assumes independent actions.
    Also supports action masking.
    """

    def __init__(
        self,
        preference,
        *,
        epsilon: float = 0.0,
        action_mask: PyTree | None = None,
        dtype: jnp.dtype | type[Any] = int,
    ):
        if action_mask is None:
            distributions = jax.tree.map(
                lambda dist: make_independent(
                    distrax.EpsilonGreedy(dist, epsilon=epsilon, dtype=dtype)
                ),
                preference,
            )

        else:
            distributions = jax.tree.map(
                lambda dist, mask: make_independent(
                    _masked_epsilon_greedy(dist, epsilon, mask, dtype)
                ),
                preference,
                action_mask,
            )

        super().__init__(distributions)
