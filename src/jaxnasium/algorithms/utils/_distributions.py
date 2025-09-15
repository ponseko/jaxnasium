from functools import partial
from typing import Callable

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def _result_tuple_to_tuple_result(r):
    """
    Some functions may return tuples. Rather than returning
    a pytree of tuples, we convert it to a tuple of pytrees.
    """
    one_level_leaves, structure = eqx.tree_flatten_one_level(r)
    if isinstance(one_level_leaves[0], tuple):
        tupled = tuple([list(x) for x in zip(*one_level_leaves)])
        r = tuple(jax.tree.unflatten(structure, leaves) for leaves in tupled)
    return r


class DistraxContainer(eqx.Module):
    """Container for (possibly nested as PyTrees) distrax distributions."""

    distribution: distrax.Distribution | PyTree[distrax.Distribution]

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
            res = jax.tree.map(
                lambda dist: getattr(dist, name)(*args, **kwargs),
                self.distribution,
                is_leaf=lambda x: isinstance(x, distrax.Distribution),
            )
            return _result_tuple_to_tuple_result(res)

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
        return _result_tuple_to_tuple_result(res)

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
    def __init__(self, mean, std, shift=0.0, scale=1.0):
        dist = distrax.Normal(loc=mean, scale=std)
        tanh = distrax.Tanh()
        scaler = distrax.ScalarAffine(shift=shift, scale=scale)
        super().__init__(dist, distrax.Chain([tanh, scaler]))
        self._mean = mean
        self._std = std
        self._shift = shift
        self._scale = scale

    def mode(self):
        return self._shift + self._scale * jnp.tanh(self._mean)


def TanhNormalFactory(low, high) -> Callable[..., TanhNormal]:
    scale = (high - low) / 2.0
    shift = (high + low) / 2.0

    return partial(TanhNormal, shift=shift, scale=scale)
