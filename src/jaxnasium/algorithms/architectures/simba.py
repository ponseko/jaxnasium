import logging
from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

logger = logging.getLogger(__name__)


class _SimBaBlock(eqx.Module):
    """With 'inverted bottleneck' (4x)."""

    layer_norm: eqx.nn.LayerNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, shape: int, *, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key)
        self.layer_norm = eqx.nn.LayerNorm(shape)
        self.linear1 = eqx.nn.Linear(
            in_features=shape, out_features=shape * 4, key=key1
        )
        self.linear2 = eqx.nn.Linear(
            in_features=shape * 4, out_features=shape, key=key2
        )
        self.in_features = shape
        self.out_features = shape

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        _x = self.layer_norm(x)
        _x = self.linear1(_x, key=key)
        _x = jax.nn.relu(_x)
        _x = self.linear2(_x, key=key)
        return x + _x


class SimBa(eqx.Module):
    """
    Create a Simba neural network with the given hidden dimensions and output space.
    https://arxiv.org/pdf/2410.09754

    Operates on 1D inputs.
    Observation normalization (RSNorm) is left to the agent (enable `normalize_observations`).

    depth is the number of blocks.
    """

    embedding: eqx.nn.Linear
    blocks: list[_SimBaBlock]
    post_layer_norm: eqx.nn.LayerNorm
    in_features: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        *,
        key: PRNGKeyArray,
        depth: int = 1,
        width_size: int = 256,
    ):
        keys = jax.random.split(key, depth + 1)
        self.in_features = in_features
        self.width_size = width_size
        self.depth = depth
        self.out_features = width_size
        self.embedding = eqx.nn.Linear(
            in_features=in_features, out_features=width_size, key=keys[0]
        )
        self.blocks = [
            _SimBaBlock(width_size, key=keys[i]) for i in range(1, depth + 1)
        ]
        self.post_layer_norm = eqx.nn.LayerNorm(width_size)

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        x = self.embedding(x, key=key)
        for block in self.blocks:
            x = block(x, key=key)
        return self.post_layer_norm(x)

    @classmethod
    def with_params(
        cls, *, depth: int = 1, width_size: int = 256
    ) -> Callable[..., Self]:
        return eqx.Partial(cls, depth=depth, width_size=width_size)
