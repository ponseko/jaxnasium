import logging
from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

logger = logging.getLogger(__name__)


class _BroNetBlock(eqx.Module):
    layers: list
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, shape: int, *, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(in_features=shape, out_features=shape, key=key1),
            eqx.nn.LayerNorm(shape),
            eqx.nn.Linear(in_features=shape, out_features=shape, key=key2),
            eqx.nn.LayerNorm(shape),
        ]
        self.in_features = shape
        self.out_features = shape

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        _x = self.layers[0](x, key=key)
        _x = self.layers[1](_x, key=key)
        _x = jax.nn.relu(_x)
        _x = self.layers[2](_x, key=key)
        _x = self.layers[3](_x, key=key)
        return x + _x


class BroNet(eqx.Module):
    """
    Create a BroNet neural network with the given hidden dimensions and output space.
    https://arxiv.org/html/2405.16158v1

    Operates on 1D inputs.
    """

    layers: list
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
        self.layers = [
            eqx.nn.Linear(
                in_features=in_features, out_features=width_size, key=keys[0]
            ),
            eqx.nn.LayerNorm(width_size),
        ]
        for i in range(1, depth + 1):
            self.layers.append(_BroNetBlock(width_size, key=keys[i]))

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        x = self.layers[0](x, key=key)  # dense
        x = self.layers[1](x, key=key)  # layernorm
        x = jax.nn.relu(x)
        # then the bronet blocks:
        for block in self.layers[2:]:
            x = block(x, key=key)
        return x

    @classmethod
    def with_params(
        cls, *, depth: int = 1, width_size: int = 256
    ) -> Callable[..., Self]:
        return eqx.Partial(cls, depth=depth, width_size=width_size)
