import logging
from typing import Callable, List, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from jaxnasium import Space

Identity = eqx.nn.Identity

logger = logging.getLogger(__name__)


class MLP(eqx.Module):
    """Simple MLP architecture."""

    layers: List[eqx.nn.Linear]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    hidden_sizes: Sequence[int] = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_features: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: Callable = jax.nn.relu,
        **kwargs,
    ):
        depth = len(hidden_sizes) + 1
        keys = jax.random.split(key, depth + 1)
        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.out_features = hidden_sizes[-1]
        self.activation = activation

        self.layers = []
        for i, hidden_dim in enumerate(hidden_sizes):
            self.layers.append(
                eqx.nn.Linear(
                    in_features=in_features, out_features=hidden_dim, key=keys[i]
                )
            )
            in_features = hidden_dim

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class CNN(eqx.Module):
    """Standard CNN architecture similar to the DQN Nature paper.

    Operates on 2D inputs.
    Assumes channels first format (C, H, W).
    """

    layers: List[eqx.nn.Conv2d]
    in_channels: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    channels_axis: int | None = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: Space,
        hidden_sizes: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        padding: Sequence[int],
        **kwargs,
    ):
        assert len(hidden_sizes) == len(kernel_sizes) == len(strides) == len(padding)

        if len(obs_space.shape) == 2:
            logger.warning(
                "2D input without channels, adding leading channels in __call__()"
                "In case the observation should be treated as 1d, use a FlattenObservationWrapper."
            )
            self.channels_axis = None
            in_channels = 1
        elif (
            obs_space.shape[0] == obs_space.shape[1]
            and obs_space.shape[2] != obs_space.shape[0]
        ):
            logger.warning(
                "2D input is in channels last format, moving channels to first dimension"
                "Prefer providing channels first observations (C, H, W)."
            )
            self.channels_axis = -1
            in_channels = obs_space.shape[self.channels_axis]
        else:  # channels first
            self.channels_axis = 0
            in_channels = obs_space.shape[self.channels_axis]

        self.in_channels = in_channels

        self.layers = []
        keys = jax.random.split(key, len(hidden_sizes))

        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(
                eqx.nn.Conv2d(
                    in_channels,
                    hidden_size,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=padding[i],
                    key=keys[i],
                )
            )
            in_channels = hidden_size

        out_shape = jax.eval_shape(
            lambda x: self(x), jnp.zeros(obs_space.shape, dtype=jnp.float32)
        ).shape
        assert len(out_shape) == 1 and out_shape[0] > 0, (
            f"Invalid CNN output (after flattening): {out_shape}. "
            "Perhaps the observation space shape is too small for the CNN architecture."
        )
        self.out_features = out_shape[0]

    def __call__(self, x):
        if self.channels_axis is None:
            x = jnp.expand_dims(x, axis=0)
        elif self.channels_axis == -1:
            x = jnp.moveaxis(x, -1, 0)

        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        x = jnp.reshape(x, -1)
        return x


class BroNet(eqx.Module):
    """
    Create a BroNet neural network with the given hidden dimensions and output space.
    https://arxiv.org/html/2405.16158v1

    Operates on 1D inputs.
    """

    layers: List[eqx.Module]
    in_features: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self, key: PRNGKeyArray, in_features: int, depth: int, width_size: int, **kwargs
    ):
        class BroNetBlock(eqx.Module):
            layers: list
            in_features: int = eqx.field(static=True)
            out_features: int = eqx.field(static=True)

            def __init__(self, key: PRNGKeyArray, shape: int):
                key1, key2 = jax.random.split(key)
                self.layers = [
                    eqx.nn.Linear(in_features=shape, out_features=shape, key=key1),
                    eqx.nn.LayerNorm(shape),
                    eqx.nn.Linear(in_features=shape, out_features=shape, key=key2),
                    eqx.nn.LayerNorm(shape),
                ]
                self.in_features = shape
                self.out_features = shape

            def __call__(self, x):
                _x = self.layers[0](x)
                _x = self.layers[1](_x)
                _x = jax.nn.relu(_x)
                _x = self.layers[2](_x)
                _x = self.layers[3](_x)
                return x + _x

        keys = jax.random.split(key, depth + 1)
        self.in_features = in_features
        self.width_size = width_size
        self.depth = depth
        self.out_size = width_size
        self.layers = [
            eqx.nn.Linear(
                in_features=in_features, out_features=width_size, key=keys[0]
            ),
            eqx.nn.LayerNorm(width_size),
        ]
        for i in range(1, depth + 1):
            self.layers.append(BroNetBlock(keys[i], width_size))
