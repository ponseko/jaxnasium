import logging
from collections.abc import Callable, Sequence
from typing import Literal, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

logger = logging.getLogger(__name__)


class CNN(eqx.Module):
    """Standard CNN architecture similar to the DQN Nature paper.

    Operates on 2D inputs.
    Assumes channels first format (C, H, W).

    Flattens the output of the CNN into a 1d vector.
    """

    layers: list[eqx.nn.Conv2d]
    in_channels: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    channels_axis: Literal["first", "last"] = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        *,
        key: PRNGKeyArray,
        out_channels: Sequence[int] = (32, 64, 64),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        strides: Sequence[int] = (1, 1, 1),
        padding: Sequence[int] = (1, 1, 1),
        channels_axis: Literal["first", "last"] = "first",
        activation: Callable = jax.nn.relu,
    ):
        assert len(out_channels) == len(kernel_sizes) == len(strides) == len(padding), (
            f"Lengths of out_channels, kernel_sizes, strides, and padding must match. "
            f"Got {len(out_channels)}, {len(kernel_sizes)}, {len(strides)}, and {len(padding)}."
            f"This CNN implementation assumes _per layer_ ints for out_channels, kernel_sizes, strides, and padding."
        )

        self.channels_axis = channels_axis
        self.activation = activation

        if channels_axis == "last":
            logger.warning(
                "2D input is in channels last format, moving channels to first dimension. "
                "Prefer providing channels first observations (C, H, W)."
            )

        if self.channels_axis == "first":
            in_channels = input_shape[0]
        elif self.channels_axis == "last":
            in_channels = input_shape[-1]
        else:
            raise ValueError(f"Invalid channels axis: {self.channels_axis}")

        self.in_channels = in_channels

        self.layers = []
        keys = jax.random.split(key, len(out_channels))
        for i, out_channel in enumerate(out_channels):
            self.layers.append(
                eqx.nn.Conv2d(
                    in_channels,
                    out_channel,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=padding[i],
                    key=keys[i],
                )
            )
            in_channels = out_channel

        out_shape = jax.eval_shape(
            lambda x: self(x), jnp.zeros(input_shape, dtype=jnp.float32)
        ).shape
        assert len(out_shape) == 1 and out_shape[0] > 0, (
            f"Invalid CNN output (after flattening): {out_shape}. "
            "Perhaps the observation space shape is too small for the CNN architecture."
        )
        self.out_features = out_shape[0]

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        if self.channels_axis == "last":
            x = jnp.moveaxis(x, -1, 0)

        for layer in self.layers:
            x = self.activation(layer(x, key=key))
        x = jnp.reshape(x, -1)
        return x

    @classmethod
    def with_params(
        cls,
        *,
        out_channels: Sequence[int] = (32, 64, 64),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        strides: Sequence[int] = (1, 1, 1),
        padding: Sequence[int] = (1, 1, 1),
        channels_axis: Literal["first", "last"] | None = None,
        activation: Callable = jax.nn.relu,
    ) -> Callable[..., Self]:
        kwargs = {
            "out_channels": out_channels,
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "padding": padding,
            "activation": activation,
        }
        if channels_axis is not None:
            kwargs["channels_axis"] = channels_axis
        return eqx.Partial(cls, **kwargs)
