import logging
from collections.abc import Callable, Sequence
from typing import Self

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

logger = logging.getLogger(__name__)


class MLP(eqx.Module):
    """Simple MLP architecture. Final hidden size is the output features.
    Final outputs are non-linear (unless the activation is set to identity).
    Attach additional layers if a different final activation is desired.
    """

    layers: list[eqx.nn.Linear]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    hidden_sizes: Sequence[int] = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        *,
        key: PRNGKeyArray,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: Callable = jax.nn.relu,
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

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        for layer in self.layers:
            x = self.activation(layer(x, key=key))
        return x

    @classmethod
    def with_params(
        cls,
        *,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: Callable = jax.nn.relu,
    ) -> Callable[..., Self]:
        return eqx.Partial(cls, hidden_sizes=hidden_sizes, activation=activation)
