from collections.abc import Callable, Sequence
from typing import Any, Protocol

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class SpaceLike(Protocol):
    shape: tuple[int, ...]
    sample: Callable[[PRNGKeyArray], Array]
    dtype: jnp.dtype


class DiscreteSpaceLike(SpaceLike, Protocol):
    n: int | None = None
    nvec: Sequence[int] | None = None


class ContinuousSpaceLike(SpaceLike, Protocol):
    low: Array
    high: Array


class Network(Protocol):
    """Any module with a __call__ defined."""

    def __call__(self, *args, **kwargs) -> Any: ...


class OutSizedNetwork(Network, Protocol):
    """Any module with a __call__ defined and an out_features attribute."""

    out_features: int
