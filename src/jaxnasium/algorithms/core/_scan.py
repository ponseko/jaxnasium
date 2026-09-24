import warnings
from collections.abc import Callable
from typing import TypeVar

import jax
from jaxtyping import PRNGKeyArray

from ._transition import Transition

Carry = TypeVar("Carry")
Y = TypeVar("Y")


def scan_minibatch_epoch(
    f: Callable[[Carry, Transition], tuple[Carry, Y]],
    init: Carry,
    xs_batch: Transition,
    *,
    minibatch_rng: PRNGKeyArray,
    num_epochs: int,
    num_minibatches: int,
    unroll: int | bool | tuple[int | bool, int | bool] = (False, False),
):
    """
    A simple double `jax.lax.scan` loop over num_epochs x num_minibatches.

    Each epoch reshuffles the `xs_batch` into `num_minibatches`.
    Follows the same pattern as `jax.lax.scan`, but has a nested loop.

    **Arguments:**

    - `f`: A function of the form `(carry, xs) -> (carry, ys)`.
    - `init`: The initial carry value.
    - `xs_batch`: The batch of inputs to scan over.
    - `key`: A PRNG key for shuffling the data when using epochs or minibatches.
    - `num_epochs`: Number of epochs in the batch.
    - `num_minibatches`: Number of minibatches in the batch.
    - `unroll`: Unroll for both scans. Can be a tuple, where the the first element
        is the outer (epoch) scan and the second element is the inner (minibatch) scan.
    """

    def _inner(carry: Carry, key: PRNGKeyArray):
        minibatches = xs_batch.make_minibatches(key, num_minibatches)
        inner_unroll = unroll if isinstance(unroll, int) else unroll[1]
        return jax.lax.scan(f, carry, minibatches, unroll=inner_unroll)

    keys = jax.random.split(minibatch_rng, num_epochs)
    outer_unroll = unroll if isinstance(unroll, int) else unroll[0]
    return jax.lax.scan(_inner, init, keys, unroll=outer_unroll)


def scan_transitions(
    f: Callable[[Carry, Transition], tuple[Carry, Y]],
    init: Carry,
    xs: Transition,
    reverse: bool = False,
    unroll: int | bool = 1,
    *,
    key: PRNGKeyArray | None = None,
    num_epochs: int = 1,
    num_minibatches: int = 1,
) -> tuple[Carry, Y]:
    """
    A wrapper around jax.lax.scan that works with Transition PyTrees.

    Args:
        f: A function of the form (carry, x) -> (carry, y)
        init: The initial carry value
        xs: A Transition PyTree containing the inputs to scan over
        reverse: Whether to scan in reverse order
        unroll: Unroll factor for the scan
        key: A PRNG key for shuffling the data when using epochs or minibatches
        num_epochs: Number of epochs in the Transition
        num_minibatches: Number of minibatches in the Transition

    Returns:
        A tuple of (final carry, stacked outputs)
    """

    warnings.warn(
        "scan_transitions is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )

    assert num_epochs >= 1, "num_epochs must be at least 1"
    assert num_minibatches >= 1, "num_minibatches must be at least 1"
    if num_epochs == 1 and num_minibatches == 1:
        # Fallback to standard fn call if no epochs or minibatches
        return f(init, xs)

    assert key is not None, (
        "A PRNG key must be provided when using epochs or minibatches."
    )

    def do_epoch(carry: Carry, shuffle_key: PRNGKeyArray) -> tuple[Carry, Y]:
        def do_minibatch(carry: Carry, minibatch: Transition) -> tuple[Carry, Y]:
            return f(carry, minibatch)

        minibatches = xs.make_minibatches(
            key=shuffle_key, n_minibatches=num_minibatches
        )
        return jax.lax.scan(
            do_minibatch, carry, minibatches, reverse=reverse, unroll=unroll
        )

    if num_epochs == 1:
        return do_epoch(init, key)

    keys = jax.random.split(key, num_epochs)
    return jax.lax.scan(do_epoch, init, keys, reverse=reverse, unroll=unroll)
