import operator
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, PyTreeDef

""" 
Convenience pytree functions used in the various RL algorithms which 
aren't found in used higher-level libraries (equinox / jax).
"""


def _tree_size(tree):
    r"""Get the total number of elements (size of each leaf) in a pytree.
    Ported from: https://github.com/google-deepmind/optax/pull/1321/files/cadb2bca89e2af6af0e70cf0007080d5f68794a4
    """
    return sum([jnp.size(leaf) for leaf in jax.tree.leaves(tree)])


def _tree_sum(tree: Any, axis: Optional[int | tuple[int, ...]] = None) -> Array:
    """
    Compute the sum of all the elements in a pytree
    If axis is provided, sums each leaf over the specified axis and
    then adds adds the resulting leafs.
    """
    sums = jax.tree.map(lambda x: jnp.sum(x, axis=axis), tree)
    return jax.tree.reduce(operator.add, sums, initializer=0)


def _is_child_of(root: PyTree) -> Callable[[PyTree], bool]:
    """`is_leaf` operator for pytree operations useful when the desired operation
    should apply on the first-level children of a pytree.

    **Example**:
    ```python
    >>> jax.tree.map(f, tree, *rest, is_leaf=_is_child_of(tree))
    ```
    """
    return lambda x: x is not root


def tree_mean(tree):
    """Computes the global mean of the leaves of a pytree."""
    sum = _tree_sum(tree)
    size = _tree_size(tree)
    return sum / size


def tree_map_one_level(fn: Callable, tree, *rest):
    """Simple `jax.tree.map` operation over the first level of a pytree.

    **Arguments:**

    - `fn`: A function to map over the pytree.
    - `tree`: A pytree.
    - `*rest`: Additional pytrees to map over.
    """
    return jax.tree.map(fn, tree, *rest, is_leaf=_is_child_of(tree))


def tree_map_distribution(fn: Callable, tree, *rest):
    """Map a function with `distrax.Distribution` instances marked as leaves.
    Additionally, if one of the inputs is a `DistraxContainer`, the function
    is applied to the `distribution` attribute of the `DistraxContainer`.

    **Arguments**:

    - `fn`: A function to map over the pytree.
    - `tree`: A pytree.
    - `*rest`: Additional pytrees to map over.
    """
    try:
        import distrax

        from jaxnasium.algorithms.utils import DistraxContainer
    except ImportError:
        raise ImportError(
            "jaxnasium.algorithms is required for `jaxnasium.tree.map_distributions()`. Please install  `pip install jaxnasium[algs]`."
        )

    if isinstance(tree, DistraxContainer):
        tree = tree.distribution
    # any of *rest should also be converted:
    rest = tuple(r.distribution if isinstance(r, DistraxContainer) else r for r in rest)

    return jax.tree.map(
        fn, tree, *rest, is_leaf=lambda x: isinstance(x, distrax.Distribution)
    )


def tree_concatenate(trees: PyTree) -> Array:
    """Concatenate the leaves of a pytree into a single 1D array.

        **Arguments**:

        - `trees`: A pytree whose leaves are array-like and all 1d or 0d.

        **Returns**: A 1D array containing the concatenated leaves of the pytree.

        **Example**:
    ```python
        >>> tree = {'a': jnp.array([1, 2]), 'b': jnp.array(3)}
        >>> tree_concatenate(tree)
        Array([1, 2, 3], dtype=int32)
    ```
    """
    trees = jax.tree.map(jnp.atleast_1d, trees)
    leaves = jax.tree.leaves(trees)
    return jnp.concatenate(leaves)


def tree_get_first(tree: PyTree, key: str) -> Any:
    """Get the first value from a pytree with the given key.
    Like `optax.tree.get()` but returns the first value found in case
    of multiple matches instead of raising an error.

    **Arguments**:

    - `tree`: A pytree.
    - `key`: A string key.

    **Returns**:
        The first value from the pytree with the given key.

    **Raises**:
        KeyError: If the key is not found in the pytree.
    """
    try:
        import optax
    except ImportError:
        raise ImportError(
            "optax is (for now) required for `jaxnasium.tree.get_first()`. Please install optax with `pip install optax`."
        )
    found_values_with_path = optax.tree.get_all_with_path(tree, key)
    if not found_values_with_path:
        raise KeyError(f"Key '{key}' not found in tree: {tree}.")
    return found_values_with_path[0][1]


def tree_stack(pytrees: PyTree, *, axis=0) -> PyTree:
    """Stack corresponding leaves of pytrees along the specified axis.

    Interprets the root node's immediate children as a batch of N pytrees that all
    share the same structure. For each leaf, stacks the N leaves along `axis` using
    jnp.stack. This does not traverse deeper than one level when determining what to stack.

    **Arguments**:

    - `pytrees`: A pytree whose root has N immediate children. Each child must have
        the same pytree structure. Corresponding leaves must be array-like and
        have identical shapes and dtypes (compatible with jnp.stack).
    - `axis`: Axis along which to insert the new dimension of size N in each stacked leaf (default=0).

    **Returns**:
        A pytree with the same structure as a single direct-child element of `pytrees`, where each
        leaf is the stack of the corresponding leaves across all elements, with a new
        dimension of size N inserted at `axis`.

    **Example**:
    ```python
        >>> trees = (
        ...     [jnp.array([1, 2]), jnp.array(4)],
        ...     [jnp.array([5, 5]), jnp.array(3)],
        ... )
        >>> stack_one_level(trees, axis=0)
        [Array([[1, 2], [5, 5]], dtype=int32), Array([4, 3], dtype=int32)]
    ```
    Evolved from: [link](https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557).
    """
    leaves, _ = eqx.tree_flatten_one_level(pytrees)
    return jax.tree.map(lambda *v: jnp.stack(v, axis=axis), *leaves)


def tree_unstack(tree, *, axis=0, structure: Optional[PyTreeDef] = None):  # type: ignore # TODO: return when completed: https://github.com/jax-ml/jax/issues/29037
    """Inverse of `stack`: split a pytree whose leaves were stacked along `axis`
    into N separate pytrees.

    If `structure` is provided (e.g., from `eqx.tree_flatten_one_level`),
    the list of N pytrees is immediately placed back into that container and
    returned as a single pytree.

    **Arguments**:

    - `tree`: A pytree whose leaves are array-like and all share the same size N along `axis`.
    - `axis`: The axis that carries the size-N dimension in each leaf (default=0).
    - `structure`: Optional `PyTreeDef`. If provided, the list of N pytrees is immediately placed
    back into that container and returned as a single pytree.

    **Returns**:
        If `structure` is `None`: a list of N pytrees.
        Otherwise: a single pytree produced by unflattening `structure` with those N pytrees.

    **Example**:
    ```python
        >>> trees = (
        ...     [jnp.array([1, 2]), jnp.array(4)],
        ...     [jnp.array([5, 5]), jnp.array(3)],
        ... )
        >>> batched = stack(trees, axis=0)
        >>> unstack(batched, axis=0)
        [[Array([1, 2], dtype=int32), Array(4, dtype=int32)],
         [Array([5, 5], dtype=int32), Array(3, dtype=int32)]]
    ```
    """

    if axis != 0:
        tree = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0), tree)

    leaves, treedef = jax.tree.flatten(tree)
    list_of_leaves = [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
    if structure is not None:
        return structure.unflatten(list_of_leaves)
    return list_of_leaves
