import math
import operator
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, PRNGKeyArray, PyTree, PyTreeDef

"""
Convenience pytree functions used in the various RL algorithms which
aren't found in used higher-level libraries (equinox / jax).
"""


def _transpose_tree_of_tuples(r, outer_treedef):
    """
    Some functions may return tuples and pytree operations may return these
    as a pytree of tuples, rather than a possibly desired tuple of pytrees.
    Here, we convert these with a `jax.tree.transpose`
    """
    flat = outer_treedef.flatten_up_to(r)
    if not flat or not isinstance(flat[0], tuple):
        return r
    inner_treedef = jax.tree.structure(tuple(range(len(flat[0]))))
    return jax.tree.transpose(outer_treedef, inner_treedef, r)


def _tree_size(tree):
    r"""Get the total number of elements (size of each leaf) in a pytree.
    Ported from: https://github.com/google-deepmind/optax/pull/1321/files/cadb2bca89e2af6af0e70cf0007080d5f68794a4
    """
    return sum([jnp.size(leaf) for leaf in jax.tree.leaves(tree)])


def tree_sum(tree: Any, axis: int | tuple[int, ...] | None = None):
    """
    Compute the sum of all the elements in a pytree
    If axis is provided, sums each leaf over the specified axis and
    then adds adds the resulting leafs.
    """
    sums = jax.tree.map(lambda x: jnp.sum(x, axis=axis), tree)
    return jax.tree.reduce_associative(operator.add, sums)  # type: ignore[reportGeneralTypeIssues]


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
    sum = tree_sum(tree)
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

        from jaxnasium.algorithms.core._distributions import DistraxContainer
    except ImportError:
        raise ImportError(
            "jaxnasium.algorithms is required for `jaxnasium.tree.map_distributions()`. Please install  `pip install jaxnasium[algs]`."
        )

    if isinstance(tree, DistraxContainer):
        tree = tree.distribution
    # any of *rest should also be converted:
    rest = tuple(r.distribution if isinstance(r, DistraxContainer) else r for r in rest)

    if isinstance(tree, distrax.Joint):
        tree = tree.distributions
    rest = tuple(r.distributions if isinstance(r, distrax.Joint) else r for r in rest)

    is_dist = lambda x: isinstance(x, distrax.Distribution)
    result = jax.tree.map(fn, tree, *rest, is_leaf=is_dist)

    # e.g. sample_and_log_prob returns a (sample, log_prob) tuple,
    # rather than returning a [(sample1, log_prob1), (sample2, log_prob2), ...] pytree,
    # we return a ([sample1, sample2, ...], [log_prob1, log_prob2, ...]) tuple of pytrees.
    structure = jax.tree.structure(tree, is_leaf=is_dist)
    return _transpose_tree_of_tuples(result, structure)


def tree_ravel(tree: PyTree) -> Array:
    """Flatten a pytree of arrays into a single 1D array.

    **Arguments**:

    - `tree`: A pytree whose leaves are array-like.

    **Returns**: A 1D array containing every element of the pytree.

    **Example**:
    ```python
        >>> tree = {'a': jnp.array([1, 2]), 'b': jnp.array(3)}
        >>> tree_ravel(tree)
        Array([1, 2, 3], dtype=int32)
    ```
    """
    return jnp.concatenate(jax.tree.leaves(jax.tree.map(jnp.ravel, tree)))


def _key_entry_name(key_entry: Any) -> str | None:
    """Return the string name of a JAX/optax pytree key entry, if available."""
    if isinstance(key_entry, jax.tree_util.GetAttrKey):
        return key_entry.name
    if isinstance(key_entry, jax.tree_util.DictKey):
        dict_key = key_entry.key
        return dict_key if isinstance(dict_key, str) else None
    try:
        from optax.tree_utils._state_utils import NamedTupleKey

        if isinstance(key_entry, NamedTupleKey):
            return key_entry.name
    except ImportError:
        pass
    return None


def tree_get_first(tree: PyTree, key: str) -> Any:
    """Get the first value from a pytree with the given key.

    **Arguments**:

    - `tree`: A pytree.
    - `key`: A string key.

    **Returns**:
        The first value from the pytree with the given key.

    **Raises**:
        KeyError: If the key is not found in the pytree.
    """
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if not path:
            continue
        if _key_entry_name(path[-1]) == key:
            return leaf
    raise KeyError(f"Key '{key}' not found in tree: {tree}.")


def tree_batch_sum(values, batch_axes: int | tuple[int, ...] = 0):
    """
    Sum over all non-batch axes of each leaf in a pytree, then sum (reduce) across leaves.
    The batch axes(s) is/are assumed to be the leading axes.

    This is essentially `jaxnasium.tree.sum` or `optax.tree.sum` but with a variable
    axis argument resulting in a sum over all non-batch axes.

    **Arguments**:
        values:  Pytree of JAX arrays. Every leaf must have at least `len(batch_axes)` leading dimensions.
        batch_axes: Leading axes to exclude from the sum.

    **Returns**:
        A JAX array with the same shape as the batch dimensions.

    **Notes**:
       - For a single leaf with only batch dimensions, this is a no-op.

    **Example**:
        >>> tree = {"a": jnp.array([[1, 2], [3, 4]]), "b": jnp.array([[5, 6], [7, 8]])}
        >>> tree_batch_sum(tree, batch_axes=0)
        Array([14, 22])

        >>> tree2 = {"x": jnp.ones((2, 3, 4)), "y": jnp.ones((2, 3, 4))}
        >>> tree_batch_sum(tree2, batch_axes=(0, 1))
        Array([[8., 8., 8.], [8., 8., 8.]])

    """

    batch_axes = (batch_axes,) if isinstance(batch_axes, int) else tuple(batch_axes)
    if batch_axes != tuple(range(len(batch_axes))):
        raise ValueError(
            f"batch_axes must be a leading prefix (0, 1, ..., k-1), got {batch_axes}"
        )

    num_batch_dimensions = len(batch_axes)

    assert all(x.ndim >= num_batch_dimensions for x in jax.tree.leaves(values)), (
        f"Each array in the pytree must have at least {num_batch_dimensions} leading batch dimensions, "
        f"but got {values}"
    )
    assert all(
        x.shape[:num_batch_dimensions]
        == jax.tree.leaves(values)[0].shape[:num_batch_dimensions]
        for x in jax.tree.leaves(values)
    ), (
        f"Each array in the pytree must have the same shape for the first {num_batch_dimensions} dimensions, "
        f"but got {values}"
    )
    batch_wise_sums = jax.tree.map(
        lambda x: jnp.sum(x, axis=tuple(range(num_batch_dimensions, x.ndim))),
        values,
    )
    return jax.tree.reduce(operator.add, batch_wise_sums, initializer=0)


def tree_batch_mean(values, batch_axes: int | tuple[int, ...] = 0):
    """
    Average over all non-batch axes of each leaf in a pytree, and across leaves.
    The batch axes(s) is/are assumed to be the leading axes.

    All elements are weighted equally; essentially this is then a
    `jaxnasium.tree.sum` / <total number of elements>.

    Identical to `tree_batch_sum` up to a division by the (static) number of
    reduced elements.

    **Arguments**:
        values:  Pytree of JAX arrays. Every leaf must have at least `len(batch_axes)` leading dimensions.
        batch_axes: Leading axes to exclude from the reduction.

    **Returns**:
        A JAX array with the same shape as the batch dimensions.

    **Notes**:
       - For a single leaf with only batch dimensions, this is a no-op.

    **Example**:
        >>> tree = {
        ...     "a": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        ...     "b": jnp.array([[5.0, 6.0], [7.0, 8.0]]),
        ... }
        >>> tree_batch_mean(tree, batch_axes=0)
        Array([3.5, 5.5])
    """
    batch_axes = (batch_axes,) if isinstance(batch_axes, int) else tuple(batch_axes)
    num_batch_dimensions = len(batch_axes)

    total = tree_batch_sum(values, batch_axes)
    leaves = jax.tree.leaves(values)
    if not leaves:
        return total

    # number of elements excluding the batch axis.
    num_elements = sum(
        [math.prod(leaf.shape[num_batch_dimensions:]) for leaf in leaves]
    )
    if num_elements == 0:
        return total
    return total / num_elements


def tree_gather_actions(tree: PyTree, actions: PyTree, axis=-1):
    """Given a (pytree of) array-like values, gather the elements based
    on the indices provided in `actions`. If the arrays in `tree` are of the same
    shape as `actions`, the tree is assumed to be array of actions taken and
    the array is returned as is. This may be the case in continuous action spaces.

    For example, when given a (pytree of) q-values for each possible action,
    this function will return the q-values corresponding to the actions taken.
    In continuous action spaces, q-values cannot be generated per action and
    tree will already contain the q-value for the action taken. This q-value
    is then returned as is. This is also infered when the indices are floating
    point indices.

    **Arguments**:
        tree: Array or Pytree of arrays.
        actions: Array or same-structure Pytree of arrays as `tree`. The final axis of
        `actions` must contain elements that are valid indices for the corresponding arrays in `tree`.
        axis: The axis on which the actions live on each leaf within the tree.
    """

    def gather_actions(arr, indices):
        if arr.squeeze().shape == indices.squeeze().shape:
            return arr
        indices = jnp.asarray(indices)
        if jnp.isdtype(indices.dtype, "real floating"):
            return arr
        indices = jnp.reshape(indices, indices.shape + (1,) * (arr.ndim - indices.ndim))
        return jnp.take_along_axis(arr, indices, axis=axis).squeeze(axis)

    return jax.tree.map(gather_actions, tree, actions)


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


def tree_unstack(tree, *, axis=0, structure: PyTreeDef | None = None):  # type: ignore # TODO: return when completed: https://github.com/jax-ml/jax/issues/29037
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


def tree_concatenate(pytrees: PyTree, *, axis=0) -> PyTree:
    """Concatenate corresponding leaves of pytrees along the specified axis.

    Interprets the root node's immediate children as a batch of N pytrees that all
    share the same structure. For each leaf, concatenates the N leaves along `axis`
    using `jnp.concatenate`. This does not traverse deeper than one level when
    determining what to concatenate.

    **Arguments**:

    - `pytrees`: A pytree whose root has N immediate children. Each child must have
        the same pytree structure. Corresponding leaves must be array-like and
        compatible with `jnp.concatenate` along `axis`.
    - `axis`: Axis along which to concatenate corresponding leaves (default=0).

    **Returns**:
        A pytree with the same structure as a single direct-child element of
        `pytrees`, where each leaf is the concatenation of the corresponding
        leaves across all elements.

    **Example**:
    ```python
        >>> trees = (
        ...     [jnp.array([1, 2]), jnp.array([4])],
        ...     [jnp.array([5, 5]), jnp.array([3])],
        ... )
        >>> tree_concatenate(trees, axis=0)
        [Array([1, 2, 5, 5], dtype=int32), Array([4, 3], dtype=int32)]
    ```
    """
    leaves, _ = eqx.tree_flatten_one_level(pytrees)
    return jax.tree.map(lambda *v: jnp.concatenate(v, axis=axis), *leaves)


def tree_split_key_like_structure(key: PRNGKeyArray, structure: PyTreeDef):  # pyright: ignore[reportInvalidTypeForm]
    """Split a JAX PRNGKey into a pytree of keys with the same structure as `structure`.

    Similar to `optax.tree_utils.tree_split_key_like`, but operates on PyTreeDefs.

    *Arguments*:
        `key`: A PRNGKeyArray to be split.
        `agent_structure`: A pytree structure of agents.
    """
    num_keys = structure.num_leaves
    keys = list(jax.random.split(key, num_keys))
    return jax.tree.unflatten(structure, keys)


def tree_split_key_like(
    key: PRNGKeyArray, tree: PyTree, is_leaf: None | (Callable[[Any], bool]) = None
):  # pyright: ignore[reportInvalidTypeForm]
    """Split a JAX PRNGKey into a pytree of keys with the same structure as `tree`.

    *Arguments*:
        `key`: A PRNGKeyArray to be split.
        `tree`: A pytree whose structure will be used to determine the number of keys to split.
        `is_leaf`: A function that determines which nodes in the tree are leaves.
    """
    structure = jax.tree.structure(tree, is_leaf=is_leaf)
    return tree_split_key_like_structure(key, structure)


def tree_zeros_like(tree: PyTree, dtype: DTypeLike | None = None) -> PyTree:
    """
    Creates an all-zeros PyTree with the same structure as `tree`.

    **Arguments**:
        `tree`: A pytree.
        `dtype`: The dtype of the tree of zeros.
    """
    return jax.tree.map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)


def tree_ones_like(tree: PyTree, dtype: DTypeLike | None = None) -> PyTree:
    """
    Creates an all-ones PyTree with the same structure as `tree`.

    **Arguments**:
        `tree`: A pytree.
        `dtype`: The dtype of the tree of ones.
    """
    return jax.tree.map(lambda x: jnp.ones_like(x, dtype=dtype), tree)


def tree_add(tree_A: PyTree, tree_B_or_prefix: PyTree | float | Array) -> PyTree:
    """Add two pytrees or add a scalar, array, or prefix-pytree to each leaf of a pytree.

    **Arguments**:
        `tree_A`: First pytree.
        `tree_B_or_prefix`: Second pytree or scalar, array, or prefix-pytree of tree_A.

    **Example**:
    ```python
        >>> tree_A = [5, 6]
        >>> tree_B = [10, 11]
        >>> tree_add(tree_A, tree_B)
        [Array(15, dtype=int32), Array(17, dtype=int32)]
    ```
    ```python
        >>> tree_A = {'a': jnp.array([1, 2]), 'b': jnp.array(3)}
        >>> tree_B = 1
        >>> tree_add(tree_A, tree_B)
        {'a': Array([2, 3], dtype=int32), 'b': Array(4, dtype=int32)}
    ```
    """
    tree_B = jax.tree.broadcast(tree_B_or_prefix, tree_A)
    return jax.tree.map(jnp.add, tree_A, tree_B)


def tree_mul(tree_A: PyTree, tree_B_or_prefix: PyTree | float | Array) -> PyTree:
    """Multiply two pytrees or multiply a scalar, array, or prefix-pytree to each leaf of a pytree.

    **Arguments**:
        `tree_A`: First pytree.
        `tree_B_or_prefix`: Second pytree or scalar, array, or prefix-pytree of tree_A.

    **Example**:
    ```python
    >>> tree_A = [5, 6]
    >>> tree_B = [10, 11]
    >>> tree_mul(tree_A, tree_B)
    [Array(50, dtype=int32), Array(66, dtype=int32)]
    ```
    ```python
    >>> tree_A = {'a': jnp.array([1, 2]), 'b': jnp.array(3)}
    >>> tree_B = 2
    >>> tree_mul(tree_A, tree_B)
    {'a': Array([2, 4], dtype=int32), 'b': Array(6, dtype=int32)}
    ```
    """
    tree_B = jax.tree.broadcast(tree_B_or_prefix, tree_A)
    return jax.tree.map(jnp.multiply, tree_A, tree_B)


def tree_clip(
    tree: PyTree, min_value: float | Array, max_value: float | Array
) -> PyTree:
    """Clip the leaves of a pytree to a specified range.

    **Arguments**:
        `tree`: A pytree whose leaves are array-like.
        `min_value`: Minimum value to clip to (scalar or array).
        `max_value`: Maximum value to clip to (scalar or array).
    """
    return jax.tree.map(lambda x: jnp.clip(x, min_value, max_value), tree)
