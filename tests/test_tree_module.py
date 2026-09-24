import jax
import jax.numpy as jnp
import optax
import pytest

import jaxnasium as jym
from jaxnasium.algorithms.core import Schedule


def _leaves_all_equal(tree_a, tree_b):
    """Compare two pytrees leaf-wise with jnp.all."""
    leaves_a = jax.tree.leaves(tree_a)
    leaves_b = jax.tree.leaves(tree_b)
    assert len(leaves_a) == len(leaves_b)
    return all(jnp.all(a == b) for a, b in zip(leaves_a, leaves_b, strict=True))


def test_mean():
    tree = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(3.0)}
    assert jym.tree.mean(tree) == 2.0


def test_get_first_nested_dict():
    tree = {"outer": {"inner": {"count": jnp.array(7)}}}
    assert jym.tree.get_first(tree, "count") == 7


def test_get_first_missing_key_raises():
    with pytest.raises(KeyError, match="count"):
        jym.tree.get_first({"a": 1}, "count")


def test_get_first_optax_optimizer_count():
    """Matches SAC usage: read the Adam/AdaBelief step count from optimizer state.
    This is also used for other schedules (e.g. epsilon, alpha, etc.)"""
    schedule = Schedule(3e-3, None, 100)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adabelief(learning_rate=schedule),
    )
    params = {"weight": jnp.array(1.0)}
    optimizer_state = optimizer.init(params)

    assert jym.tree.get_first(optimizer_state, "count") == 0

    grads = {"weight": jnp.array(0.1)}
    for expected_count in (1, 2, 3):
        _, optimizer_state = optimizer.update(grads, optimizer_state, params)
        assert jym.tree.get_first(optimizer_state, "count") == expected_count

    assert jym.tree.get_first(optimizer_state, "count") == 3


def test_map_one_level():
    tree = {"a": jnp.array(1), "b": jnp.array(2)}
    doubled = jym.tree.map_one_level(lambda x: x * 2, tree)
    assert doubled == {"a": jnp.array(2), "b": jnp.array(4)}


def test_ravel():
    tree = {"a": jnp.array([1, 2]), "b": jnp.array(3)}
    assert jnp.array_equal(jym.tree.ravel(tree), jnp.array([1, 2, 3]))


def test_concatenate():
    trees = (
        [jnp.array([1, 2]), jnp.array([4])],
        [jnp.array([5, 5]), jnp.array([3])],
    )
    out = jym.tree.concatenate(trees)
    assert jnp.array_equal(out[0], jnp.array([1, 2, 5, 5]))
    assert jnp.array_equal(out[1], jnp.array([4, 3]))


def test_batch_sum():
    tree = {
        "a": jnp.array([[1, 2], [3, 4]]),
        "b": jnp.array([[5, 6], [7, 8]]),
    }
    assert jnp.array_equal(jym.tree.batch_sum(tree, batch_axes=0), jnp.array([14, 22]))


def test_batch_mean():
    tree = {
        "a": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "b": jnp.array([[5.0, 6.0], [7.0, 8.0]]),
    }
    assert jnp.allclose(jym.tree.batch_mean(tree, batch_axes=0), jnp.array([3.5, 5.5]))
    assert jnp.allclose(
        jym.tree.batch_mean(tree, batch_axes=0),
        jym.tree.batch_sum(tree, batch_axes=0) / 4,
    )


def test_batch_mean_multiple_batch_axes():
    tree = {"x": jnp.arange(24.0).reshape(2, 3, 4), "y": jnp.ones((2, 3, 4))}
    expected = jym.tree.batch_sum(tree, batch_axes=(0, 1)) / 8
    assert jnp.allclose(jym.tree.batch_mean(tree, batch_axes=(0, 1)), expected)


def test_gather_actions_discrete():
    q_values = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actions = jnp.array([0, 2])
    gathered = jym.tree.gather_actions(q_values, actions)
    assert jnp.array_equal(gathered, jnp.array([1.0, 6.0]))


def test_gather_actions_discrete_not_last_axis():
    q_values = jnp.array(
        [[[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]], [[4.0, 0.4], [5.0, 0.5], [6.0, 0.6]]]
    )
    actions = jnp.array([0, 2])
    gathered = jym.tree.gather_actions(q_values, actions, axis=-2)
    assert jnp.array_equal(gathered, jnp.array([[1.0, 0.1], [6.0, 0.6]]))


def test_gather_actions_continuous_passthrough():
    """When shapes already match, the value is returned unchanged."""
    q_value = jnp.array([1.5, 2.5])
    action = jnp.array([0.1, 0.2])
    assert jnp.array_equal(jym.tree.gather_actions(q_value, action), q_value)


def test_gather_actions_continuous_mismatched_shape():
    """Continuous Q(s, a) is already computed; do not index with float actions."""
    q_value = jnp.array(1.5)
    action = jnp.array([0.1, 0.2, 0.3])
    assert jnp.array_equal(jym.tree.gather_actions(q_value, action), q_value)


def test_gather_actions_composite_continuous():
    q_values = {"throttle": jnp.array(1.5), "steer": jnp.array(2.5)}
    actions = {
        "throttle": jnp.array([0.1, 0.2]),
        "steer": jnp.array([0.3, 0.4, 0.5]),
    }
    gathered = jym.tree.gather_actions(q_values, actions)
    assert gathered == q_values


def test_stack_and_unstack_roundtrip():
    trees = (
        [jnp.array([1, 2]), jnp.array(4)],
        [jnp.array([5, 5]), jnp.array(3)],
    )
    stacked = jym.tree.stack(trees)
    unstacked = jym.tree.unstack(stacked)
    assert _leaves_all_equal(trees, unstacked)


def test_add_and_mul():
    tree = {"a": jnp.array([1, 2]), "b": jnp.array(3)}
    assert jnp.array_equal(jym.tree.add(tree, 1)["a"], jnp.array([2, 3]))
    assert jnp.array_equal(jym.tree.mul(tree, 2)["b"], jnp.array(6))


def test_zeros_and_ones_like():
    tree = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(3.0)}
    zeros = jym.tree.zeros_like(tree)
    ones = jym.tree.ones_like(tree)
    assert all(jnp.all(leaf == 0.0) for leaf in jax.tree.leaves(zeros))
    assert all(jnp.all(leaf == 1.0) for leaf in jax.tree.leaves(ones))


def test_split_key_like_structure():
    structure = jax.tree.structure({"a": 0, "b": 0})
    keys = jym.tree.split_key_like_structure(jax.random.PRNGKey(0), structure)
    leaves = jax.tree.leaves(keys)
    assert len(leaves) == 2
    assert all(leaf.shape == (2,) for leaf in leaves)


def test_clip():
    tree = {
        "a": jnp.array([-2.0, 0.5, 3.0]),
        "b": jnp.array([1.0, -1.0]),
        "c": jnp.array([0.0]),
        "d": jnp.array([2.0, -3.0, 4.0]),
    }
    clipped_tree = jym.tree.clip(tree, -1.0, 1.0)
    assert jnp.array_equal(clipped_tree["a"], jnp.array([-1.0, 0.5, 1.0]))
    assert jnp.array_equal(clipped_tree["b"], jnp.array([1.0, -1.0]))
    assert jnp.array_equal(clipped_tree["c"], jnp.array([0.0]))
    assert jnp.array_equal(clipped_tree["d"], jnp.array([1.0, -1.0, 1.0]))
