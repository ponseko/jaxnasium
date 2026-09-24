"""Tests for the Transition container and n-step collapse utility."""

import jax
import jax.numpy as jnp
import pytest

from jaxnasium.algorithms.core import Transition, n_step_to_cumulative_single_step

SEED = jax.random.PRNGKey(0)


def _single_agent_transition(n: int) -> Transition:
    return Transition(
        observation=jnp.arange(n, dtype=jnp.float32).reshape(n, 1),
        action=jnp.arange(n, dtype=jnp.float32),
        reward=jnp.ones((n,)),
        terminated=jnp.zeros((n,), dtype=bool),
        truncated=jnp.zeros((n,), dtype=bool),
        next_observation=(jnp.arange(n, dtype=jnp.float32) + 10.0).reshape(n, 1),
    )


def _multi_agent_transition(n: int) -> Transition:
    """Per-agent leaves for observation/action/reward, shared terminated/truncated."""
    return Transition(
        observation={"a0": jnp.ones((n, 2)), "a1": jnp.zeros((n, 2))},  # type: ignore
        action={"a0": jnp.ones((n,)), "a1": jnp.zeros((n,))},  # type: ignore
        reward={"a0": jnp.ones((n,)), "a1": jnp.full((n,), 2.0)},  # type: ignore
        terminated=jnp.zeros((n,), dtype=bool),
        truncated=jnp.zeros((n,), dtype=bool),
        next_observation={
            "a0": (jnp.arange(n, dtype=jnp.float32) + 10.0).reshape(n, 1),
            "a1": (jnp.arange(n, dtype=jnp.float32) + 20.0).reshape(n, 1),
        },  # type: ignore
    )


def test_structure_single_vs_multi_agent():
    single = _single_agent_transition(3)
    multi = _multi_agent_transition(3)
    assert single.structure == jax.tree.structure(0)
    assert multi.structure == jax.tree.structure({"a0": 0, "a1": 0})


def test_transpose_single_agent_is_noop():
    t = _single_agent_transition(3)
    assert t.view_transposed is t
    assert Transition.from_transposed(t) is t


def test_view_transposed_and_from_transposed_roundtrip():
    t = _multi_agent_transition(3)
    per_agent = t.view_transposed

    assert isinstance(per_agent, dict)
    assert set(per_agent.keys()) == {"a0", "a1"}
    assert isinstance(per_agent["a0"], Transition)
    assert jnp.array_equal(per_agent["a1"].reward, jnp.full((3,), 2.0))
    assert jnp.array_equal(per_agent["a0"].terminated, t.terminated)

    rebuilt = Transition.from_transposed(per_agent)
    assert rebuilt.structure == t.structure
    assert jnp.array_equal(rebuilt.reward["a0"], t.reward["a0"])
    assert jnp.array_equal(rebuilt.reward["a1"], t.reward["a1"])
    assert jnp.array_equal(rebuilt.observation["a0"], t.observation["a0"])


def test_make_minibatches():
    batch_size = 8
    t = Transition(
        observation=jnp.arange(batch_size * 2, dtype=jnp.float32).reshape(
            batch_size, 2
        ),
        action=jnp.arange(batch_size, dtype=jnp.float32),
        reward=jnp.arange(batch_size, dtype=jnp.float32),
        terminated=jnp.zeros((batch_size,), dtype=bool),
        truncated=jnp.zeros((batch_size,), dtype=bool),
    )
    minibatches = t.make_minibatches(SEED, n_minibatches=4)
    # (n_minibatches, minibatch_size, ...)
    assert minibatches.reward.shape == (4, 2)
    assert minibatches.observation.shape == (4, 2, 2)

    # check that the minibatches are actually different
    assert not jnp.array_equal(minibatches.observation[0], minibatches.observation[1])


def test_make_minibatches_n_epochs_stacks():
    batch_size = 8
    t = Transition(
        observation=jnp.arange(batch_size, dtype=jnp.float32).reshape(batch_size, 1),
        action=jnp.arange(batch_size, dtype=jnp.float32),
        reward=jnp.arange(batch_size, dtype=jnp.float32),
        terminated=jnp.zeros((batch_size,), dtype=bool),
        truncated=jnp.zeros((batch_size,), dtype=bool),
    )
    minibatches = t.make_minibatches(SEED, n_minibatches=2, n_epochs=3)
    # 3 epochs * 2 minibatches stacked along the leading axis.
    assert minibatches.reward.shape == (6, 4)


# --------------------------------------------------------------------------- #
# n-step collapse
# --------------------------------------------------------------------------- #
def test_n_step_one_is_identity():
    t = _single_agent_transition(4)
    assert n_step_to_cumulative_single_step(t, n_step=1, gamma=0.9) is t


def test_n_step_cumulative_reward_no_done():
    n, gamma = 4, 0.9
    t = _single_agent_transition(n)  # all 1 rewards
    collapsed = n_step_to_cumulative_single_step(t, n_step=n, gamma=gamma)

    expected = sum(1.0 * gamma**i for i in range(n))  # rewards are all 1
    assert collapsed.reward == pytest.approx(expected)
    # observation/action taken from the first step; next_obs from the boundary (last).
    assert jnp.array_equal(collapsed.observation, t.observation[0])
    assert jnp.array_equal(collapsed.action, t.action[0])
    assert jnp.array_equal(collapsed.next_observation, t.next_observation[n - 1])  # type: ignore
    assert bool(collapsed.terminated) is False


def test_n_step_stops_reward_at_first_termination():
    n, gamma = 4, 0.9
    t = _single_agent_transition(n)
    t = t.replace(terminated=jnp.array([False, True, False, False]))

    collapsed = n_step_to_cumulative_single_step(t, n_step=n, gamma=gamma)

    # Reward accumulates up to and including the done step (index 1).
    expected = 1.0 + gamma
    assert collapsed.reward == pytest.approx(expected)
    # Boundary is the first done index.
    assert jnp.array_equal(collapsed.next_observation, t.next_observation[1])  # type: ignore
    assert bool(collapsed.terminated) is True


def test_n_step_stops_reward_at_first_truncation():
    n, gamma = 4, 0.9
    t = _single_agent_transition(n)
    t = t.replace(truncated=jnp.array([False, False, True, False]))

    collapsed = n_step_to_cumulative_single_step(t, n_step=n, gamma=gamma)

    expected = 1.0 + gamma + gamma**2
    assert collapsed.reward == pytest.approx(expected)
    assert jnp.array_equal(collapsed.next_observation, t.next_observation[2])  # type: ignore
    assert bool(collapsed.truncated) is True


def test_n_step_multi_agent_collapse_per_agent():
    n, gamma = 4, 0.9
    t = _multi_agent_transition(n)
    collapsed = n_step_to_cumulative_single_step(t, n_step=n, gamma=gamma)

    discount_sum_a0 = sum(1.0 * gamma**i for i in range(n))
    discount_sum_a1 = sum(2.0 * gamma**i for i in range(n))
    assert collapsed.reward["a0"] == pytest.approx(discount_sum_a0)
    assert collapsed.reward["a1"] == pytest.approx(discount_sum_a1)
    # Structure is preserved through the transpose/merge.
    assert collapsed.structure == t.structure
    assert collapsed.observation["a0"].shape == (2,)


def test_n_step_multi_agent_collapse_per_agent_diff_terminations():
    n, gamma = 4, 0.9
    t = _multi_agent_transition(n)

    # test with different termination signals
    t = t.replace(
        terminated={
            "a0": jnp.array([False, True, False, False]),
            "a1": jnp.array([False, False, True, False]),
        }
    )
    collapsed = n_step_to_cumulative_single_step(t, n_step=n, gamma=gamma)

    discount_sum_a0 = sum(1.0 * gamma**i for i in range(2))
    discount_sum_a1 = sum(2.0 * gamma**i for i in range(3))
    assert collapsed.reward["a0"] == pytest.approx(discount_sum_a0)
    assert collapsed.reward["a1"] == pytest.approx(discount_sum_a1)
    # Structure is preserved through the transpose/merge.
    assert collapsed.structure == t.structure
    assert collapsed.observation["a0"].shape == (2,)

    # check that the next observations are correct
    assert jnp.array_equal(
        collapsed.next_observation["a0"],  # type: ignore
        t.next_observation["a0"][1],
    )
    assert jnp.array_equal(
        collapsed.next_observation["a1"],  # type: ignore
        t.next_observation["a1"][2],
    )
    assert bool(collapsed.terminated["a0"]) is True
    assert bool(collapsed.terminated["a1"]) is True
