import equinox as eqx
import jax
import jax.numpy as jnp
from _proxy_test_envs import (
    make_proxy_env,
    obs_ma_dict_heterogeneous,
    obs_ma_dict_homogeneous,
)

from jaxnasium import AgentObservation
from jaxnasium.algorithms import DQN
from jaxnasium.algorithms.core import Transition, _multi_agent
from jaxnasium.algorithms.core._multi_agent import MultiAgentWrapper, map_multi_agent

SEED = jax.random.PRNGKey(0)

"""Direct unit tests for the multi-agent primitives (`map_multi_agent`, `MultiAgentWrapper`).

The end-to-end multi-agent paths (init_agent / get_action on multi-agent proxy envs) are already exercised in ``test_multi_agent_input_output.py``.
"""


SMALL_DQN = {
    "total_timesteps": 64,
    "update_every": 16,
    "num_envs": 4,
    "batch_size": 16,
    "replay_buffer_size": 64,
    "warmup_steps": 16,
    "log_function": None,
}


def _ma_env():
    return make_proxy_env(obs_ma_dict_homogeneous, multi_agent=True)


def _multi_agent_transition(n: int = 5) -> Transition:
    return Transition(
        observation={"a0": jnp.ones((n, 3)), "a1": jnp.zeros((n, 3))},  # type: ignore
        action={"a0": jnp.ones((n, 1)), "a1": jnp.zeros((n, 1))},  # type: ignore
        reward={"a0": jnp.ones((n, 1)), "a1": jnp.full((n, 1), 2.0)},  # type: ignore
        terminated=jnp.zeros((n,), dtype=bool),
        truncated=jnp.zeros((n,), dtype=bool),
    )


def test_map_multi_agent_broadcasts_shared_arg():
    states = {"a0": jnp.array([1.0, 2.0]), "a1": jnp.array([3.0, 4.0])}
    obs = {"a0": jnp.array([0.1, 0.2]), "a1": jnp.array([0.3, 0.4])}

    result = map_multi_agent(lambda s, o, shared: s + o + shared, states, obs, 10.0)

    assert set(result.keys()) == {"a0", "a1"}
    assert jnp.allclose(result["a0"], jnp.array([11.1, 12.2]))
    assert jnp.allclose(result["a1"], jnp.array([13.3, 14.4]))


def test_map_multi_agent_handles_heterogeneous_agent_shapes():
    """Agents differ in what they carry"""
    states = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0, 5.0])]
    obs = [jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4, 0.5])]

    result = map_multi_agent(lambda s, o, shared: s + o + shared, states, obs, 10.0)

    assert jnp.allclose(result[0], jnp.array([11.1, 12.2]))
    assert jnp.allclose(result[1], jnp.array([13.3, 14.4, 15.5]))


def test_map_multi_agent_splits_prng_keys_per_agent():
    states = {"a0": 0, "a1": 0}
    result = map_multi_agent(
        lambda key, s: s + jax.random.randint(key, (), 0, 1000), SEED, states
    )

    assert set(result.keys()) == {"a0", "a1"}
    assert result["a0"] != result["a1"]


def test_map_multi_agent_transposes_transition_input():
    transition = _multi_agent_transition()

    result = map_multi_agent(lambda t: t.reward + t.action, transition)

    assert set(result.keys()) == {"a0", "a1"}
    assert jnp.allclose(result["a0"], 1.0 + 1.0)
    assert jnp.allclose(result["a1"], 2.0 + 0.0)


def test_map_multi_agent_merges_transition_output():
    transition = _multi_agent_transition()

    result = map_multi_agent(lambda t: t.replace(reward=t.reward * 2.0), transition)

    assert isinstance(result, Transition)
    assert jnp.allclose(result.reward["a0"], 2.0)
    assert jnp.allclose(result.reward["a1"], 4.0)


def test_map_multi_agent_tuple_return_becomes_tuple_of_trees():
    states = {"a0": jnp.array(1.0), "a1": jnp.array(2.0)}

    result = map_multi_agent(lambda s: (s + 1, s - 1), states)

    assert isinstance(result, tuple) and len(result) == 2
    plus, minus = result
    assert plus == {"a0": 2.0, "a1": 3.0}
    assert minus == {"a0": 0.0, "a1": 1.0}


def test_map_multi_agent_preserves_agent_observation():
    obs = (
        AgentObservation(observation=jnp.ones(3), action_mask=jnp.ones(2, dtype=bool)),
        AgentObservation(
            observation=jnp.zeros(3), action_mask=jnp.zeros(2, dtype=bool)
        ),
    )
    for vmap in (False, True):
        result = map_multi_agent(lambda o: o, obs, vmap=vmap)
        assert type(result[0]) is AgentObservation
        assert type(result[1]) is AgentObservation


class _TestAgent(eqx.Module):
    w: jax.Array

    def act(self, x):
        return x * self.w


def _make_wrapper() -> MultiAgentWrapper:
    return MultiAgentWrapper(
        {"a0": _TestAgent(jnp.array(2.0)), "a1": _TestAgent(jnp.array(3.0))}
    )


def test_wrapper_dispatches_method_per_agent():
    wrapper = _make_wrapper()
    out = wrapper.act({"a0": jnp.array(1.0), "a1": jnp.array(1.0)})
    assert out["a0"] == 2.0
    assert out["a1"] == 3.0


def test_wrapper_gathers_leaf_attribute_per_agent():
    wrapper = _make_wrapper()
    out = wrapper.w
    assert out["a0"] == 2.0  # type: ignore
    assert out["a1"] == 3.0  # type: ignore


def test_wrapper_structure_reflects_agents():
    wrapper = _make_wrapper()
    assert wrapper._structure == jax.tree.structure({"a0": 0, "a1": 0})
    assert wrapper._matches_structure({"a0": 99.0, "a1": 999.0})
    assert not wrapper._matches_structure({"a0": 1.0})


def test_wrapper_subclass_survives_with_hyperparams():
    class Team(MultiAgentWrapper):
        pass

    agent = DQN(**SMALL_DQN).init_agent(SEED, _ma_env())
    assert isinstance(agent, MultiAgentWrapper)
    team = Team(agent.agents, trainer=agent.trainer)
    updated = team.with_hyperparams()
    assert type(updated) is Team


def test_collective_methods_are_not_dispatched_per_agent():
    """`@collective` methods must run once, with the wrapper as `self`."""
    agent = DQN(**SMALL_DQN).init_agent(SEED, _ma_env())
    assert isinstance(agent, MultiAgentWrapper)

    for name in ("train", "evaluate", "save"):
        method = getattr(agent, name)
        assert getattr(method, "__self__", None) is agent, (
            f"`{name}` was dispatched per agent; it should run once on the wrapper"
        )

    assert not hasattr(agent.get_action, "__self__")


def test_multi_agent_train_runs_a_single_loop():
    env = _ma_env()
    agent, _ = DQN(**SMALL_DQN).train(SEED, env)

    assert isinstance(agent, MultiAgentWrapper)
    assert sorted(agent.agents) == ["agent_0", "agent_1"]

    rewards = agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=2)
    assert rewards["agent_0"].shape == (2,) and rewards["agent_1"].shape == (2,)


def test_multi_agent_train_continues_from_an_existing_agent():
    env = _ma_env()
    agent, _ = DQN(**SMALL_DQN).train(SEED, env)
    assert isinstance(agent, MultiAgentWrapper)
    before = jax.tree.leaves(agent.agents["agent_0"].critic)[0]

    agent, _ = agent.train(jax.random.PRNGKey(1), env)
    after = jax.tree.leaves(agent.agents["agent_0"].critic)[0]

    assert not jnp.allclose(before, after), "continued training changed nothing"


def test_per_agent_hyperparameters_reach_each_agent():
    """`gamma={"agent_0": ..., "agent_1": ...}` must reach the *updates*.

    The per-agent trainers are built at promotion time and kept by each agent, so
    `self.trainer.gamma` inside an update is that agent's value.
    """
    gamma = {"agent_0": 0.999, "agent_1": 0.5}
    agent, _ = DQN(gamma=gamma, **SMALL_DQN).train(SEED, _ma_env())  # type: ignore

    assert isinstance(agent, MultiAgentWrapper)
    assert {k: a.trainer.gamma for k, a in agent.agents.items()} == gamma
    # the team trainer keeps what the user configured, unsplit
    assert agent.trainer.gamma == gamma


def test_with_hyperparams_keeps_team_and_agents_in_sync():
    """Fanning out alone would leave the wrapper's trainer -- the loop-level view --
    behind, so the loop and the updates would silently disagree."""
    agent = DQN(**SMALL_DQN).init_agent(SEED, _ma_env())
    agent = agent.with_hyperparams(gamma=0.5, total_timesteps=32)
    assert isinstance(agent, MultiAgentWrapper)

    assert agent.trainer.gamma == 0.5
    assert agent.trainer.total_timesteps == 32
    assert all(a.trainer.gamma == 0.5 for a in agent.agents.values())
    assert all(a.trainer.total_timesteps == 32 for a in agent.agents.values())


def test_with_hyperparams_splits_per_agent_values():
    gamma = {"agent_0": 0.9, "agent_1": 0.1}
    agent = DQN(**SMALL_DQN).init_agent(SEED, _ma_env())
    agent = agent.with_hyperparams(gamma=gamma)
    assert isinstance(agent, MultiAgentWrapper)

    assert {k: a.trainer.gamma for k, a in agent.agents.items()} == gamma
    assert agent.trainer.gamma == gamma


def _count_traces(states, **kwargs):
    """How many times `f` is traced when mapped over `states`."""
    traces = []
    result = map_multi_agent(
        lambda s, *a: (traces.append(1), s * 2)[1], states, **kwargs
    )
    return len(traces), result


def test_map_multi_agent_vmaps_homogeneous_agents():
    """The point of the vmap path: one trace of `f` regardless of agent count."""
    states = {f"a{i}": jnp.array([float(i)]) for i in range(6)}
    assert _count_traces(states)[0] == 1
    assert _count_traces(states, vmap=False)[0] == 6


def test_map_multi_agent_loops_over_heterogeneous_agents():
    """Differing shapes have no shared agent axis, so `f` is traced per agent."""
    states = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0, 5.0])]

    traces, result = _count_traces(states)

    assert traces == 2
    assert jnp.allclose(result[0], jnp.array([2.0, 4.0]))
    assert jnp.allclose(result[1], jnp.array([6.0, 8.0, 10.0]))


def test_map_multi_agent_vmap_matches_the_loop():
    states = {"a0": jnp.array([1.0, 2.0]), "a1": jnp.array([3.0, 4.0])}
    obs = {"a0": jnp.array([0.1, 0.2]), "a1": jnp.array([0.3, 0.4])}

    def add(s, o, shared):
        return s + o + shared

    looped = map_multi_agent(add, states, obs, 10.0, vmap=False)
    vmapped = map_multi_agent(add, states, obs, 10.0)

    for key in states:
        assert jnp.allclose(looped[key], vmapped[key])


def test_map_multi_agent_vmap_keeps_the_special_cased_arguments():
    """Key splitting, `Transition` transposition and tuple returns survive the vmap path."""
    states = {"a0": jnp.array(0), "a1": jnp.array(0)}
    keyed = map_multi_agent(
        lambda key, s: s + jax.random.randint(key, (), 0, 100_000), SEED, states
    )
    assert keyed["a0"] != keyed["a1"]

    merged = map_multi_agent(
        lambda t: t.replace(reward=t.reward * 2.0), _multi_agent_transition()
    )
    assert isinstance(merged, Transition)
    assert jnp.allclose(merged.reward["a0"], 2.0)
    assert jnp.allclose(merged.reward["a1"], 4.0)

    plus, minus = map_multi_agent(lambda s: (s + 1, s - 1), states)
    assert plus["a0"] == 1 and minus["a1"] == -1


def test_map_multi_agent_batch_size_env_var_chunks_the_vmap(monkeypatch):
    states = {f"a{i}": jnp.array([float(i)]) for i in range(6)}
    reference = map_multi_agent(lambda s: s * 2, states, vmap=False)

    monkeypatch.setenv("JAXNASIUM_MULTI_AGENT_BATCH_SIZE", "2")
    chunked = map_multi_agent(lambda s: s * 2, states)

    for key in states:
        assert jnp.allclose(reference[key], chunked[key])


def test_multi_agent_train_vmaps_homogeneous_agents(monkeypatch):
    """Homogeneous agents are vmapped."""
    calls = []
    original = _multi_agent._vmap_over_agents
    monkeypatch.setattr(
        _multi_agent,
        "_vmap_over_agents",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )

    agent, _ = DQN(**SMALL_DQN).train(SEED, _ma_env())

    assert calls, "homogeneous agents should have taken the vmap path"
    assert isinstance(agent, MultiAgentWrapper)


def test_multi_agent_train_loops_over_heterogeneous_agents(monkeypatch):
    """Agents with differing observation are not vmapped"""
    calls = []
    original = _multi_agent._vmap_over_agents
    monkeypatch.setattr(
        _multi_agent,
        "_vmap_over_agents",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )

    env = make_proxy_env(obs_ma_dict_heterogeneous, multi_agent=True)
    agent, _ = DQN(**SMALL_DQN).train(SEED, env)

    assert not calls, "heterogeneous agents must fall back to the per-agent loop"
    assert isinstance(agent, MultiAgentWrapper)
