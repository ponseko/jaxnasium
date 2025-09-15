from dataclasses import replace

import jax
import jax.numpy as jnp

from jaxnasium.algorithms.utils._multi_agent import transform_multi_agent
from jaxnasium.algorithms.utils._transition import Transition


def test_transform_multi_agent_default_w_dict():
    """Test transform_multi_agent with default behavior and dictionary inputs."""

    @transform_multi_agent
    def simple_agent_function(agent_state, observation, shared_param):
        """Simple function that processes agent state and observation."""
        return {
            "new_state": agent_state + 1,
            "action": jnp.argmax(observation),
            "shared_info": shared_param,
        }

    agent_states = {
        "agent0": jnp.array([1.0, 2.0, 3.0]),
        "agent1": jnp.array([4.0, 5.0, 6.0]),
        "agent2": jnp.array([7.0, 8.0, 9.0]),
    }

    observations = {
        "agent0": jnp.array([0.1, 0.8, 0.1]),
        "agent1": jnp.array([0.9, 0.05, 0.05]),
        "agent2": jnp.array([0.2, 0.3, 0.5]),
    }

    shared_param = jnp.array([10.0, 20.0, 30.0])

    result = simple_agent_function(agent_states, observations, shared_param)

    # Verify the structure is preserved
    assert isinstance(result, dict)
    assert set(result.keys()) == set({"agent0", "agent1", "agent2"})

    # Verify each agent's result has the expected structure
    for expected_key in ["new_state", "action", "shared_info"]:
        assert expected_key in result["agent0"]
        assert expected_key in result["agent1"]
        assert expected_key in result["agent2"]

    # Verify the computation is correct
    assert jnp.allclose(result["agent0"]["new_state"], jnp.array([2.0, 3.0, 4.0]))
    assert result["agent0"]["action"] == 1  # argmax of [0.1, 0.8, 0.1]
    assert result["agent1"]["action"] == 0  # argmax of [0.9, 0.05, 0.05]
    assert result["agent2"]["action"] == 2  # argmax of [0.2, 0.3, 0.5]


def test_transform_multi_agent_with_list_inputs():
    @transform_multi_agent
    def simple_agent_function(agent_state, observation, shared_param):
        return {
            "new_state": agent_state + 1,
            "action": jnp.argmax(observation),
            "shared_info": shared_param,
        }

    agent_states_list = [
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0]),
        jnp.array([5.0, 6.0]),
    ]

    observations_list = [
        jnp.array([0.7, 0.3]),
        jnp.array([0.2, 0.8]),
        jnp.array([0.6, 0.4]),
    ]

    shared_param = 42

    result_list = simple_agent_function(
        agent_states_list, observations_list, shared_param
    )

    # Verify the structure is preserved
    assert isinstance(result_list, list), "Result should be a list"
    assert len(result_list[0]["new_state"]) == 2, "Expected len == 2"
    assert result_list[2]["shared_info"] == 42, "Expected argmax to be 0"

    # Verify the computation is correct
    assert jnp.allclose(result_list[0]["new_state"], jnp.array([2.0, 3.0]))
    assert result_list[1]["action"] == 1  # argmax of [0.7, 0.3]
    assert result_list[2]["action"] == 0  # argmax of [0.6, 0.4]


def test_transform_multi_agent_with_tuple_inputs():
    @transform_multi_agent
    def simple_agent_function(agent_state, observation, shared_param):
        return {
            "new_state": agent_state + 1,
            "action": jnp.argmax(observation),
            "shared_info": shared_param,
        }

    agent_states_tuple = (jnp.array([1.0]), jnp.array([2.0]), jnp.array([3.0]))
    observations_tuple = (jnp.array([0.9]), jnp.array([0.8]), jnp.array([0.7]))
    shared_param = 42

    result_tuple = simple_agent_function(
        agent_states_tuple, observations_tuple, shared_param
    )

    # Verify the structure is preserved
    assert isinstance(result_tuple, tuple)
    assert len(result_tuple[0]["new_state"]) == 1
    assert result_tuple[2]["shared_info"] == 42

    # Verify the computation is correct
    assert jnp.allclose(result_tuple[0]["new_state"], jnp.array([2.0]))
    assert result_tuple[0]["action"] == 0  # argmax of [0.9]
    assert result_tuple[1]["action"] == 0  # argmax of [0.8]
    assert result_tuple[2]["action"] == 0  # argmax of [0.7]


def test_transform_multi_agent_with_nested_inputs():
    @transform_multi_agent
    def simple_agent_function(agent_state, observation, shared_param):
        action = jnp.argmax(observation["sensor1"])
        return {
            "new_pos": agent_state["pos"] + action,
            "new_vel": agent_state["vel"] + action,
            "action": action,
            "shared_info": shared_param,
        }

    agent_states_nested = {
        "agent0": {"pos": jnp.array([1.0, 2.0]), "vel": jnp.array([0.1, 0.2])},
        "agent1": {"pos": jnp.array([3.0, 4.0]), "vel": jnp.array([0.3, 0.4])},
        "agent2": {"pos": jnp.array([5.0, 6.0]), "vel": jnp.array([0.5, 0.6])},
    }

    observations_nested = {
        "agent0": {"sensor1": jnp.array([0.1, 0.9]), "sensor2": jnp.array([0.8, 0.2])},
        "agent1": {"sensor1": jnp.array([0.7, 0.3]), "sensor2": jnp.array([0.4, 0.6])},
        "agent2": {"sensor1": jnp.array([0.2, 0.8]), "sensor2": jnp.array([0.9, 0.1])},
    }

    shared_param = 42

    result_nested = simple_agent_function(
        agent_states_nested, observations_nested, shared_param
    )

    # Verify the structure is preserved
    assert isinstance(result_nested, dict)
    assert "agent0" in result_nested
    assert "new_pos" in result_nested["agent0"]
    assert "new_vel" in result_nested["agent0"]

    # Verify the computation is correct
    assert jnp.allclose(result_nested["agent0"]["new_pos"], jnp.array([2.0, 3.0]))
    assert jnp.allclose(result_nested["agent0"]["new_vel"], jnp.array([1.1, 1.2]))
    assert jnp.allclose(result_nested["agent0"]["shared_info"], 42)
    assert jnp.allclose(result_nested["agent1"]["new_pos"], jnp.array([3.0, 4.0]))
    assert jnp.allclose(result_nested["agent1"]["new_vel"], jnp.array([0.3, 0.4]))
    assert jnp.allclose(result_nested["agent1"]["shared_info"], 42)
    assert jnp.allclose(result_nested["agent2"]["new_pos"], jnp.array([6.0, 7.0]))
    assert jnp.allclose(result_nested["agent2"]["new_vel"], jnp.array([1.5, 1.6]))

    # Check actions
    assert result_nested["agent0"]["action"] == 1
    assert result_nested["agent1"]["action"] == 0
    assert result_nested["agent2"]["action"] == 1


def test_transform_multi_agent_with_tuple_return():
    """Test transform_multi_agent with a function that returns a tuple."""
    # This should return a tuple of PyTrees rather than a PyTree of tuples.

    @transform_multi_agent
    def tuple_return_function(agent_state, observation):
        """Function that returns a tuple."""
        return (agent_state + 1, jnp.argmax(observation))

    agent_states = {"agent0": jnp.array([1.0, 2.0]), "agent1": jnp.array([3.0, 4.0])}
    observations = {"agent0": jnp.array([0.8, 0.2]), "agent1": jnp.array([0.3, 0.7])}

    result_tuple_return = tuple_return_function(agent_states, observations)

    # Verify the result is a tuple of pytrees (not a pytree of tuples)
    assert isinstance(result_tuple_return, tuple)
    assert len(result_tuple_return) == 2
    assert isinstance(result_tuple_return[0], dict)
    assert isinstance(result_tuple_return[1], dict)
    assert "agent0" in result_tuple_return[0]
    assert "agent1" in result_tuple_return[0]
    assert "agent0" in result_tuple_return[1]
    assert "agent1" in result_tuple_return[1]

    # Verify the computation is correct
    assert jnp.allclose(result_tuple_return[0]["agent0"], jnp.array([2.0, 3.0]))
    assert jnp.allclose(result_tuple_return[0]["agent1"], jnp.array([4.0, 5.0]))
    assert result_tuple_return[1]["agent0"] == 0
    assert result_tuple_return[1]["agent1"] == 1


def test_transform_multi_agent_with_shared_args():
    """Test transform_multi_agent with explicitly specified shared arguments."""

    @transform_multi_agent(shared_argnames=["observation"])
    def function_with_shared_args(agent_state, observation):
        assert isinstance(observation, dict)
        action0 = jnp.argmax(observation["agent0"])
        action1 = jnp.argmax(observation["agent1"])

        return {
            "new_state": agent_state + action0 + action1,
            "action": (action0, action1),
        }

    agent_states = {"agent0": jnp.array([1.0, 2.0]), "agent1": jnp.array([3.0, 4.0])}
    observations = {"agent0": jnp.array([0.8, 0.2]), "agent1": jnp.array([0.3, 0.7])}

    result = function_with_shared_args(agent_states, observations)

    # Verify shared parameters are not split across agents
    assert result["agent0"]["action"] == (0, 1) == result["agent1"]["action"]
    assert jnp.allclose(result["agent0"]["new_state"], jnp.array([2.0, 3.0]))
    assert jnp.allclose(result["agent1"]["new_state"], jnp.array([4.0, 5.0]))


def test_transform_multi_agent_diff_first_level_structure():
    """Test transform_multi_agent error handling for invalid inputs."""

    @transform_multi_agent
    def simple_function(agent_state, observation):
        return agent_state + observation["agent0"]

    # Test with pytrees of different first-level structure
    agent_states = {"agent0": jnp.array([1.0, 2.0]), "agent1": jnp.array([3.0, 4.0])}

    observations = {
        "agent0": jnp.array([0.1, 0.2]),
        "agent2": jnp.array([0.3, 0.4]),  # Different key structure
    }

    result = simple_function(agent_states, observations)

    # The result should have the structure of the first argument
    assert "agent0" in result
    assert "agent1" in result
    assert "agent2" not in result  # Should not include keys from second argument


def test_transform_multi_agent_with_diff_shapes():
    """Test transform_multi_agent with the same first-level pytree structure but different underneath.
    This should work because the function will use map instead of vmap when the shapes don't match exactly.
    """

    @transform_multi_agent
    def simple_function(agent_state, observation, shared):
        return agent_state + observation + shared

    agent_states = {
        "agent0": jnp.array([1.0, 2.0]),
        "agent1": jnp.array([3.0, 4.0, 5.0]),
    }
    observations = {
        "agent0": jnp.array([0.1, 0.2]),
        "agent1": jnp.array([0.3, 0.4, 0.5]),
    }
    shared = 42

    result = simple_function(agent_states, observations, shared)

    assert "agent0" in result
    assert "agent1" in result
    assert jnp.allclose(result["agent0"], jnp.array([1.1, 2.2]) + shared)
    assert jnp.allclose(result["agent1"], jnp.array([3.3, 4.4, 5.5]) + shared)


def test_transform_multi_agent_with_auto_split_keys():
    """Test transform_multi_agent with auto split keys."""

    @transform_multi_agent(auto_split_keys=True)
    def simple_function(key, agent_state, observation):
        randint = jax.random.randint(key, (1,), 0, 100)
        return (agent_state + observation), (agent_state + observation + randint), key

    agent_states = {"agent0": jnp.ones((2,)), "agent1": jnp.ones((2,))}
    observations = {"agent0": jnp.ones((2,)), "agent1": jnp.ones((2,))}

    key = jax.random.PRNGKey(42)
    result = simple_function(key, agent_states, observations)

    assert isinstance(result, tuple)

    assert jnp.allclose(result[0]["agent0"], result[0]["agent1"])
    assert jnp.allclose(result[0]["agent0"], jnp.ones((2,)) * 2)
    assert not jnp.allclose(result[1]["agent0"], result[1]["agent1"])
    assert not jnp.allclose(result[2]["agent0"], key)
    assert not jnp.allclose(result[2]["agent1"], key)
    assert not jnp.allclose(result[2]["agent0"], result[2]["agent1"])


def test_transform_multi_agent_with_auto_transpose_transitions():
    """Test transform_multi_agent with auto transpose transitions."""

    # Transitions are fed as regular Transition objects, rather than
    # a PyTree of Transitions. Yet these are auto-transposed before being
    # passed to the function and then auto-reconstructed back into the original
    # structure if a Transition object is returned.

    @transform_multi_agent(auto_transpose_transitions=True)
    def simple_function(transition: Transition, additive):
        return transition.observation + transition.action + additive

    transition = Transition(
        observation={  # type: ignore
            "agent0": jnp.ones((128, 5)),
            "agent1": jnp.zeros((128, 5)),
        },
        action={  # type: ignore
            "agent0": jnp.ones((128, 1)),
            "agent1": jnp.zeros((128, 1)),
        },
        reward={  # type: ignore
            "agent0": jnp.ones((128, 1)),
            "agent1": jnp.zeros((128, 1)),
        },
        terminated=jnp.zeros((2,)),
        truncated=jnp.zeros((2,)),
    )
    additive = {"agent0": 10, "agent1": -10}

    result = simple_function(transition, additive)

    assert jnp.allclose(result["agent0"], jnp.ones((128, 5)) + jnp.ones((128, 1)) + 10)
    assert jnp.allclose(
        result["agent1"], jnp.zeros((128, 5)) + jnp.zeros((128, 1)) + -10
    )


def test_transform_multi_agent_with_auto_transpose_transitions_and_return():
    """Test transform_multi_agent with auto transpose transitions."""

    # Transitions are fed as regular Transition objects, rather than
    # a PyTree of Transitions. Yet these are auto-transposed before being
    # passed to the function and then auto-reconstructed back into the original
    # structure if a Transition object is returned.

    @transform_multi_agent(auto_transpose_transitions=True)
    def flat_obs(transition: Transition, additive):
        return replace(
            transition,
            observation=transition.observation.reshape(-1) + additive,
        )

    transition = Transition(
        observation={  # type: ignore
            "agent0": jnp.ones((128, 5)),
            "agent1": jnp.zeros((128, 5)),
        },
        action={  # type: ignore
            "agent0": jnp.ones((128, 1)),
            "agent1": jnp.zeros((128, 1)),
        },
        reward={  # type: ignore
            "agent0": jnp.ones((128, 1)),
            "agent1": jnp.zeros((128, 1)),
        },
        terminated=jnp.zeros((2,)),
        truncated=jnp.zeros((2,)),
    )
    additive = {"agent0": 10, "agent1": -10}

    result = flat_obs(transition, additive)

    assert isinstance(result, Transition)
    assert result.observation["agent0"].shape == (128 * 5,)
    assert jnp.allclose(
        result.observation["agent0"], jnp.ones((128, 5)).reshape(-1) + 10
    )
    assert jnp.allclose(
        result.observation["agent1"], jnp.zeros((128, 5)).reshape(-1) + -10
    )
    assert result.action["agent0"].shape == (128, 1)


def test_transform_multi_agent_with_auto_transpose_transitions_and_return_tuple():
    """Test transform_multi_agent with auto transpose transitions."""

    # Transitions are fed as regular Transition objects, rather than
    # a PyTree of Transitions. Yet these are auto-transposed before being
    # passed to the function and then auto-reconstructed back into the original
    # structure if a Transition object is returned.

    @transform_multi_agent(auto_transpose_transitions=True)
    def flat_obs_extra(transition: Transition, additive):
        some_value = {
            "added": transition.action + additive,
            "removed": transition.action - additive,
        }
        updated_transition = replace(
            transition,
            observation=transition.observation.reshape(-1) + additive,
        )
        return some_value, updated_transition

    transition = Transition(
        observation={  # type: ignore
            "agent0": jnp.ones((128, 5)),
            "agent1": jnp.zeros((128, 5)),
        },
        action={  # type: ignore
            "agent0": jnp.ones((128, 1)),
            "agent1": jnp.zeros((128, 1)),
        },
        reward={  # type: ignore
            "agent0": jnp.ones((128, 1)),
            "agent1": jnp.zeros((128, 1)),
        },
        terminated=jnp.zeros((2,)),
        truncated=jnp.zeros((2,)),
    )
    additive = {"agent0": 10, "agent1": -10}

    result = flat_obs_extra(transition, additive)

    assert isinstance(result, tuple)
    assert len(result) == 2
    some_value, updated_transition = result
    assert isinstance(updated_transition, Transition)

    assert updated_transition.observation["agent0"].shape == (128 * 5,)
    assert jnp.allclose(
        updated_transition.observation["agent0"], jnp.ones((128, 5)).reshape(-1) + 10
    )
    assert jnp.allclose(
        updated_transition.observation["agent1"], jnp.zeros((128, 5)).reshape(-1) + -10
    )
    assert updated_transition.action["agent0"].shape == (128, 1)
    assert jnp.allclose(some_value["agent0"]["added"], jnp.ones((128, 1)) + 10)
    assert jnp.allclose(some_value["agent1"]["added"], jnp.zeros((128, 1)) + -10)
    assert jnp.allclose(some_value["agent0"]["removed"], jnp.ones((128, 1)) - 10)
    assert jnp.allclose(some_value["agent1"]["removed"], jnp.zeros((128, 1)) + 10)
