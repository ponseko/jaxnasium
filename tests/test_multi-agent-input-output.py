from typing import List

import _consts as TEST_CONSTS
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, PRNGKeyArray

from jaxnasium import Discrete, Environment, MultiDiscrete, Space, TimeStep
from jaxnasium.algorithms import PPO


class SimpleEnvState(eqx.Module):
    last_actions: Array
    current_step: int = 0


class SimpleMultiAgentEnv(Environment):
    num_agents: int = 3
    episode_length: int = 100
    _multi_agent: bool = True

    def reset_env(self, key: PRNGKeyArray):
        state = SimpleEnvState(
            last_actions=jnp.ones((self.num_agents,), dtype=jnp.int32) * -1,
            current_step=0,
        )
        return self.get_observation(state), state

    def step_env(self, key: PRNGKeyArray, state: SimpleEnvState, action: Array):
        action_array = jnp.array(action, dtype=jnp.int32)
        # Reward is 1 for each agent if they all take the same action
        reward = jnp.sum(jnp.all(action_array == action_array[0]))
        reward = [reward for _ in range(self.num_agents)]

        new_state = SimpleEnvState(
            last_actions=action_array,
            current_step=state.current_step + 1,
        )

        truncated = new_state.current_step >= self.episode_length

        return TimeStep(
            self.get_observation(new_state),
            reward,
            False,  # terminated
            truncated,  # truncated
            {},
        ), new_state

    def get_observation(self, state: SimpleEnvState):
        # Each agent observes the last actions of all agents
        actions = state.last_actions
        obs = [actions for _ in range(self.num_agents)]

        return obs

    @property
    def action_space(self) -> List[Space]:
        num_actions = 3
        return [Discrete(num_actions) for _ in range(self.num_agents)]

    @property
    def observation_space(self) -> List[MultiDiscrete]:
        num_actions = 3
        nvec = np.array([num_actions] * self.num_agents)
        # Each agent observes the last action of the other two agents (vector of length 2)
        return [MultiDiscrete(nvec) for _ in range(self.num_agents)]


class HeterogeneousMultiAgentEnvState(eqx.Module):
    agent0_state: Array  # Discrete state for agent 0
    agent1_state: Array  # Discrete state for agent 1
    current_step: int = 0


class HeterogeneousMultiAgentSimpleEnv(Environment):
    num_agents: int = 2
    episode_length: int = 50
    _multi_agent: bool = True

    def reset_env(self, key: PRNGKeyArray):
        # Use dict keys to access spaces for sampling
        state = HeterogeneousMultiAgentEnvState(
            agent0_state=jnp.array(-1, dtype=jnp.int32),
            agent1_state=jnp.array(-1, dtype=jnp.int32),
            current_step=0,
        )
        return self.get_observation(state), state

    def step_env(
        self,
        key: PRNGKeyArray,
        state: HeterogeneousMultiAgentEnvState,
        action: dict[str, Array],
    ):
        action0 = action["agent0"]
        action1 = action["agent1"]

        # Agent 0's state changes based on its action (e.g., action 0 -> state 0, action 1 -> state 1)
        new_agent0_state = (action0).astype(jnp.int32)

        # Agent 1's state changes based on its action (e.g., sum of its multi-discrete actions modulo 3)
        new_agent1_state = jnp.sum(action1) % 3

        new_state = HeterogeneousMultiAgentEnvState(
            agent0_state=new_agent0_state,
            agent1_state=new_agent1_state,
            current_step=state.current_step + 1,
        )

        # Simple reward: +1 if both agents are in state 1
        reward_val = jnp.where(
            (new_agent0_state == 1) & (new_agent1_state == 1), 1.0, 0.0
        )
        reward = {"agent0": reward_val, "agent1": reward_val}

        truncated = new_state.current_step >= self.episode_length
        terminated = False

        return TimeStep(
            self.get_observation(new_state),
            reward,
            terminated,
            truncated,
            {},
        ), new_state

    def get_observation(self, state: HeterogeneousMultiAgentEnvState):
        # Agent 0 observes its own state
        obs0 = jnp.atleast_1d(state.agent0_state)
        # Agent 1 observes both states
        obs1 = jnp.array([state.agent0_state, state.agent1_state])
        return {"agent0": obs0, "agent1": obs1}

    @property
    def action_space(self) -> dict[str, Space]:
        # Agent 0: Discrete action (0 or 1)
        space0 = Discrete(2)
        # Agent 1: MultiDiscrete action (two actions, each with 2 choices)
        space1 = MultiDiscrete(np.array([2, 2]))
        return {"agent0": space0, "agent1": space1}

    @property
    def observation_space(self) -> dict[str, Space]:
        # Agent 0: Discrete observation (state 0 or 1)
        space0 = Discrete(2)
        # Agent 1: MultiDiscrete observation (observes agent 0 state [0,1] and its own state [0,1,2])
        space1 = MultiDiscrete(np.array([2, 3]))
        return {"agent0": space0, "agent1": space1}


class PyTreeObsSpaceEnv(Environment):
    """
    A simple environment where the observation space is a PyTree.
    This is useful for testing if the agent can handle PyTree observations.
    """

    episode_length: int = 100

    def reset_env(self, key: PRNGKeyArray):
        state = {"position": jnp.ones(()), "velocity": jnp.zeros(()), "current_step": 0}
        return state, state

    def step_env(self, key: PRNGKeyArray, state: dict[str, Array], action: Array):
        new_state = {
            "position": state["position"] + action,
            "velocity": state["velocity"] + action * 0.1,
            "current_step": state["current_step"] + 1,
        }
        reward = jnp.sum(new_state["position"])

        truncated = new_state["current_step"] >= self.episode_length

        return TimeStep(new_state, reward, False, truncated, {}), new_state

    @property
    def action_space(self) -> Space:
        return Discrete(3)

    @property
    def observation_space(self) -> dict[str, Space]:
        return {
            "position": Discrete(10),
            "velocity": Discrete(10),
            "current_step": Discrete(100),
        }


class PyTreeActionSpaceEnv(Environment):
    """
    A simple environment where the action space is a PyTree.
    This is useful for testing if the agent can handle PyTree actions.
    """

    episode_length: int = 100

    def reset_env(self, key: PRNGKeyArray):
        state = {"position": jnp.ones(()), "velocity": jnp.zeros(()), "current_step": 0}
        return state, state

    def step_env(
        self, key: PRNGKeyArray, state: dict[str, Array], action: dict[str, Array]
    ):
        new_state = {
            "position": state["position"] + action["move"],
            "velocity": state["velocity"] + action["turn"] * 0.1,
            "current_step": state.get("current_step", 0) + 1,
        }

        truncated = new_state["current_step"] >= self.episode_length

        reward = jnp.sum(new_state["position"])
        return TimeStep(new_state, reward, False, truncated, {}), new_state

    @property
    def action_space(self) -> dict[str, Space]:
        return {"move": Discrete(3), "turn": Discrete(2)}

    @property
    def observation_space(self) -> dict[str, Space]:
        return {
            "position": Discrete(10),
            "velocity": Discrete(10),
            "current_step": Discrete(100),
        }


class SimpleMultiAgentEnvWithPyTreeAction(Environment):
    """
    A simple multi-agent environment where each agent has a PyTree action space.
    This is useful for testing if the agent can handle PyTree actions in a multi-agent setting.
    """

    num_agents: int = 2
    episode_length: int = 100
    _multi_agent: bool = True

    def reset_env(self, key: PRNGKeyArray):
        state = {"agent0": jnp.ones((1,)), "agent1": jnp.zeros((1,)), "current_step": 0}
        obs = {
            "agent0": jnp.ones((1,)),
            "agent1": jnp.zeros((1,)),
        }
        return obs, state

    def step_env(
        self, key: PRNGKeyArray, state: dict[str, Array], action: dict[str, Array]
    ):
        new_state = {
            "agent0": state["agent0"]
            + action["agent0"]["move"]
            + action["agent0"]["turn"][0] * 0.1,
            "agent1": state["agent1"] + action["agent1"]["turn"] * 0.1,
            "current_step": state.get("current_step", 0) + 1,
        }
        # obs is state without the current step
        obs = {
            "agent0": new_state["agent0"],
            "agent1": new_state["agent1"],
        }
        truncated = new_state["current_step"] >= self.episode_length
        reward = {
            "agent0": jnp.sum(new_state["agent0"]),
            "agent1": jnp.sum(new_state["agent1"]),
        }
        return TimeStep(obs, reward, False, truncated, {}), new_state

    @property
    def action_space(self) -> dict:
        return {
            "agent0": {"move": Discrete(3), "turn": MultiDiscrete(np.array([2, 2]))},
            "agent1": {"move": Discrete(3), "turn": Discrete(2)},
        }

    @property
    def observation_space(self) -> dict[str, Space]:
        return {"agent0": Discrete(10), "agent1": Discrete(10)}


CUSTOM_TEST_ENVS = [
    SimpleMultiAgentEnv,
    HeterogeneousMultiAgentSimpleEnv,
    PyTreeObsSpaceEnv,
    PyTreeActionSpaceEnv,
    SimpleMultiAgentEnvWithPyTreeAction,
]


@pytest.mark.parametrize("env_cls", CUSTOM_TEST_ENVS)
def test_custom_envs_runs(env_cls):
    env: Environment = env_cls()
    obs, state = env.reset(jax.random.PRNGKey(0))
    for i in range(5):
        key = jax.random.PRNGKey(i)
        actions = env.sample_action(key)
        (obs, reward, terminated, truncated, info), state = env.step(
            key, state, actions
        )


@pytest.mark.parametrize("alg_cls", TEST_CONSTS.DISCRETE_ALGS)
@pytest.mark.parametrize("env_cls", CUSTOM_TEST_ENVS)
def test_custom_envs_w_algs(alg_cls, env_cls):
    env = env_cls()
    seed = jax.random.PRNGKey(42)
    if alg_cls is PPO:
        agent = alg_cls(**TEST_CONSTS.PPO_MIN_CONFIG)
    else:
        agent = alg_cls(total_timesteps=1000, log_function=None, num_envs=2)

    try:
        agent = agent.train(seed, env)
    except Exception as e:
        pytest.fail(
            f"{alg_cls.__name__} training failed on {env_cls.__name__} with error: {e}"
        )

    # Evaluate agent for 5 episodes (since it's a toy env)
    key = jax.random.PRNGKey(42)
    agent.evaluate(key, env, num_eval_episodes=5)
    # No strict threshold, just check it runs and returns a float
