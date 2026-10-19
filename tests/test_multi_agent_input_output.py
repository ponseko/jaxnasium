"""Input/output coverage for the algorithm networks.

These obs/action spaces cover the paths we want supported in terms of possible
input/output networks and multi-agent support. Each one is handed to
`check_env`, the same function every registry environment goes through, so a
space is tested here exactly the way a real environment is tested elsewhere.

`check_env` is structural: it asserts what gets built and what
shape comes out, abstractly, without compiling.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _consts import AGENT_MIN_CONFIG
from _proxy_test_envs import (
    MULTI_AGENT_ACT_SPACES,
    MULTI_AGENT_OBS_SPACES,
    SINGLE_AGENT_ACT_SPACES,
    SINGLE_AGENT_OBS_SPACES,
    make_proxy_env,
    mirror_agent_structure,
    obs_agent_observation_critic,
    obs_agent_observation_multidiscrete_mask,
    obs_agent_observation_vector,
    obs_box_vector,
)
from _test_utils import check_env

from jaxnasium import AgentObservation, Discrete
from jaxnasium.algorithms import PPO
from jaxnasium.algorithms.agent_networks import (
    ActorNetwork,
    QValueNetwork,
    ValueNetwork,
)

SEED = jax.random.PRNGKey(0)
default_obs_space = obs_box_vector

INPUT_SPACE_CASES = [
    *[pytest.param(fn, False, id=name) for name, fn in SINGLE_AGENT_OBS_SPACES.items()],
    *[pytest.param(fn, True, id=name) for name, fn in MULTI_AGENT_OBS_SPACES.items()],
]


@pytest.mark.parametrize("obs_space_fn, multi_agent", INPUT_SPACE_CASES)
def test_various_input_spaces(obs_space_fn: Callable[[], Any], multi_agent: bool):
    env = make_proxy_env(obs_space_fn, multi_agent=multi_agent)
    check_env(env)


OUTPUT_SPACE_CASES = [
    *[pytest.param(fn, False, id=name) for name, fn in SINGLE_AGENT_ACT_SPACES.items()],
    *[pytest.param(fn, True, id=name) for name, fn in MULTI_AGENT_ACT_SPACES.items()],
]


@pytest.mark.parametrize("action_space_fn, multi_agent", OUTPUT_SPACE_CASES)
def test_various_output_spaces(action_space_fn: Callable[[], Any], multi_agent: bool):
    if multi_agent:
        obs_space = mirror_agent_structure(action_space_fn(), default_obs_space)
        obs_space_fn: Callable[[], Any] = lambda: obs_space
    else:
        obs_space_fn = default_obs_space

    env = make_proxy_env(obs_space_fn, action_space_fn, multi_agent=multi_agent)
    check_env(env)


def test_critic_networks_build_over_critic_observation_space():
    obs_space = obs_agent_observation_critic()
    actor = ActorNetwork(obs_space, Discrete(4), key=SEED)
    value = ValueNetwork(obs_space, key=SEED)
    q_value = QValueNetwork(obs_space, Discrete(4), key=SEED)

    assert actor.obs_processor.out_features == 6
    assert value.obs_processor.out_features == 12
    assert q_value.obs_processor.out_features == 12

    obs = AgentObservation(
        observation=jnp.ones((6,)), critic_observation=jnp.ones((12,))
    )
    actor(obs)
    value(obs)
    q_value(obs)


def test_critic_networks_fall_back_to_observation_without_critic_observation():
    obs_space = obs_agent_observation_vector()
    value = ValueNetwork(obs_space, key=SEED)
    q_value = QValueNetwork(obs_space, Discrete(4), key=SEED)

    assert value.obs_processor.out_features == 6
    assert q_value.obs_processor.out_features == 6


@pytest.mark.parametrize(
    "obs_space_fn",
    [obs_agent_observation_vector, obs_agent_observation_multidiscrete_mask],
    ids=["discrete", "multidiscrete"],
)
def test_actions_respect_action_mask(obs_space_fn: Callable[[], Any]):
    """Random masks over a batch of observations"""
    env = make_proxy_env(obs_space_fn)
    num_obs = 64

    @eqx.filter_jit
    def masked_actions(key):
        init_key, obs_key, action_key = jax.random.split(key, 3)
        agent = PPO(**AGENT_MIN_CONFIG).init_agent(init_key, env)
        obs = jax.vmap(env.sample_observation)(jax.random.split(obs_key, num_obs))
        action_keys = jax.random.split(action_key, num_obs)
        return obs.action_mask, jax.vmap(agent.get_action)(action_keys, obs)

    mask, actions = jax.device_get(masked_actions(SEED))
    chosen = np.take_along_axis(mask, actions[..., None], axis=-1)[..., 0]
    unconstrained = ~mask.any(axis=-1)
    assert (chosen | unconstrained).all(), "Agent picked a masked-out action"
