"""Input/output coverage for the algorithm networks.

These obs/action spaces cover the paths we want supported in terms of possible
input/output networks and multi-agent support. Each one is handed to
`check_env`, the same function every registry environment goes through, so a
space is tested here exactly the way a real environment is tested elsewhere.

`check_env` is structural: it asserts what gets built and what shape comes out,
abstractly, without compiling. What it cannot see is *values* -- a mask applied to
the wrong axis with a coincidentally valid shape. `TestActionMasking` below covers
that with real compiled execution on deliberately restrictive masks, and is the only
part of this file that runs anything.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from _proxy_test_envs import (
    MULTI_AGENT_ACT_SPACES,
    MULTI_AGENT_OBS_SPACES,
    SINGLE_AGENT_ACT_SPACES,
    SINGLE_AGENT_OBS_SPACES,
    make_proxy_env,
    mirror_agent_structure,
    obs_agent_observation_critic,
    obs_agent_observation_vector,
    obs_box_vector,
)
from _test_utils import check_env

from jaxnasium import AgentObservation, Discrete
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
