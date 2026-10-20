from dataclasses import asdict, is_dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import (
    ORIGINAL_OBSERVATION_KEY,
    Observation,
    TEnvState,
    TimeStep,
)
from jaxnasium._spaces import Box, Discrete, MultiDiscrete
from jaxnasium._types import AgentObservation

from ._util import gymnasium_to_jaxnasium_space
from ._wrappers import Wrapper


def _with_action_mask(obs: dict) -> Any:
    action_mask = obs.pop("action_mask", None)
    if action_mask is None:
        return obs
    return AgentObservation(observation=obs, action_mask=action_mask)


def _agent_space(space: Any, i: int) -> Any:
    """The space of agent `i`, given a space with a leading agent axis."""
    if isinstance(space, MultiDiscrete):
        nvec = np.asarray(space.nvec)[i]
        return Discrete(int(nvec)) if nvec.ndim == 0 else MultiDiscrete(nvec)
    low, high = (np.broadcast_to(b, space.shape)[i] for b in (space.low, space.high))
    return Box(low, high, space.shape[1:], space.dtype)


class JumanjiWrapper(Wrapper):
    """
    Wrapper for Jumanji environments to transform them into the Jaxnasium environment interface.
    Multi-agent Jumanji envs (with `num_agents`) become multi-agent, with agents `agent_{i}`.

    **Arguments:**

    - `_env`: Jumanji environment.
    """

    _env: Any
    _num_agents: int  # 0 for single-agent envs

    def __init__(self, env: Any):
        from jumanji.wrappers import AutoResetWrapper  # type: ignore

        self._env = AutoResetWrapper(env, next_obs_in_extras=True)
        self._num_agents = int(getattr(env, "num_agents", 0))

    @property
    def multi_agent(self) -> bool:
        return self._num_agents > 0

    def _per_agent(self, obs: dict, index: Any) -> dict:
        """Split per-agent fields of a dict observation (or space); shared fields are copied."""
        PER_AGENT_FIELDS = {
            "action_mask",
            "agents_view",
            "agents_locations",
            "positions",
            "searcher_views",
        }
        return {
            f"agent_{i}": _with_action_mask(
                {k: index(v, i) if k in PER_AGENT_FIELDS else v for k, v in obs.items()}
            )
            for i in range(self._num_agents)
        }

    def _convert_jumanji_obs(self, obs: Any) -> Observation:
        def convert_jumanji_obs_to_dict(obs: Any) -> Any:
            """Recursively convert Jumanji observations to regular dicts."""
            if isinstance(obs, tuple) and hasattr(obs, "_asdict"):  # NamedTuple
                return {
                    key: convert_jumanji_obs_to_dict(value)
                    for key, value in obs._asdict().items()  # pyright: ignore[reportAttributeAccessIssue]
                }
            elif is_dataclass(obs):
                return {
                    key: convert_jumanji_obs_to_dict(value)
                    for key, value in asdict(obs).items()  # pyright: ignore
                }
            return obs

        if isinstance(obs, tuple) and hasattr(obs, "_asdict"):  # NamedTuple
            obs = convert_jumanji_obs_to_dict(obs)
            if self._num_agents:
                return self._per_agent(obs, lambda x, i: x[i])
            return _with_action_mask(obs)
        return obs  # type: ignore[reportGeneralTypeIssues]

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        state, timestep = self._env.reset(key)
        observation = self._convert_jumanji_obs(timestep.observation)
        return observation, state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: float
    ) -> tuple[TimeStep, TEnvState]:
        n = self._num_agents
        if n:  # Jumanji takes the actions stacked over the agent axis
            action = jnp.stack([action[f"agent_{i}"] for i in range(n)])  # type: ignore
        state, timestep = self._env.step(state, action)  # No key for Jumanji
        obs = self._convert_jumanji_obs(timestep.observation)

        reward = timestep.reward
        truncated = jnp.logical_and(timestep.discount != 0, timestep.step_type == 2)
        terminated = jnp.logical_and(timestep.step_type == 2, ~truncated)
        if n:  # Per agent; shared values are copied to every agent
            split = lambda x: {
                f"agent_{i}": jnp.broadcast_to(x, (n,))[i] for i in range(n)
            }
            reward, terminated, truncated = map(split, (reward, terminated, truncated))

        info = timestep.extras
        info["DISCOUNT"] = timestep.discount
        next_obs = info.pop("next_obs", None)
        info[ORIGINAL_OBSERVATION_KEY] = self._convert_jumanji_obs(next_obs)

        timestep = TimeStep(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return timestep, state

    def __convert_gymnasium_space_to_dict(self, space: Any) -> Any:
        """Recursively convert Gymnasium Dict spaces to regular dicts."""
        from gymnasium.spaces import Dict as GymnasiumDict  # type: ignore

        if isinstance(space, GymnasiumDict):
            # Recursively convert nested spaces and exclude action_mask
            return {
                k: self.__convert_gymnasium_space_to_dict(v)
                for k, v in space.spaces.items()
            }
        return space

    def __convert_gymnasium_int_box_to_discrete(self, space: Any) -> Any:
        """Converts a gymnasium Box space with integer dtypes to a (multi-)discrete space."""
        if space.__class__.__name__ != "Box" or not jnp.isdtype(
            space.dtype, "integral"
        ):
            return space
        assert np.all(np.asarray(space.low) == 0), (
            "Cannot convert Box space with non-zero low to Discrete space"
        )
        high = np.broadcast_to(np.asarray(space.high), space.shape).astype(int)
        if space.shape in ((), (1,)):
            return Discrete(int(high.reshape(-1)[0]))
        return MultiDiscrete(high)

    @property
    def observation_space(self) -> Any:
        from jumanji.specs import jumanji_specs_to_gym_spaces  # type: ignore

        # ensuring space properties are not tracers:
        with jax.ensure_compile_time_eval():
            space = self._env.observation_spec
            space = jumanji_specs_to_gym_spaces(space)
        space = self.__convert_gymnasium_space_to_dict(space)
        space = gymnasium_to_jaxnasium_space(space)
        if self._num_agents:
            return self._per_agent(space, _agent_space)  # type: ignore
        return _with_action_mask(space)  # type: ignore

    @property
    def action_space(self) -> Any:
        from jumanji.specs import jumanji_specs_to_gym_spaces  # type: ignore

        with jax.ensure_compile_time_eval():
            space = self._env.action_spec
            space = jumanji_specs_to_gym_spaces(space)
        space = jax.tree.map(self.__convert_gymnasium_int_box_to_discrete, space)
        space = self.__convert_gymnasium_space_to_dict(space)
        space = gymnasium_to_jaxnasium_space(space)
        if self._num_agents:
            return {
                f"agent_{i}": _agent_space(space, i) for i in range(self._num_agents)
            }
        return space
