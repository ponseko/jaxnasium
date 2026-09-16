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
from jaxnasium._spaces import Discrete, MultiDiscrete
from jaxnasium._types import AgentObservation

from ._util import gymnasium_to_jaxnasium_space
from ._wrappers import Wrapper


class JumanjiWrapper(Wrapper):
    """
    Wrapper for Jumanji environments to transform them into the Jaxnasium environment interface.

    **Arguments:**

    - `_env`: Jumanji environment.
    """

    _env: Any

    def __init__(self, env: Any):
        from jumanji.wrappers import AutoResetWrapper  # type: ignore

        self._env = AutoResetWrapper(env, next_obs_in_extras=True)

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
            # Convert it to a dict and collect the action mask
            action_mask = getattr(obs, "action_mask", None)
            obs = {
                key: convert_jumanji_obs_to_dict(value)
                for key, value in obs._asdict().items()  # pyright: ignore[reportAttributeAccessIssue]
                if key != "action_mask"
            }
            if action_mask is not None:
                obs = AgentObservation(observation=obs, action_mask=action_mask)
        return obs  # type: ignore[reportGeneralTypeIssues]

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        state, timestep = self._env.reset(key)
        observation = self._convert_jumanji_obs(timestep.observation)
        return observation, state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: float
    ) -> tuple[TimeStep, TEnvState]:
        state, timestep = self._env.step(state, action)  # No key for Jumanji
        obs = self._convert_jumanji_obs(timestep.observation)

        truncated = jnp.logical_and(timestep.discount != 0, timestep.step_type == 2)
        terminated = jnp.logical_and(timestep.step_type == 2, ~truncated)

        info = timestep.extras
        info["DISCOUNT"] = timestep.discount
        next_obs = info.pop("next_obs", None)
        info[ORIGINAL_OBSERVATION_KEY] = self._convert_jumanji_obs(next_obs)

        timestep = TimeStep(
            observation=obs,
            reward=timestep.reward,
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
        action_mask_space = space.pop("action_mask", None)  # pyright: ignore[reportAttributeAccessIssue]
        if action_mask_space is not None:
            space = AgentObservation(observation=space, action_mask=action_mask_space)
        return space

    @property
    def action_space(self) -> Any:
        from jumanji.specs import jumanji_specs_to_gym_spaces  # type: ignore

        with jax.ensure_compile_time_eval():
            space = self._env.action_spec
            space = jumanji_specs_to_gym_spaces(space)
        space = jax.tree.map(self.__convert_gymnasium_int_box_to_discrete, space)
        space = self.__convert_gymnasium_space_to_dict(space)
        return gymnasium_to_jaxnasium_space(space)
