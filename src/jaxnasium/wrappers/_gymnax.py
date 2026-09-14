from typing import Any

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import Observation, TEnvState, TimeStep
from jaxnasium._spaces import Space

from ._util import gymnasium_to_jaxnasium_space
from ._wrappers import Wrapper


class GymnaxWrapper(Wrapper):
    """
    Wrapper for Gymnax environments to transform them into the Jaxnasium environment interface.

    **Arguments:**

    - `_env`: Gymnax environment.
    - `handle_truncation`: If True, the wrapper will reimplement the autoreset behavior to include
        truncated information and the terminal_observation in the info dictionary. If False, the wrapper will mirror
        the Gymnax behavior by ignoring truncations. Default=True.
    """

    _env: Any
    handle_truncation: bool = True

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        params = getattr(self._env, "default_params", None)
        obs, env_state = self._env.reset(key, params)
        return obs, env_state

    def step(
        self, key: PRNGKeyArray, state: Any, action: float
    ) -> tuple[TimeStep, Any]:
        _params = self._env.default_params  # is dataclass
        original_max_steps = getattr(_params, "max_steps_in_episode", None)

        if not self.handle_truncation or original_max_steps is None:
            obs, state_step, reward, done, info = self._env.step_env(
                key, state, action, _params
            )
            terminated, truncated = done, False

            timestep_step = TimeStep(
                observation=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            timestep, state = self.auto_reset(key, timestep_step, state_step)
            return timestep, state

        # Handle truncation:
        # We increase max_steps_in_episode by 1 so that the done flag from Gymnax
        # only triggers when the episode terminates without truncation.
        # Then we set truncate manually in this wrapper based on the original max_steps_in_episode.
        altered_params = eqx.tree_at(
            lambda x: x.max_steps_in_episode, _params, replace_fn=lambda x: x + 1
        )
        obs_step, state_step, reward, done, info = self._env.step_env(
            key, state, action, altered_params
        )
        terminated = done  # did not truncate due to the +1 in max_steps_in_episode
        truncated = state_step.time >= original_max_steps

        timestep_step = TimeStep(
            observation=obs_step,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        timestep, state = self.auto_reset(key, timestep_step, state_step)
        return timestep, state

    @property
    def observation_space(self) -> Space:
        params = self._env.default_params
        # ensuring space properties are not tracers:
        with jax.ensure_compile_time_eval():
            space = self._env.observation_space(params)
        return gymnasium_to_jaxnasium_space(space)

    @property
    def action_space(self) -> Space:
        params = self._env.default_params
        with jax.ensure_compile_time_eval():
            space = self._env.action_space(params)
        return gymnasium_to_jaxnasium_space(space)
