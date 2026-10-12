from typing import Any

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import Observation, TEnvState, TimeStep
from jaxnasium._spaces import Space

from ._util import gymnasium_to_jaxnasium_space
from ._wrappers import Wrapper

# Step-limit field on the env params and step-counter field on the env state.
# Libraries stick to different names here:
_MAX_STEPS_FIELDS = ("max_steps_in_episode", "max_timesteps")
_TIME_FIELDS = ("time", "timestep")


class GymnaxWrapper(Wrapper):
    """
    Wrapper for Gymnax environments to transform them into the Jaxnasium environment interface.

    **Arguments:**

    - `_env`: Gymnax environment.
    - `handle_truncation`: If True, the wrapper will reimplement the autoreset behavior to include
        truncated information and the terminal_observation in the info dictionary. If False, the wrapper will mirror
        the Gymnax behavior by ignoring truncations. Default=True.
    - `_params`: Gymnax environment parameters. If None, the environment's `default_params` are used.
    """

    _env: Any
    handle_truncation: bool = True
    _params: Any = None

    @property
    def params(self) -> Any:
        """The env params passed to the Gymnax environment."""
        if self._params is not None:
            return self._params
        return getattr(self._env, "default_params", None)

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        obs, env_state = self._env.reset(key, self.params)
        return obs, env_state

    def step(
        self, key: PRNGKeyArray, state: Any, action: float
    ) -> tuple[TimeStep, Any]:
        _params = self.params  # is dataclass
        max_steps_field = next(
            (f for f in _MAX_STEPS_FIELDS if hasattr(_params, f)), None
        )

        if not self.handle_truncation or max_steps_field is None:
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
        # We increase the step limit by 1 so that the done flag from Gymnax
        # only triggers when the episode terminates without truncation.
        # Then we set truncate manually in this wrapper based on the original step limit.
        original_max_steps = getattr(_params, max_steps_field)
        altered_params = eqx.tree_at(
            lambda x: getattr(x, max_steps_field), _params, replace_fn=lambda x: x + 1
        )
        obs_step, state_step, reward, done, info = self._env.step_env(
            key, state, action, altered_params
        )
        terminated = done  # did not truncate due to the +1 in the step limit
        time_field = next((f for f in _TIME_FIELDS if hasattr(state_step, f)), "time")
        truncated = getattr(state_step, time_field) >= original_max_steps

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
        # ensuring space properties are not tracers:
        with jax.ensure_compile_time_eval():
            space = self._env.observation_space(self.params)
        return gymnasium_to_jaxnasium_space(space)

    @property
    def action_space(self) -> Space:
        with jax.ensure_compile_time_eval():
            space = self._env.action_space(self.params)
        return gymnasium_to_jaxnasium_space(space)
