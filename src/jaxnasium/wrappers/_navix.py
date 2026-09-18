from typing import Any

import jax
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import Observation, TEnvState, TimeStep
from jaxnasium._spaces import Box, Discrete

from ._wrappers import Wrapper


class NavixWrapper(Wrapper):
    """
    Wrapper for Navix environments to transform them into the Jaxnasium environment interface.

    **Arguments:**

    - `_env`: Navix environment.
    """

    _env: Any

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        timestep_navix = self._env.reset(key)
        return timestep_navix.observation, timestep_navix

    def step(self, key: PRNGKeyArray, state: Any, action: int) -> tuple[TimeStep, Any]:
        timestep_navix = self._env._step(state, action)
        obs = timestep_navix.observation
        reward = timestep_navix.reward
        terminated = timestep_navix.step_type == 2
        truncated = timestep_navix.step_type == 1
        info = timestep_navix.info
        timestep_step = TimeStep(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        timestep, state = self.auto_reset(key, timestep_step, timestep_navix)
        return timestep, state

    @property
    def observation_space(self) -> Box:
        # ensuring space properties are not tracers:
        with jax.ensure_compile_time_eval():
            space = self._env.observation_space
            return Box(
                low=space.minimum,
                high=space.maximum,
                shape=space.shape,
                dtype=space.dtype,
            )

    @property
    def action_space(self) -> Discrete:
        with jax.ensure_compile_time_eval():
            num_actions = self._env.action_space.maximum
            # Add the "done" no-op action which is outside of the Navix action space (?)
            num_actions += 1
            return Discrete(num_actions)
