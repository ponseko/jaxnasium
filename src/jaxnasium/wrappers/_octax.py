from typing import Any

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import Observation, TEnvState, TimeStep
from jaxnasium._spaces import Box, Discrete, Space

from ._wrappers import Wrapper


class OctaxWrapper(Wrapper):
    """
    Wrapper for [Octax](https://github.com/riiswa/octax) CHIP-8 environments.

    Octax stores the CHIP-8 display as `(width, height)`, so observations are
    transposed to the usual channels-first image layout `(frame_skip, height, width)`.
    This mirrors their GymnaxWrapper.

    **Arguments:**

    - `_env`: Octax environment.
    """

    _env: Any

    @staticmethod
    def _to_image(obs):
        """Transpose `(frame_skip, width, height)` frames to `(frame_skip, height, width)`."""
        return obs.transpose(0, 2, 1)

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        state, obs, _info = self._env.reset(key)
        return obs.transpose(0, 2, 1), state

    def step(self, key: PRNGKeyArray, state: Any, action: int) -> tuple[TimeStep, Any]:
        state_step, obs, reward, terminated, truncated, info = self._env.step(
            state, action
        )
        timestep_step = TimeStep(
            observation=obs.transpose(0, 2, 1),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        timestep, state = self.auto_reset(key, timestep_step, state_step)
        return timestep, state

    @property
    def observation_space(self) -> Space:
        display = self._env.cached_reset_state.display  # (width, height)
        shape = (self._env.frame_skip, *reversed(display.shape))
        with jax.ensure_compile_time_eval():
            return Box(
                low=np.zeros(shape, dtype=display.dtype),
                high=np.ones(shape, dtype=display.dtype),
                shape=shape,
                dtype=display.dtype,
            )

    @property
    def action_space(self) -> Discrete:
        return Discrete(self._env.num_actions)
