from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import Observation, TimeStep
from jaxnasium._spaces import Box

from ._wrappers import Wrapper


class BraxWrapperState(eqx.Module):
    brax_env_state: Any  # The state of the Brax environment
    timestep: int = 0


class BraxWrapper(Wrapper):
    """
    Wrapper for Brax environments to transform them into the Jaxnasium environment interface.

    Note: Brax environments would typically be wrapped with a VmapWrapper, EpisodeWrapper and AutoResetWrapper
    VmapWrapper is not included here, as it is replaced by Jaxnasium's `VecEnvWrapper`.
    The effects of EpisodeWrapper (truncation) and AutoResetWrapper are merged into this wrapper.

    **Arguments:**

    - `_env`: Brax environment.
    """

    _env: Any
    max_episode_steps: int = 1000  # Brax defaults to 1000

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, BraxWrapperState]:
        env_state = self._env.reset(key)
        env_state = BraxWrapperState(brax_env_state=env_state, timestep=0)
        return env_state.brax_env_state.obs, env_state

    def step(
        self, key: PRNGKeyArray, state: BraxWrapperState, action: float
    ) -> tuple[TimeStep, BraxWrapperState]:
        brax_env_state = self._env.step(state.brax_env_state, action)
        state_step = BraxWrapperState(
            brax_env_state=brax_env_state,
            timestep=state.timestep + 1,
        )
        truncated = state_step.timestep >= self.max_episode_steps
        terminated = brax_env_state.done
        info = dict(brax_env_state.info)

        timestep_step = TimeStep(
            observation=brax_env_state.obs,
            reward=brax_env_state.reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        timestep, state = self.auto_reset(key, timestep_step, state_step)
        return timestep, state

    @property
    def observation_space(self) -> Any:
        shape = (self._env.observation_size,)
        return Box(low=-np.inf, high=np.inf, shape=shape, dtype=jnp.float32)

    @property
    def action_space(self) -> Any:
        sys = self._env.sys
        limited = np.asarray(sys.actuator_ctrllimited, dtype=bool)
        ctrl_range = np.asarray(sys.actuator_ctrlrange, dtype=np.float32)
        low = np.where(limited, ctrl_range[:, 0], -np.inf)
        high = np.where(limited, ctrl_range[:, 1], np.inf)
        return Box(low=low, high=high, shape=low.shape, dtype=jnp.float32)
