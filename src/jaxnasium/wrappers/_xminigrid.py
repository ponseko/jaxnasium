from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import Observation, TEnvState, TimeStep
from jaxnasium._spaces import Box, Discrete

from ._wrappers import Wrapper


class xMinigridWrapper(Wrapper):
    """
    Wrapper for xMinigrid environments to transform them into the Jaxnasium environment interface.

    **Arguments:**

    - `_env`: xMinigrid environment.
    - `_params`: xMinigrid environment parameters.
    """

    _env: Any
    _params: Any

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        timestep_xminigrid = self._env.reset(self._params, key)
        return timestep_xminigrid.observation, timestep_xminigrid

    def step(self, key: PRNGKeyArray, state: Any, action: int) -> tuple[TimeStep, Any]:
        timestep_x = self._env.step(self._params, state, action)  # key is in state
        truncated = jnp.logical_and(timestep_x.discount != 0, timestep_x.step_type == 2)
        terminated = jnp.logical_and(timestep_x.step_type == 2, ~truncated)

        timestep_step = TimeStep(
            observation=timestep_x.observation,
            reward=timestep_x.reward,
            terminated=terminated,
            truncated=truncated,
            info={},
        )
        timestep, state = self.auto_reset(key, timestep_step, timestep_x)
        return timestep, state

    @property
    def observation_space(self) -> Box:
        obs_shape = self._env.observation_shape(self._params)
        return Box(
            low=np.full(obs_shape, -10),
            high=np.full(obs_shape, 10),
            shape=obs_shape,
            dtype=jnp.int32,
        )

    @property
    def action_space(self) -> Discrete:
        return Discrete(self._env.num_actions(self._params))
