import jax
import jax.numpy as jnp
import numpy as np

import jaxnasium as jym

from ._brax import BraxWrapper


class PlaygroundWrapper(BraxWrapper):
    # TODO: add option for adding privileged state (i.e. to critic_obs)
    @property
    def observation_space(self):
        with jax.ensure_compile_time_eval():
            try:
                obs_size = self._env.observation_size["state"]
            except TypeError:
                obs_size = self._env.observation_size
            return jym.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=jnp.float32
            )

    @property
    def action_space(self):
        with jax.ensure_compile_time_eval():
            r = np.asarray(self._env.mj_model.actuator_ctrlrange, dtype=np.float32)
            return jym.Box(
                low=r[:, 0],
                high=r[:, 1],
                shape=(self._env.action_size,),
                dtype=jnp.float32,
            )
