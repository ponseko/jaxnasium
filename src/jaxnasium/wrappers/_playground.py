import jax
import jax.numpy as jnp
import numpy as np

import jaxnasium as jym

from ._brax import BraxWrapper


class PlaygroundWrapper(BraxWrapper):
    # TODO: add option for adding privileged state (i.e. to critic_obs)
    max_episode_steps: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.max_episode_steps is None:  # playground uses config.episode_length
            self.max_episode_steps = int(self._env._config.episode_length)  # type: ignore

    @property
    def action_space(self):
        with jax.ensure_compile_time_eval():
            r = np.asarray(self._env.mj_model.actuator_ctrlrange, dtype=np.float32)
            action_size = self._env.action_size
            if isinstance(action_size, int):
                action_size = (action_size,)
            return jym.Box(
                low=r[:, 0], high=r[:, 1], shape=action_size, dtype=jnp.float32
            )
