from typing import Any, Literal

import jax
import jax.numpy as jnp
import jaxtari.wrappers  # type: ignore
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import (
    ORIGINAL_OBSERVATION_KEY,
    Observation,
    TEnvState,
    TimeStep,
)
from jaxnasium._spaces import Box, Discrete, Space
from jaxnasium.wrappers._wrappers import Wrapper


class JaxAtariWrapper(Wrapper):
    """
    Wrapper for Jaxtari environments, using Jaxtari's own `AtariWrapper` followed by
    either its `PixelObsWrapper` or `ObjectCentricWrapper`.

    Observations are 4 stacked frames: `(4, 84, 84)` grayscale pixels or a flat
    `(4 * num_features,)` vector of object features.

    **Arguments:**

    - `_env`: Jaxtari environment.
    - `observation_style`: `"pixels"` or `"objects"`.
    """

    _env: Any
    observation_style: Literal["pixels", "objects"]

    def __init__(self, env: Any, observation_style: Literal["pixels", "objects"]):
        self.observation_style = observation_style

        env = jaxtari.wrappers.AtariWrapper(env)
        if observation_style == "objects":
            env = jaxtari.wrappers.ObjectCentricWrapper(env)
        elif observation_style == "pixels":
            env = jaxtari.wrappers.PixelObsWrapper(
                env, do_pixel_resize=True, grayscale=True
            )
        else:
            raise ValueError(f"Invalid observation style: {observation_style}")
        self._env = env

    def _process_obs(self, obs):
        # in objects observation: (stack, features) -> flat vector;
        # in pixels observation: (stack, H, W, 1) -> (stack, H, W)
        if self.observation_style == "objects":
            return obs.reshape(-1)
        return obs[..., 0]

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            key = jax.random.key_data(key)
        obs, env_state = self._env.reset(key)
        return self._process_obs(obs), env_state

    def step(self, key: PRNGKeyArray, state: Any, action: int) -> tuple[TimeStep, Any]:
        obs, state, reward, terminated, truncated, info = self._env.step(state, action)
        obs = self._process_obs(obs)

        # Jaxtari does its own auto-reset and, while passing a truncation; does not
        # pass a terminal observation on reset.
        info[ORIGINAL_OBSERVATION_KEY] = obs

        # Some games emit unsigned rewards or int terminated
        timestep = TimeStep(
            observation=obs,
            reward=jnp.asarray(reward, jnp.float32),
            terminated=jnp.asarray(terminated, bool),
            truncated=jnp.asarray(truncated, bool),
            info=info,
        )
        return timestep, state

    @property
    def observation_space(self) -> Space:
        obs = jax.eval_shape(lambda: self.reset(jax.random.PRNGKey(0))[0])
        with jax.ensure_compile_time_eval():  # jaxtari builds bounds with jnp
            space = self._env.observation_space()
            return Box(
                low=jnp.reshape(space.low, obs.shape),
                high=jnp.reshape(space.high, obs.shape),
                shape=obs.shape,
                dtype=obs.dtype,
            )

    @property
    def action_space(self) -> Space:
        return Discrete(self._env.action_space().n)
