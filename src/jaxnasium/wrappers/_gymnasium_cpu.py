import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jaxtyping import Array, Int, PRNGKeyArray, PyTree, Real

from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY, Observation, TimeStep
from jaxnasium._spaces import Space

from ._util import gymnasium_to_jaxnasium_space
from ._wrappers import Wrapper

logger = logging.getLogger(__name__)

_SEED_MAX = np.iinfo(np.int32).max


def _autoreset_mode(env: Any) -> Any:
    """The autoreset mode of a Gymnasium vector env, or None if it does not declare one."""
    return getattr(env, "autoreset_mode", env.metadata.get("autoreset_mode"))


class _CallbackEnv:
    """Python side of the environment: converts between a Gymnasium environment and the
    fixed shapes/dtypes that `io_callback` requires.
    """

    def __init__(
        self, env: Any, num_envs: int | None, autoreset_mode: Any = None
    ) -> None:
        self.env = env
        self.num_envs = num_envs  # None when the environment has no batch axis
        self.autoreset_mode = autoreset_mode

        # SAME_STEP autoreset environments already follow the api; else we have to do a manual
        # reset on finished environments.
        self.manual_autoreset = (
            num_envs is not None
            and getattr(autoreset_mode, "name", None) != "SAME_STEP"
        )
        batch_shape = () if num_envs is None else (num_envs,)

        # io_callback needs the shapestructs of whatever is returned, so we construct that here
        # and set it on self
        probe_obs, _info = env.reset(seed=0)
        self.obs_struct = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(np.shape(x), jnp.asarray(x).dtype), probe_obs
        )
        self.reward_struct = jax.ShapeDtypeStruct(batch_shape, jnp.result_type(float))
        self.done_struct = jax.ShapeDtypeStruct(batch_shape, jnp.bool_)

        if self.manual_autoreset and num_envs > 1:  # pyright: ignore[reportOptionalOperand]
            self._check_masked_reset(probe_obs)

    def _check_masked_reset(self, obs: Any) -> None:
        """Manual autoreset relies on `reset(options={"reset_mask": mask})` to reset only finished environments.
        This is not always present though, so we check for that here."""
        before = [np.array(x, copy=True) for x in jax.tree.leaves(obs)]
        mask = np.zeros(self.num_envs, dtype=np.bool_)  # type: ignore[reportOptionalOperand]
        mask[0] = True
        ignored_mask = ValueError(
            f"{type(self.env).__name__} (autoreset_mode={self.autoreset_mode}) cannot reset "
            "individual environments via `reset(options={'reset_mask': mask})`, which "
            "GymnasiumWrapper needs to auto-reset it within the same step. Use a vector "
            "environment with AutoresetMode.SAME_STEP, or one that supports masked resets."
        )
        try:
            reset_obs, _info = self.env.reset(options={"reset_mask": mask})
        except Exception as e:
            raise ignored_mask from e
        for old, new in zip(before, jax.tree.leaves(reset_obs), strict=True):
            new = np.asarray(new)
            # A full batch must keep the unmasked environments untouched.
            if new.shape[0] == self.num_envs and not np.array_equal(new[1:], old[1:]):
                raise ignored_mask

    def _to_jax_obs(self, obs: Any) -> Any:
        # Makes a copy
        return jax.tree.map(
            lambda struct, x: np.array(x, dtype=struct.dtype).reshape(struct.shape),
            self.obs_struct,
            obs,
        )

    def _to_gym_action(self, action: Any) -> Any:
        if self.num_envs is None:
            space = self.env.action_space
            shape = space.shape
        else:
            # The `action_space` of a vector env is not always batched (implementations differ),
            # so we build the batched shape from the single environment's space.
            space = self.env.single_action_space
            shape = (self.num_envs, *space.shape)
        gym_action = np.asarray(action, dtype=space.dtype).reshape(shape)

        # Discrete / MultiDiscrete may not start at 0
        start = getattr(space, "start", None)
        if start is not None:
            gym_action = gym_action + start
        return gym_action

    def _terminal_obs(self, obs: Any, info: dict) -> Any:
        """AutoresetMode.SAME_STEP mirrors the jaxnasium API and puts the terminal obs in info.
        Here we grab it from info and ensure it is returned as a regular property
        because we need to drop the info dict.
        """
        final_obs, mask = info.get("final_obs"), info.get("_final_obs")
        if final_obs is None or mask is None or not mask.any():
            return obs
        terminal = jax.tree.map(lambda x: np.array(x, copy=True), obs)
        for i in np.flatnonzero(mask):
            jax.tree.map(
                lambda dst, src: dst.__setitem__(i, src), terminal, final_obs[i]
            )
        return terminal

    def reset(self, seeds: Any) -> Any:
        if self.num_envs is None:
            obs, _info = self.env.reset(seed=int(seeds))
        else:
            obs, _info = self.env.reset(seed=[int(s) for s in seeds])
        return self._to_jax_obs(obs)

    def _reset_finished_envs(self, obs: Any, done: np.ndarray) -> Any:
        """Emulates SAME_STEP Autoreset for each environment"""

        # reset each env that has finished
        reset_obs, _info = self.env.reset(options={"reset_mask": done})

        def replace_w_new_obs(batch: np.ndarray, maybe_reset_obs: Any) -> np.ndarray:
            fresh = np.asarray(maybe_reset_obs, dtype=batch.dtype)
            # Some implementations return the whole batch, others only the reset envs.
            batch[done] = fresh[done] if fresh.shape[0] == self.num_envs else fresh
            return batch

        return jax.tree.map(replace_w_new_obs, obs, reset_obs)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        obs, reward, terminated, truncated, info = self.env.step(
            self._to_gym_action(action)
        )
        # Convert the step's outputs before any reset below.
        reward = np.array(reward, dtype=self.reward_struct.dtype)
        terminated = np.array(terminated, dtype=np.bool_)
        truncated = np.array(truncated, dtype=np.bool_)

        if self.num_envs is None:
            # A regular env does not reset itself, so we autoreset here.
            terminal_obs = self._to_jax_obs(obs)
            if terminated or truncated:
                obs, _info = self.env.reset()
            obs = self._to_jax_obs(obs)
        elif self.manual_autoreset:
            # Autoreset when AutoresetMode is not set to SAME_STEP
            terminal_obs = self._to_jax_obs(obs)
            obs = self._to_jax_obs(obs)
            done = terminated | truncated
            if done.any():
                obs = self._reset_finished_envs(obs, done)
        else:
            # AutoresetMode is already SAME_STEP, so we do nothing
            terminal_obs = self._to_jax_obs(self._terminal_obs(obs, info))
            obs = self._to_jax_obs(obs)
        return obs, terminal_obs, reward, terminated, truncated


class GymnasiumWrapper(Wrapper):
    """Runs CPU-based [Gymnasium](https://gymnasium.farama.org/) environments inside a
    JAX training loop, by calling back into Python on every step. Every step round-trips
    through Python, so throughput is likely slower compared to
    native JAX environments; this wrapper aims for compatibility, not speed.

    Envpool environments are also possible through this wrapper
    ([more information](https://ponseko.github.io/jaxnasium/gymnasium/#envpool)).

    **Vectorization (required for training jaxnasium algorithms)**
    Jaxnasium algorithms expect a leading batch axis and as such expect a vectorized
    environment. For Gymnasium environments, this needs to happen on the Gymnasium side
    instead of via a jaxnasium `VecEnvWrapper`.
    As such, wrap the gymnasium environment with a `gymnasium.vector.VectorEnv`.
    Note that, in contrast to training, `algorithm.evaluate` expects a
    regular non-vectorized environment.

    Example:

    ```python
    import gymnasium
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    venv = SyncVectorEnv(
        [lambda: gymnasium.make("CartPole-v1")] * 8,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    train_env = GymnasiumWrapper(venv)  # batched, for training
    eval_env = GymnasiumWrapper(
        gymnasium.make("CartPole-v1")
    )  # unbatched, for evaluation
    ```

    **Arguments:**

    - `env`: a Gymnasium environment, or a `gymnasium.vector.VectorEnv` of them, preferably
      built with `autoreset_mode=AutoresetMode.SAME_STEP`.

    !!! warning
        Gymnasium `info` dictionaries are dropped: their keys vary between steps, while a
        callback has to return a fixed PyTree of fixed shapes. Only
        `_TERMINAL_OBSERVATION` is filled in by this wrapper.
    """

    _env: Any
    _callback_env: _CallbackEnv

    def __init__(self, env: Any):
        import gymnasium  # type: ignore

        num_envs, mode = None, None
        if isinstance(env, gymnasium.vector.VectorEnv):
            num_envs, mode = env.num_envs, _autoreset_mode(env)
            if mode is None:
                raise ValueError(
                    f"{type(env).__name__} is vectorized, but we could not determine the autoreset "
                    "mode: no `autoreset_mode` attribute or metadata entry found."
                )
        self._env = env
        self._callback_env = _CallbackEnv(env, num_envs, mode)

    def __check_init__(self):
        if self._callback_env.manual_autoreset:
            logger.info(
                f"{type(self._env).__name__} uses autoreset_mode="
                f"{self._callback_env.autoreset_mode}. For compatibility with the Jaxnasium API, "
                "the GymnasiumWrapper performs the reset, which costs an extra reset call on every "
                "episode end. Use AutoresetMode.SAME_STEP for a small performance boost if possible."
            )

    @property
    def _internal_num_envs(self) -> int | None:
        """Size of the batch axis. This is None for a regular (unbatched) environment."""
        return self._callback_env.num_envs

    @property
    def is_vectorized(self) -> bool:
        """Whether the wrapped environment is a `gymnasium.vector.VectorEnv`, in that case,
        its observations, rewards and flags carry a leading `num_envs` axis."""
        return self._callback_env.num_envs is not None

    def _require_env_keys(self, key: PRNGKeyArray, name: str) -> None:
        assert jax.random.key_data(key).shape[:-1] == (self._internal_num_envs,), (
            f"{type(self._env).__name__} is a vectorized Gymnasium environment "
            f"(num_envs={self._internal_num_envs}), but {name}() was not given one "
            "key per environment"
        )

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, Int[Array, " num_envs"]]:
        if self.is_vectorized:
            self._require_env_keys(key, "reset")
            seeds = jax.vmap(lambda k: jax.random.randint(k, (), 0, _SEED_MAX))(key)
        else:
            seeds = jax.random.randint(key, (), 0, _SEED_MAX)

        obs = io_callback(
            self._callback_env.reset, self._callback_env.obs_struct, seeds, ordered=True
        )

        # no explicit state, but we must return something.
        dummy_state = jnp.zeros(self._callback_env.done_struct.shape, dtype=jnp.int32)
        return obs, dummy_state

    def step(
        self,
        key: PRNGKeyArray,
        state: Int[Array, " num_envs"],
        action: PyTree[Real[Array, "..."]],
    ) -> tuple[TimeStep, Int[Array, " num_envs"]]:
        obs_struct = self._callback_env.obs_struct
        obs, terminal_obs, reward, terminated, truncated = io_callback(
            self._callback_env.step,
            (
                obs_struct,
                obs_struct,
                self._callback_env.reward_struct,
                self._callback_env.done_struct,
                self._callback_env.done_struct,
            ),
            action,
            ordered=True,
        )
        info = {ORIGINAL_OBSERVATION_KEY: terminal_obs}
        # state returned as is, is just a dummy
        return TimeStep(obs, reward, terminated, truncated, info), state

    def sample_action(self, key: PRNGKeyArray) -> PyTree[Real[Array, "..."]]:
        if not self.is_vectorized:
            return super().sample_action(key)
        self._require_env_keys(key, "sample_action")
        return jax.vmap(super().sample_action)(key)

    def sample_observation(self, key: PRNGKeyArray) -> Observation:
        if not self.is_vectorized:
            return super().sample_observation(key)
        self._require_env_keys(key, "sample_observation")
        return jax.vmap(super().sample_observation)(key)

    def reset_env(self, key: PRNGKeyArray) -> tuple[Observation, Any]:
        return self.reset(key)

    def step_env(
        self, key: PRNGKeyArray, state: Any, action: PyTree[Real[Array, "..."]]
    ) -> tuple[TimeStep, Any]:
        raise NotImplementedError(
            "GymnasiumWrapper auto-resets inside its step callback, so no `step_env` is used."
        )

    @property
    def observation_space(self) -> Space | PyTree[Space]:
        space = (
            self._env.single_observation_space
            if self.is_vectorized
            else self._env.observation_space
        )
        return gymnasium_to_jaxnasium_space(space)

    @property
    def action_space(self) -> Space | PyTree[Space]:
        space = (
            self._env.single_action_space
            if self.is_vectorized
            else self._env.action_space
        )
        return gymnasium_to_jaxnasium_space(space)
