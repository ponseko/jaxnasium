from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jaxtyping import Array, PRNGKeyArray, PyTree, Real, UInt8

from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY, Observation, TimeStep
from jaxnasium._spaces import Space

from ._gymnasium_cpu import GymnasiumWrapper


class EnvPoolXLAWrapper(GymnasiumWrapper):
    """Runs an [EnvPool](https://github.com/sail-sg/envpool) environment through EnvPool's XLA
    interface. Compared to `GymnasiumWrapper(envpool_env)`, regular steps run EnvPool's C++
    environments directly from the compiled program, without calling back into Python. Python is
    only called on steps where an episode ended somewhere in the batch, to reset those environments.

    ```python
    import envpool

    venv = envpool.make_gymnasium("LunarLander-v2", num_envs=8, seed=0)
    env = EnvPoolXLAWrapper(venv)
    ```

    **Arguments:**

    - `env`: a bare EnvPool environment from `envpool.make_gymnasium(..., num_envs=n)`. Gymnasium
        wrappers around it would be bypassed by the XLA step, and are therefore rejected.
    - `autoreset_mode`: `"same_step"` (default) or `"next_step"`, see above.

    !!! note "Autoreset behaviour"
        EnvPool itself resets a finished environment on the *next* step, which ignores its action
        and returns the first observation with a reward of 0. Jaxnasium by default expect same step
        reset behaviour, so we default to that here; however, since we have to do that manually,
        we step into Python and essentially sacrafice performance in exchange for keeping the trainsitions
        clean of observations on which we shouldn't train.

    !!! warning
        - Float64 observations or actions (e.g. EnvPool's MuJoCo) require JAX's 64-bit mode.
        - As is the case for `GymnasiumWrapper`: training only (evaluate with a non-vectorized environment),
          and seeds are set when creating the EnvPool environment.
    """

    _handle: bytes = eqx.field(static=True)
    _xla_step: Callable = eqx.field(static=True)
    _action_dtype: Any = eqx.field(static=True)
    autoreset_mode: Literal["same_step", "next_step"] = eqx.field(static=True)

    def __init__(
        self,
        env: Any,
        autoreset_mode: Literal["same_step", "next_step"] = "same_step",
    ):
        import gymnasium  # type: ignore

        if autoreset_mode not in ("same_step", "next_step"):
            raise ValueError(
                f"autoreset_mode must be 'same_step' or 'next_step', got {autoreset_mode!r}."
            )
        self.autoreset_mode = autoreset_mode

        if isinstance(env, gymnasium.vector.VectorWrapper) or not hasattr(env, "xla"):
            raise ValueError(
                "EnvPoolXLAWrapper needs a bare EnvPool environment: Gymnasium wrappers are "
                "bypassed by the XLA step."
            )

        # init so that we have the obs/action space
        super().__init__(env)
        spaces = jax.tree.leaves(
            (self.observation_space, self.action_space),
            is_leaf=lambda x: isinstance(x, Space),
        )
        if not jax.config.jax_enable_x64 and any(
            np.dtype(s.dtype) == np.float64 for s in spaces
        ):
            raise ValueError(
                "This EnvPool environment uses float64, which its XLA "
                "interface only supports with JAX's 64-bit mode enabled "
                "(`jax.config.update('jax_enable_x64', True)`). Or use GymnasiumWrapper instead."
            )
        try:
            handle, _recv, _send, step = env.xla()
        except Exception as e:
            raise RuntimeError("EnvPool's XLA interface does not load.") from e
        self._handle = np.asarray(handle).tobytes()
        self._xla_step = step
        self._action_dtype = jax.dtypes.canonicalize_dtype(
            env.single_action_space.dtype
        )

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, UInt8[Array, " handle"]]:
        obs, _ = super().reset(key)  # <-- Does a Python Callback
        # env_state is not required, but we use it to store the handle
        # handle is passed along to ensure that steps are dependendent on each other
        # and as a result should properly occur in order.
        # The barrier here makes it available only after the reset above, so the first XLA step runs after it.
        handle = jnp.asarray(np.frombuffer(self._handle, dtype=np.uint8))
        return jax.lax.optimization_barrier((obs, handle))

    def _reset_finished_envs(
        self, obs: Any, done: np.ndarray, handle: np.ndarray
    ) -> Any:
        """Python side of the masked reset: `_CallbackEnv._reset_finished_envs`, passing the
        handle through so that the next XLA step runs after this reset."""
        obs = jax.tree.map(np.array, obs)  # create writable copy
        return self._callback_env._reset_finished_envs(obs, done), handle

    def step(
        self,
        key: PRNGKeyArray,
        state: UInt8[Array, " handle"],
        action: PyTree[Real[Array, "..."]],
    ) -> tuple[TimeStep, UInt8[Array, " handle"]]:
        cb = self._callback_env
        action = jnp.asarray(action).astype(self._action_dtype)
        handle, (obs, reward, terminated, truncated, _info) = self._xla_step(
            state, action
        )
        obs = jax.tree.map(
            lambda s, x: x.astype(s.dtype).reshape(s.shape), cb.obs_struct, obs
        )

        # If some env episode ended, call reset (EnvPool would do so on the next step).
        if self.autoreset_mode == "next_step":
            # No reset here: EnvPool resets these environments on the next step instead.
            reset_obs = obs
        else:
            # else we do a manual reset in a python callback whenever any env finished
            # The handle is passed through the reset, so the next XLA step runs after it.
            done = terminated | truncated
            handle_struct = jax.ShapeDtypeStruct(handle.shape, handle.dtype)
            reset_obs, handle = jax.lax.cond(
                done.any(),
                lambda o, h: io_callback(
                    self._reset_finished_envs,
                    (cb.obs_struct, handle_struct),
                    o,
                    done,
                    h,
                    ordered=True,
                ),
                lambda o, h: (o, h),
                obs,
                handle,
            )
        timestep = TimeStep(
            observation=reset_obs,
            reward=reward.astype(cb.reward_struct.dtype),
            terminated=terminated,
            truncated=truncated,
            info={ORIGINAL_OBSERVATION_KEY: obs},
        )
        return timestep, handle
