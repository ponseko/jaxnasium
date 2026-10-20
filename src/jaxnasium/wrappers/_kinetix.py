import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxnasium._spaces import Discrete, Space

from ._gymnax import GymnaxWrapper
from ._util import gymnasium_to_jaxnasium_space


class KinetixWrapper(GymnaxWrapper):
    """
    Wrapper for Kinetix environments.
    Kinetix follows the Gymnax API, but with a custom MultiDiscrete space.
    The MultiDiscrete space is converted to a list of `Discrete` spaces (because the space
    is heterogeneous).  The `symbolic_entity` observation type is not supported.

    `make_kinetix_env_from_name` builds the common setups; other `KinetixEnv`s (e.g. with
    your own reset function) can be wrapped directly, preferably with `auto_reset=False`.

    **Arguments:**

    - `_env`: Kinetix environment (`KinetixEnv`).
    - `handle_truncation`: See `GymnaxWrapper`. Default=True.
    - `_params`: Kinetix `EnvParams`. If None, the environment's default params are used.
    """

    def __check_init__(self):
        from kinetix.environment import EntityObservations  # type: ignore

        if isinstance(self._env.observation_type, EntityObservations):
            raise NotImplementedError(
                "Kinetix `symbolic_entity` observations are not supported. "
                "Use `symbolic_flat` or `pixels` instead."
            )

    def step(self, key, state, action):
        if isinstance(action, list):  # convert back
            action = jnp.stack(action, axis=-1)
        return super().step(key, state, action)  # type: ignore

    @property
    def action_space(self) -> Space | list[Space]:  # type: ignore
        with jax.ensure_compile_time_eval():
            space = self._env.action_space(self.params)
        if space.__class__.__name__ == "MultiDiscrete":
            # Kinetix MultiDiscrete spaces may be heterogeneous, which we convert
            # to a list of Discrete spaces.
            nvec = np.asarray(space.number_of_dims_per_distribution)
            return [Discrete(int(n)) for n in nvec]
        return gymnasium_to_jaxnasium_space(space)


def make_kinetix_env_from_name(
    name: str,
    action_type: str = "multi_discrete",
    observation_type: str = "symbolic_flat",
    **env_params_kwargs,
) -> tuple[Any, Any]:
    """Build a Kinetix env and its `EnvParams` from a name:

    - `s`, `m`, `l`: all hand-made levels of that size.
    - `{s,m,l}/<level>`: a single hand-made level, e.g. `l/h4_cartpole`.
    - `random-{s,m,l}`: procedurally generated levels of that size.

    Remaining kwargs are passed to `EnvParams` (e.g. `dense_reward_scale`, `max_timesteps`).
    """
    from kinetix.environment import (  # type: ignore
        ActionType,
        EnvParams,
        ObservationType,
        make_kinetix_env,
        static_env_params_from_size,
    )
    from kinetix.environment.ued.ued import (  # type: ignore
        make_reset_fn_list_of_levels,
        make_reset_fn_sample_kinetix_level,
    )
    from kinetix.util.saving import BASE_DIR  # type: ignore

    size = name.removeprefix("random-").split("/")[0]
    static_env_params: Any = static_env_params_from_size(size)
    env_params = EnvParams(**env_params_kwargs)
    if name.startswith("random-"):
        reset_fn = make_reset_fn_sample_kinetix_level(env_params, static_env_params)
    else:
        levels = (
            [name]
            if "/" in name
            else sorted(f"{size}/{f}" for f in os.listdir(os.path.join(BASE_DIR, size)))
        )
        reset_fn = make_reset_fn_list_of_levels(levels, static_env_params)

    env = make_kinetix_env(
        action_type=ActionType.from_string(action_type),
        observation_type=ObservationType.from_string(observation_type),
        reset_fn=reset_fn,
        env_params=env_params,
        static_env_params=static_env_params,
        auto_reset=False,  # KinetixWrapper does its own auto-reset
    )
    return env, env_params
