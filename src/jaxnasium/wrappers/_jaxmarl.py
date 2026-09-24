from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import Observation, TEnvState, TimeStep
from jaxnasium._spaces import MultiDiscrete, Space
from jaxnasium._types import AgentObservation

from ._util import gymnasium_to_jaxnasium_space
from ._wrappers import Wrapper


def _convert_jaxmarl_multidiscrete(space):
    # Somewhat confusingly, jaxmarl opts for `num_categories` instead of `nvec` for MultiDiscrete spaces.
    # Other then that, jaxmarl adheres to gymnasium spaces.
    space_class_name = space.__class__.__name__
    if space_class_name == "MultiDiscrete":
        return MultiDiscrete(nvec=space.num_categories, dtype=space.dtype)
    return space


class JaxMARLWrapper(Wrapper):
    """
    Wrapper for JaxMARL environments to transform them into the Jaxnasium environment interface.

    **Arguments:**

    - `_env`: JaxMARL environment.
    """

    _env: Any
    _multi_agent: bool = True
    remove_world_state: bool = True
    """ Removes the world_state that is present in some environments from the observation. Required in Jaxnasium algorithms """

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        obs, state = self._env.reset(key)
        if "world_state" in obs and self.remove_world_state:
            obs.pop("world_state")

        try:  # Some environments have an action mask
            action_masks = self._env.get_avail_actions(state)
            obs = {k: AgentObservation(v, action_masks[k]) for k, v in obs.items()}
        except Exception:
            pass

        return obs, state

    def step(
        self, key: PRNGKeyArray, state: Any, action: float
    ) -> tuple[TimeStep, Any]:
        obs, state_step, reward, done, info = self._env.step(key, state, action)

        terminated = done  # No truncation in JaxMARL (?)
        # remove the __all__ key from the done dict
        info["JAXMARL_ORIG_DONE"] = done
        done.pop("__all__")
        if "__all__" in reward:
            reward.pop("__all__")
        truncated = jax.tree.map(lambda x: jnp.full_like(x, False), done)

        if "world_state" in obs and self.remove_world_state:
            obs.pop("world_state")

        try:  # Some environments have an action mask
            action_masks = self._env.get_avail_actions(state)
            obs = {k: AgentObservation(v, action_masks[k]) for k, v in obs.items()}
        except Exception:
            pass

        timestep_step = TimeStep(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        timestep, state = self.auto_reset(key, timestep_step, state_step)
        return timestep, state

    @property
    def observation_space(self) -> dict[str, Space]:
        # Extract the observation space for each agent and return it as a dictionary
        agents = self._env.agents
        # Ensure that the space properties are not tracers:
        with jax.ensure_compile_time_eval():
            try:
                obs_spaces = {a: self._env.observation_space(a) for a in agents}
            except TypeError:
                # space does not accept an agent argument
                # in those cases, JaxMARL uses the same space for all agents
                obs_space = self._env.observation_space()
                obs_spaces = {a: obs_space for a in agents}

        # Only MultiDiscrete is special, rest is gymnasium spaces.
        obs_spaces = jax.tree.map(_convert_jaxmarl_multidiscrete, obs_spaces)

        return gymnasium_to_jaxnasium_space(obs_spaces)  # type: ignore[reportGeneralTypeIssues]

    @property
    def action_space(self) -> dict[str, Space]:
        # Extract the action space for each agent and return it as a dictionary
        agents = self._env.agents
        with jax.ensure_compile_time_eval():
            try:
                spaces = {a: self._env.action_space(a) for a in agents}
            except TypeError:
                # space does not accept an agent argument
                # in those cases, JaxMARL uses the same space for all agents
                action_space = self._env.action_space()
                spaces = {a: action_space for a in agents}

        # Only MultiDiscrete is special, rest is gymnasium spaces.
        spaces = jax.tree.map(_convert_jaxmarl_multidiscrete, spaces)

        return gymnasium_to_jaxnasium_space(spaces)  # type: ignore[reportGeneralTypeIssues]
