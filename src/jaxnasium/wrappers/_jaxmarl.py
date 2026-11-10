from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import Observation, TEnvState, TimeStep
from jaxnasium._spaces import Box, MultiDiscrete, Space
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
            discrete = self._discrete_agents()
            obs = {
                k: AgentObservation(v, action_masks[k]) if k in discrete else v
                for k, v in obs.items()
            }
        except Exception:
            pass

        return obs, state

    def step(
        self, key: PRNGKeyArray, state: Any, action: float
    ) -> tuple[TimeStep, Any]:
        obs, state_step, reward, done, info = self._env.step(key, state, action)

        # No truncation in JaxMARL (?)
        info["JAXMARL_ORIG_DONE"] = done
        episode_done = done.pop("__all__")
        terminated = jax.tree.map(lambda d: jnp.logical_or(d, episode_done), done)
        if "__all__" in reward:
            reward.pop("__all__")
        truncated = jax.tree.map(lambda x: jnp.full_like(x, False), done)

        if "world_state" in obs and self.remove_world_state:
            obs.pop("world_state")

        try:  # Some environments have an action mask
            action_masks = self._env.get_avail_actions(state_step)
            discrete = self._discrete_agents()
            obs = {
                k: AgentObservation(v, action_masks[k]) if k in discrete else v
                for k, v in obs.items()
            }
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
        obs_spaces = gymnasium_to_jaxnasium_space(obs_spaces)

        # infer and adjust obs from the env.reset as some declared spaces
        # do not align with the actual step returned values
        obs, state = jax.eval_shape(self._env.reset, jax.random.PRNGKey(0))
        obs.pop("world_state", None)  # (not in declared spaces, removed in step/reset)

        def _returned_shape(space, o):
            if isinstance(space, Box) and space.shape != o.shape:
                return Box(-jnp.inf, jnp.inf, o.shape, space.dtype)
            return space

        obs_spaces = jax.tree.map(_returned_shape, obs_spaces, obs)

        try:  # Mirror the action masks that `reset` and `step` add to the observations
            masks = jax.eval_shape(self._env.get_avail_actions, state)
        except Exception:  # no action mask
            return obs_spaces  # type: ignore[reportGeneralTypeIssues]
        discrete = self._discrete_agents()
        return {
            a: AgentObservation(space, Box(0, 1, masks[a].shape, masks[a].dtype))
            if a in discrete
            else space
            for a, space in obs_spaces.items()  # type: ignore[reportAttributeAccessIssue]
        }

    def _discrete_agents(self) -> set:
        """The agents with (multi-)discrete actions. Only these should get masks"""
        return {
            a
            for a, s in self.action_space.items()
            if hasattr(s, "n") or hasattr(s, "nvec")
        }

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
