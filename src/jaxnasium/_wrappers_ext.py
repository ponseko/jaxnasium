from typing import Any, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

from ._environment import (
    ORIGINAL_OBSERVATION_KEY,
    AgentObservation,
    TEnvState,
    TimeStep,
    TObservation,
)
from ._spaces import Box, Discrete, MultiDiscrete, Space
from ._wrappers import Wrapper


def gymnasium_to_jaxnasium_space(space: Any) -> Space | PyTree[Space]:
    """Also works for Gymnax spaces"""

    def convert_single_space(space: Any) -> Space:
        space_class_name = space.__class__.__name__
        if space_class_name == "Discrete":
            return Discrete(space.n)
        elif space_class_name == "Box":
            return Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=space.dtype,
            )
        elif space_class_name == "MultiDiscrete":
            return MultiDiscrete(
                nvec=space.nvec,
                dtype=space.dtype,
            )
        else:
            raise NotImplementedError(
                f"Conversion for space type {space_class_name} is not implemented."
            )

    # Convert pytrees of spaces
    return jax.tree.map(convert_single_space, space)


class GymnaxWrapper(Wrapper):
    """
    Wrapper for Gymnax environments to transform them into the Jymkit environment interface.

    **Arguments:**

    - `_env`: Gymnax environment.
    - `handle_truncation`: If True, the wrapper will reimplement the autoreset behavior to include
        truncated information and the terminal_observation in the info dictionary. If False, the wrapper will mirror
        the Gymnax behavior by ignoring truncations. Default=True.
    """

    _env: Any
    handle_truncation: bool = True

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        params = getattr(self._env, "default_params", None)
        obs, env_state = self._env.reset(key, params)
        return obs, env_state

    def step(
        self, key: PRNGKeyArray, state: Any, action: int | float
    ) -> Tuple[TimeStep, Any]:
        _params = getattr(self._env, "default_params")  # is dataclass
        original_max_steps = getattr(_params, "max_steps_in_episode", None)

        if not self.handle_truncation or original_max_steps is None:
            obs, state_step, reward, done, info = self._env.step_env(
                key, state, action, _params
            )
            terminated, truncated = done, False

            timestep_step = TimeStep(
                observation=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            timestep, state = self.auto_reset(key, timestep_step, state_step)
            return timestep, state

        # Handle truncation:
        # We increase max_steps_in_episode by 1 so that the done flag from Gymnax
        # only triggers when the episode terminates without truncation.
        # Then we set truncate manually in this wrapper based on the original max_steps_in_episode.
        altered_params = eqx.tree_at(
            lambda x: x.max_steps_in_episode, _params, replace_fn=lambda x: x + 1
        )
        obs_step, state_step, reward, done, info = self._env.step_env(
            key, state, action, altered_params
        )
        terminated = done  # did not truncate due to the +1 in max_steps_in_episode
        truncated = state_step.time >= original_max_steps

        timestep_step = TimeStep(
            observation=obs_step,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        timestep, state = self.auto_reset(key, timestep_step, state_step)
        return timestep, state

    @property
    def observation_space(self) -> Space:
        params = self._env.default_params
        return self._env.observation_space(params)

    @property
    def action_space(self) -> Space:
        params = self._env.default_params
        return self._env.action_space(params)


class JumanjiWrapper(Wrapper):
    """
    Wrapper for Jumanji environments to transform them into the Jymkit environment interface.

    **Arguments:**

    - `_env`: Jumanji environment.
    """

    _env: Any

    def __init__(self, env: Any):
        from jumanji.wrappers import AutoResetWrapper

        self._env = AutoResetWrapper(env, next_obs_in_extras=True)

    def _convert_jumanji_obs(self, obs: Any) -> TObservation:  # pyright: ignore[reportInvalidTypeVarUse]
        if isinstance(obs, tuple) and hasattr(obs, "_asdict"):  # NamedTuple
            # Convert it to a dict and collect the action mask
            action_mask = getattr(obs, "action_mask", None)
            obs = {
                key: value
                for key, value in obs._asdict().items()  # pyright: ignore[reportAttributeAccessIssue]
                if key != "action_mask"
            }
            obs = AgentObservation(observation=obs, action_mask=action_mask)
        return obs  # type: ignore[reportGeneralTypeIssues]

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        state, timestep = self._env.reset(key)
        observation = self._convert_jumanji_obs(timestep.observation)
        return observation, state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: int | float
    ) -> Tuple[TimeStep, TEnvState]:
        state, timestep = self._env.step(state, action)  # No key for Jumanji
        obs = self._convert_jumanji_obs(timestep.observation)

        truncated = jnp.logical_and(timestep.discount != 0, timestep.step_type == 2)
        terminated = jnp.logical_and(timestep.step_type == 2, ~truncated)

        info = timestep.extras
        info["DISCOUNT"] = timestep.discount
        next_obs = info.pop("next_obs", None)
        info[ORIGINAL_OBSERVATION_KEY] = self._convert_jumanji_obs(next_obs)

        timestep = TimeStep(
            observation=obs,
            reward=timestep.reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return timestep, state

    def __convert_gymnasium_space_to_dict(self, space: Any) -> Any:
        """Recursively convert Gymnasium Dict spaces to regular dicts."""
        from gymnasium.spaces import Dict as GymnasiumDict

        if isinstance(space, GymnasiumDict):
            # Recursively convert nested spaces and exclude action_mask
            return {
                k: self.__convert_gymnasium_space_to_dict(v)
                for k, v in space.spaces.items()
                if k != "action_mask"
            }
        return space

    @property
    def observation_space(self) -> Any:
        from jumanji.specs import jumanji_specs_to_gym_spaces

        space = self._env.observation_spec
        space = jumanji_specs_to_gym_spaces(space)
        space = self.__convert_gymnasium_space_to_dict(space)
        return gymnasium_to_jaxnasium_space(space)
        return self.__convert_gymnasium_space_to_dict(space)

    @property
    def action_space(self) -> Any:
        from jumanji.specs import jumanji_specs_to_gym_spaces

        space = self._env.action_spec
        space = jumanji_specs_to_gym_spaces(space)
        space = self.__convert_gymnasium_space_to_dict(space)
        return gymnasium_to_jaxnasium_space(space)
        return self.__convert_gymnasium_space_to_dict(space)


class BraxWrapperState(eqx.Module):
    brax_env_state: Any  # The state of the Brax environment
    timestep: int = 0


class BraxWrapper(Wrapper):
    """
    Wrapper for Brax environments to transform them into the Jymkit environment interface.

    Note: Brax environments would typically be wrapped with a VmapWrapper, EpisodeWrapper and AutoResetWrapper
    VmapWrapper is not included here, as it is replaced by the Jymkit's `VecEnvWrapper`.
    The effects of EpisodeWrapper (truncation) and AutoResetWrapper are merged into this wrapper.

    **Arguments:**

    - `_env`: Brax environment.
    """

    _env: Any
    max_episode_steps: int = 1000  # Brax defaults to 1000

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, BraxWrapperState]:  # pyright: ignore[reportInvalidTypeVarUse]
        env_state = self._env.reset(key)
        env_state = BraxWrapperState(brax_env_state=env_state, timestep=0)
        return env_state.brax_env_state.obs, env_state

    def step(
        self, key: PRNGKeyArray, state: BraxWrapperState, action: int | float
    ) -> Tuple[TimeStep, BraxWrapperState]:
        brax_env_state = self._env.step(state.brax_env_state, action)
        state_step = BraxWrapperState(
            brax_env_state=brax_env_state,
            timestep=state.timestep + 1,
        )
        truncated = state_step.timestep >= self.max_episode_steps
        terminated = brax_env_state.done
        info = brax_env_state.info

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
        from brax.envs.wrappers import gym as braxGym

        obs_space = braxGym.GymWrapper(self._env).observation_space
        return gymnasium_to_jaxnasium_space(obs_space)

    @property
    def action_space(self) -> Any:
        from brax.envs.wrappers import gym as braxGym

        action_space = braxGym.GymWrapper(self._env).action_space
        return gymnasium_to_jaxnasium_space(action_space)


class PgxWrapper(Wrapper):
    """
    Wrapper for Pgx environments to transform them into the Jymkit environment interface.

    **Arguments:**

    - `_env`: Pgx environment.
    - `self_play`: Use a single model for all players in multi-agent environments.
    """

    _env: Any
    self_play: bool = eqx.field(static=True, default=False)

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        state = self._env.init(key)
        observation = state.observation
        action_mask = state.legal_action_mask
        obs = AgentObservation(observation, action_mask)
        if self.multi_agent:
            obs = (obs,) * self._env.num_players
        return obs, state  # pyright: ignore

    def step(
        self, key: PRNGKeyArray, state: Any, action: Tuple[int | float] | int | float
    ) -> Tuple[TimeStep, Any]:
        current_player_index = state.current_player
        active_player_action = jnp.array(action)
        try:  # If trainer returns actions for each player: only excecute the active player action
            if len(active_player_action) == self._env.num_players:
                active_player_action = jnp.array(action)[current_player_index]
        except Exception:
            pass

        pgx_state = self._env.step(state, active_player_action, key)

        observation = pgx_state.observation
        action_mask = pgx_state.legal_action_mask
        reward = pgx_state.rewards.squeeze()
        obs = AgentObservation(observation, action_mask)
        if self.multi_agent:
            obs = (obs,) * self._env.num_players
            reward = tuple(reward)
        timestep_step = TimeStep(
            observation=obs,
            reward=reward,
            terminated=pgx_state.terminated,
            truncated=pgx_state.truncated,
            info={"current_player": pgx_state.current_player},
        )
        timestep, state = self.auto_reset(key, timestep_step, pgx_state)
        return timestep, state

    @property
    def observation_space(self) -> Box | List[Box]:
        num_players = self._env.num_players
        shape = self._env.observation_shape
        obs_space = Box(
            low=jnp.full(shape, -10),
            high=jnp.full(shape, 10),
            shape=shape,
            dtype=jnp.int32,
        )
        if self.multi_agent:
            return (obs_space,) * num_players
        return obs_space

    @property
    def action_space(self) -> Discrete | List[Discrete]:
        num_players = self._env.num_players
        num_actions = self._env.num_actions
        action_space = Discrete(num_actions)
        if self.multi_agent:
            return (action_space,) * num_players
        return action_space

    @property
    def _multi_agent(self) -> bool:
        if self.self_play:
            return False
        return self._env.num_players > 1


class NavixWrapper(Wrapper):
    """
    Wrapper for Navix environments to transform them into the Jymkit environment interface.

    **Arguments:**

    - `_env`: Navix environment.
    """

    _env: Any

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        timestep_navix = self._env.reset(key)
        return timestep_navix.observation, timestep_navix

    def step(self, key: PRNGKeyArray, state: Any, action: int) -> Tuple[TimeStep, Any]:
        timestep_navix = self._env._step(state, action)
        obs = timestep_navix.observation
        reward = timestep_navix.reward
        terminated = timestep_navix.step_type == 2
        truncated = timestep_navix.step_type == 1
        info = timestep_navix.info
        timestep_step = TimeStep(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        timestep, state = self.auto_reset(key, timestep_step, timestep_navix)
        return timestep, state

    @property
    def observation_space(self) -> Box:
        return Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=self._env.observation_space.shape,
            dtype=self._env.observation_space.dtype,
        )

    @property
    def action_space(self) -> Discrete:
        num_actions = self._env.action_space.maximum
        # Add the "done" no-op action which is outside of the Navix action space (?)
        num_actions += 1
        return Discrete(num_actions)


class xMinigridWrapper(Wrapper):
    """
    Wrapper for xMinigrid environments to transform them into the Jymkit environment interface.

    **Arguments:**

    - `_env`: xMinigrid environment.
    - `_params`: xMinigrid environment parameters.
    """

    _env: Any
    _params: Any

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        timestep_xminigrid = self._env.reset(self._params, key)
        return timestep_xminigrid.observation, timestep_xminigrid

    def step(self, key: PRNGKeyArray, state: Any, action: int) -> Tuple[TimeStep, Any]:
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
            low=jnp.full(obs_shape, -10),
            high=jnp.full(obs_shape, 10),
            shape=obs_shape,
            dtype=jnp.int32,
        )

    @property
    def action_space(self) -> Discrete:
        return Discrete(self._env.num_actions(self._params))


class JaxMARLWrapper(Wrapper):
    """
    Wrapper for JaxMARL environments to transform them into the Jymkit environment interface.

    **Arguments:**

    - `_env`: JaxMARL environment.
    """

    _env: Any
    _multi_agent: bool = True
    remove_world_state: bool = True
    """ Removes the world_state that is present in some environments from the observation. Required in Jaxnasium algorithms """

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, state = self._env.reset(key)
        if "world_state" in obs and self.remove_world_state:
            obs.pop("world_state")

        try:  # Some environments have an action mask
            action_masks = self._env.get_avail_actions(state)
            obs = {k: AgentObservation(v, action_masks[k]) for k, v in obs.items()}
        except Exception:
            pass

        return obs, state  # pyright: ignore

    def step(
        self, key: PRNGKeyArray, state: Any, action: int | float
    ) -> Tuple[TimeStep, Any]:
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
    def observation_space(self) -> Dict[str, Space]:
        # Extract the observation space for each agent and return it as a dictionary
        agents = self._env.agents
        try:
            obs_spaces = {str(a): self._env.observation_space(a) for a in agents}
        except TypeError:
            # space does not accept an agent argument
            # in those cases, JaxMARL uses the same space for all agents
            obs_space = self._env.observation_space()
            obs_spaces = {str(a): obs_space for a in agents}
        return gymnasium_to_jaxnasium_space(obs_spaces)  # type: ignore[reportGeneralTypeIssues]

    @property
    def action_space(self) -> Dict[str, Space]:
        # Extract the action space for each agent and return it as a dictionary
        agents = self._env.agents
        try:
            spaces = {str(a): self._env.action_space(a) for a in agents}
        except TypeError:
            # space does not accept an agent argument
            # in those cases, JaxMARL uses the same space for all agents
            action_space = self._env.action_space()
            spaces = {str(a): action_space for a in agents}
        return gymnasium_to_jaxnasium_space(spaces)  # type: ignore[reportGeneralTypeIssues]
