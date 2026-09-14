from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import Observation, TEnvState, TimeStep
from jaxnasium._spaces import Box, Discrete
from jaxnasium._types import AgentObservation

from ._wrappers import Wrapper


class PgxWrapper(Wrapper):
    """
    Wrapper for Pgx environments to transform them into the Jaxnasium environment interface.

    **Arguments:**

    - `_env`: Pgx environment.
    - `self_play`: Use a single model for all players in multi-agent environments.
    """

    _env: Any
    self_play: bool = eqx.field(static=True, default=False)

    def reset(self, key: PRNGKeyArray) -> tuple[Observation, TEnvState]:
        state = self._env.init(key)
        observation = state.observation
        action_mask = state.legal_action_mask
        obs = AgentObservation(observation, action_mask)
        if self.multi_agent:
            obs = (obs,) * self._env.num_players
        return obs, state

    def step(
        self, key: PRNGKeyArray, state: Any, action: tuple[int | float] | float
    ) -> tuple[TimeStep, Any]:
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
    def observation_space(self) -> Box | list[Box]:
        num_players = self._env.num_players
        shape = self._env.observation_shape
        obs_space = Box(
            low=np.full(shape, -10),
            high=np.full(shape, 10),
            shape=shape,
            dtype=jnp.int32,
        )
        if self.multi_agent:
            return (obs_space,) * num_players
        return obs_space

    @property
    def action_space(self) -> Discrete | list[Discrete]:
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
