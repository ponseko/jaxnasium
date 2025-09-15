from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, PRNGKeyArray

from jaxnasium._environment import Environment, TimeStep
from jaxnasium._registry import registry
from jaxnasium._spaces import Box, Discrete


class EnvState(eqx.Module):
    position: jnp.ndarray
    velocity: jnp.ndarray
    time: int = 0


@registry.register("MountainCar-v0")
class MountainCar(Environment[EnvState]):
    """
    Mountaincar environment from OpenAI Gym.
    """

    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.5
    goal_velocity: float = 0.0

    force: float = 0.001
    gravity: float = 0.0025

    max_episode_steps: int = 200

    @property
    def low(self):
        return jnp.array([self.min_position, -self.max_speed])

    @property
    def high(self):
        return jnp.array([self.max_position, self.max_speed])

    def step_env(
        self, key: PRNGKeyArray, state: EnvState, action: int
    ) -> Tuple[TimeStep, EnvState]:
        position = state.position
        velocity = state.velocity

        velocity += (action - 1) * self.force + jnp.cos(3 * position) * (-self.gravity)
        velocity = jnp.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = jnp.clip(position, self.min_position, self.max_position)
        velocity = velocity * (1 - (position == self.min_position) * (velocity < 0))

        state = EnvState(
            position=position,
            velocity=velocity,
            time=state.time + 1,
        )

        timestep = TimeStep(
            self.get_observation(state),
            self.get_reward(state),
            self.get_terminated(state),
            self.get_truncated(state),
            {},
        )

        return timestep, state

    def reset_env(self, key: PRNGKeyArray) -> Tuple[Array, EnvState]:
        init_position = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = EnvState(
            position=init_position,
            velocity=jnp.array(0.0),
            time=0,
        )
        observation = self.get_observation(state)
        return observation, state

    def get_observation(self, state: EnvState) -> Array:
        return jnp.array(
            [state.position, state.velocity],
            dtype=jnp.float32,
        )

    def get_reward(self, state: EnvState) -> float:
        return -1.0

    def get_terminated(self, state: EnvState) -> Bool[Array, ""]:
        return jnp.logical_and(
            state.position >= self.goal_position,
            state.velocity >= self.goal_velocity,
        )

    def get_truncated(self, state: EnvState) -> bool:
        return state.time >= self.max_episode_steps

    @property
    def observation_space(self) -> Box:
        return Box(
            low=self.low,
            high=self.high,
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> Discrete:
        return Discrete(3)
