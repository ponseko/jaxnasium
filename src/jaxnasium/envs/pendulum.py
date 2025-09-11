from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from jaxnasium._environment import Environment, TimeStep
from jaxnasium._registry import registry
from jaxnasium._spaces import Box

DEFAULT_X = jnp.pi
DEFAULT_Y = 1.0


class EnvState(eqx.Module):
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int = 0


@registry.register("Pendulum-v1")
class Pendulum(Environment[EnvState]):
    """
    Pendulum environment from OpenAI Gym.
    """

    max_speed: float = 8.0
    max_torque: float = 2.0
    dt: float = 0.05
    g: float = 10.0
    m: float = 1.0
    l: float = 1.0  # noqa: E741

    max_episode_steps: int = 200

    def step_env(
        self, key: PRNGKeyArray, state: EnvState, action: int
    ) -> Tuple[TimeStep, EnvState]:
        u = jnp.clip(action, -self.max_torque, self.max_torque)
        costs = (
            self.angle_normalize(state.theta) ** 2
            + 0.1 * state.theta_dot**2
            + 0.001 * (u**2)
        )

        new_theta_dot = (
            state.theta_dot
            + (
                3 * self.g / (2 * self.l) * jnp.sin(state.theta)
                + 3.0 / (self.m * self.l**2) * u
            )
            * self.dt
        )
        new_theta_dot = jnp.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = state.theta + new_theta_dot * self.dt

        state = EnvState(
            theta=new_theta,
            theta_dot=new_theta_dot,
            time=state.time + 1,
        )

        timestep = TimeStep(
            self.get_observation(state),
            self.get_reward(costs),
            self.get_terminated(state),
            self.get_truncated(state),
            {},
        )

        return timestep, state

    def reset_env(self, key: PRNGKeyArray) -> Tuple[Array, EnvState]:
        high = jnp.array([DEFAULT_X, DEFAULT_Y])
        state_variables = jax.random.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = EnvState(
            theta=state_variables[0],
            theta_dot=state_variables[1],
            time=0,
        )
        observation = self.get_observation(state)
        return observation, state

    def get_observation(self, state: EnvState) -> Array:
        return jnp.array(
            [jnp.cos(state.theta), jnp.sin(state.theta), state.theta_dot],
            dtype=jnp.float32,
        )

    def get_reward(self, costs) -> float:
        return -costs.squeeze()

    def get_terminated(self, state: EnvState) -> bool:
        return False  # Only truncates

    def get_truncated(self, state: EnvState) -> bool:
        return state.time >= self.max_episode_steps

    @property
    def observation_space(self) -> Box:
        high = jnp.array([1.0, 1.0, self.max_speed])
        return Box(
            low=-high,
            high=high,
            shape=(3,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> Box:
        return Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=jnp.float32
        )

    @staticmethod
    def angle_normalize(x: jnp.ndarray) -> jnp.ndarray:
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi
