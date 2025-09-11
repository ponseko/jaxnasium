from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from jaxnasium._environment import Environment, TimeStep
from jaxnasium._registry import registry
from jaxnasium._spaces import Box, Discrete


class EnvState(eqx.Module):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int = 0


@registry.register("CartPole-v1")
class CartPole(Environment[EnvState]):
    """
    CartPole environment from OpenAI Gym.
    """

    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02
    # kinematics_integrator = "euler"
    theta_threshold_radians: float = 12 * 2 * np.pi / 360
    x_threshold: float = 2.4

    max_episode_steps: int = 500

    @property
    def total_mass(self):
        return self.masscart + self.masspole

    @property
    def polemass_length(self):
        return self.masspole * self.length

    def step_env(
        self, key: PRNGKeyArray, state: EnvState, action: int
    ) -> Tuple[TimeStep, EnvState]:
        force = self.force_mag * action - self.force_mag * (1 - action)
        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (
            force + self.polemass_length * state.theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = state.x + self.tau * state.x_dot
        x_dot = state.x_dot + self.tau * xacc
        theta = state.theta + self.tau * state.theta_dot
        theta_dot = state.theta_dot + self.tau * thetaacc

        state = EnvState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=state.time + 1,
        )

        timestep = TimeStep(
            self.get_observation(state),
            self.get_reward(),
            self.get_terminated(state),
            self.get_truncated(state),
            {},
        )

        return timestep, state

    def reset_env(self, key: PRNGKeyArray) -> Tuple[Array, EnvState]:
        state_variables = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)
        state = EnvState(
            x=state_variables[0],
            x_dot=state_variables[1],
            theta=state_variables[2],
            theta_dot=state_variables[3],
        )
        observation = self.get_observation(state)
        return observation, state

    def get_observation(self, state: EnvState) -> Array:
        return jnp.array(
            [state.x, state.x_dot, state.theta, state.theta_dot], dtype=jnp.float32
        )

    def get_reward(self) -> float:
        return 1.0

    def get_terminated(self, state: EnvState) -> Array:
        return jnp.logical_or(
            jnp.abs(state.x) > self.x_threshold,
            jnp.abs(state.theta) > self.theta_threshold_radians,
        )

    def get_truncated(self, state: EnvState) -> bool:
        return state.time >= self.max_episode_steps

    @property
    def observation_space(self) -> Box:
        high = jnp.array(
            [
                self.x_threshold * 2,
                np.finfo(jnp.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(jnp.float32).max,
            ]
        )
        return Box(
            low=-high,
            high=high,
            shape=(4,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> Discrete:
        return Discrete(2)
