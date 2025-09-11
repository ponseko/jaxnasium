from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jaxnasium._environment import Environment, TimeStep
from jaxnasium._registry import registry
from jaxnasium._spaces import Box, Discrete


class EnvState(eqx.Module):
    joint_angle1: Float[Array, ""]
    joint_angle2: Float[Array, ""]
    velocity_1: Float[Array, ""]
    velocity_2: Float[Array, ""]
    time: int


@registry.register("Acrobot-v1")
class Acrobot(Environment[EnvState]):
    """
    Acrobot environment from OpenAI Gym.
    """

    dt: float = 0.2
    link_length_1: float = 1.0
    link_length_2: float = 1.0
    link_mass_1: float = 1.0
    link_mass_2: float = 1.0
    link_com_pos_1: float = 0.5
    link_com_pos_2: float = 0.5
    link_moi: float = 1.0
    max_vel_1: float = 4 * jnp.pi
    max_vel_2: float = 9 * jnp.pi
    torque_noise_max: float = 0.0
    avail_torque: jnp.ndarray = eqx.field(
        default_factory=lambda: jnp.array([-1.0, 0.0, +1.0])
    )

    max_episode_steps: int = 500

    def step_env(
        self, key: PRNGKeyArray, state: EnvState, action: int
    ) -> Tuple[TimeStep, EnvState]:
        torque = self.avail_torque[action]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += jax.random.uniform(
                key, minval=-self.torque_noise_max, maxval=self.torque_noise_max
            )

        # Augment state with force action so it can be passed to ds/dt
        s_augmented = jnp.array(
            [
                state.joint_angle1,
                state.joint_angle2,
                state.velocity_1,
                state.velocity_2,
                torque,
            ]
        )
        ns = self._rk4(s_augmented)
        joint_angle1 = self._wrap(ns[0], -jnp.pi, jnp.pi)
        joint_angle2 = self._wrap(ns[1], -jnp.pi, jnp.pi)
        velocity_1 = jnp.clip(ns[2], -self.max_vel_1, self.max_vel_1)
        velocity_2 = jnp.clip(ns[3], -self.max_vel_2, self.max_vel_2)

        state = EnvState(
            joint_angle1=joint_angle1,
            joint_angle2=joint_angle2,
            velocity_1=velocity_1,
            velocity_2=velocity_2,
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
        state_variables = jax.random.uniform(key, shape=(4,), minval=-0.1, maxval=0.1)
        state = EnvState(
            joint_angle1=state_variables[0],
            joint_angle2=state_variables[1],
            velocity_1=state_variables[2],
            velocity_2=state_variables[3],
            time=0,
        )
        observation = self.get_observation(state)
        return observation, state

    def get_observation(self, state: EnvState) -> Array:
        return jnp.array(
            [
                jnp.cos(state.joint_angle1),
                jnp.sin(state.joint_angle1),
                jnp.cos(state.joint_angle2),
                jnp.sin(state.joint_angle2),
                state.velocity_1,
                state.velocity_2,
            ]
        )

    def get_reward(self, state: EnvState) -> Float[Array, ""]:
        # -1 unless the pole reaches the target height (termination)
        target_height_reached = self.get_terminated(state)
        return -1.0 * (1 - target_height_reached)

    def get_terminated(self, state: EnvState) -> Bool[Array, ""]:
        target_height_reached = (
            -jnp.cos(state.joint_angle1)
            - jnp.cos(state.joint_angle2 + state.joint_angle1)
            > 1.0
        )
        return target_height_reached

    def get_truncated(self, state: EnvState) -> bool:
        return state.time >= self.max_episode_steps

    @property
    def observation_space(self) -> Box:
        high = jnp.array([1.0, 1.0, 1.0, 1.0, self.max_vel_1, self.max_vel_2])
        return Box(
            low=-high,
            high=high,
            shape=(6,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> Discrete:
        return Discrete(len(self.avail_torque))

    # Environment functions
    def _dsdt(self, s_augmented: Array, _: float) -> Array:
        """Compute time derivative of the state change - Use for ODE int."""
        m1, m2 = self.link_mass_1, self.link_mass_2
        l1 = self.link_length_1
        lc1, lc2 = self.link_com_pos_1, self.link_com_pos_2
        i1, i2 = self.link_moi, self.link_moi
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1, theta2, dtheta1, dtheta2 = s
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2))
            + i1
            + i2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + i2
        phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
            + phi2
        )
        ddtheta2 = (
            a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2) - phi2
        ) / (m2 * lc2**2 + i2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])

    def _wrap(self, x: Float[Array, ""], m: float, big_m: float) -> Array:
        """For example, m = -180, M = 180 (degrees), x = 360 --> returns 0."""
        diff = big_m - m
        go_up = x < m  # Wrap if x is outside the left bound
        go_down = x >= big_m  # Wrap if x is outside OR on the right bound

        how_often = go_up * jnp.ceil(
            (m - x) / diff
        ) + go_down * jnp.floor(  # if m - x is an integer, keep it
            (x - big_m) / diff + 1
        )  # if x - M is an integer, round up
        x_out = x - how_often * diff * go_down + how_often * diff * go_up
        return x_out

    def _rk4(self, y0: Array):
        """Runge-Kutta integration of ODE - Difference to OpenAI: Only 1 step!"""
        dt2 = self.dt / 2.0
        k1 = self._dsdt(y0, 0)
        k2 = self._dsdt(y0 + dt2 * k1, dt2)
        k3 = self._dsdt(y0 + dt2 * k2, dt2)
        k4 = self._dsdt(y0 + self.dt * k3, self.dt)
        yout = y0 + self.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout
