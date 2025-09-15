from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

import jaxnasium as jym


class EnvState(eqx.Module):
    x: int
    y: int
    time: int = 0

    @property
    def location(self):
        return (self.x, self.y)


class ExampleEnv(jym.Environment):
    max_episode_steps: int = 100

    def step_env(
        self, key: PRNGKeyArray, state: EnvState, action: int
    ) -> Tuple[jym.TimeStep, EnvState]:
        """
        Update the environment state based on the action taken.
        """
        # action 0 -> move up
        # action 1 -> move right
        # action 2 -> move down
        # action 3 -> move left
        new_x = state.x + (action == 0) - (action == 2)
        new_y = state.y + (action == 1) - (action == 3)

        state = EnvState(x=new_x, y=new_y, time=state.time + 1)

        timestep = jym.TimeStep(
            observation=self.get_observation(state),
            reward=self.get_reward(),
            terminated=self.get_terminated(state),
            truncated=state.time >= self.max_episode_steps,
            info={},
        )
        return timestep, state

    def reset_env(self, key: PRNGKeyArray) -> Tuple[Array, EnvState]:
        """
        Reset the environment to its initial state.
        """
        state = EnvState(x=5, y=5)  # Start in the center
        observation = self.get_observation(state)
        return observation, state

    def get_observation(self, state: EnvState) -> Array:
        """
        Get the observation from the environment state.
        """
        return jnp.array(state.location)

    def get_reward(self) -> float:
        """
        Get the reward from the environment state.
        """
        # Example reward function: 1 for each step taken
        return 1.0

    def get_terminated(self, state: EnvState) -> Float[Array, "1"]:
        """
        Check if the episode has terminated.
        """
        # Example termination condition: if the agent moves out of bounds
        out_of_bounds_x = jnp.logical_or(state.x < 0, state.x >= 10)
        out_of_bounds_y = jnp.logical_or(state.y < 0, state.y >= 10)
        return jnp.logical_or(out_of_bounds_x, out_of_bounds_y)

    @property
    def observation_space(self) -> jym.Space:
        """
        Define the observation space of the environment.
        """
        return jym.Box(low=0, high=10, shape=(2,))

    @property
    def action_space(self) -> jym.Space:
        """
        Define the action space of the environment.
        """
        return jym.Discrete(n=4)
