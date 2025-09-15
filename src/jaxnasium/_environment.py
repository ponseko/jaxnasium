from abc import abstractmethod
from typing import Generic, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, PyTreeDef, Real

from ._spaces import Space
from ._types import AgentObservation, TimeStep

ORIGINAL_OBSERVATION_KEY = "_TERMINAL_OBSERVATION"

TObservation = TypeVar("TObservation")
TEnvState = TypeVar("TEnvState")


class Environment(eqx.Module, Generic[TEnvState]):
    """
    Base environment class for JAX-compatible environments. Create your environment by subclassing this.

    `step` and `reset` should typically not be overridden, as they merely handle the
    auto-reset logic. Instead, the environment-specific logic should be implemented in the
    `step_env` and `reset_env` methods.

    """

    def step(
        self,
        key: PRNGKeyArray,
        state: TEnvState,
        action: PyTree[Real[Array, "..."]],
    ) -> Tuple[TimeStep, TEnvState]:
        """
        Steps the environment forward with the given action and performs auto-reset when necessary.
        Additionally, this function inserts the original observation (before auto-resetting) in
        the info dictionary to bootstrap correctly on truncated episodes (`info={"_TERMINAL_OBSERVATION": obs, ...}`)

        This function should typically not be overridden. Instead, the environment-specific logic
        should be implemented in the `step_env` method.

        Returns a TimeStep object (observation, reward, terminated, truncated, info) and the new state.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `state`: Current state of the environment.
        - `action`: Action to take in the environment.
        """

        timestep_step, state_step = self.step_env(key, state, action)
        timestep, state = self.auto_reset(key, timestep_step, state_step)
        return timestep, state

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        """
        Resets the environment to an initial state and returns the initial observation.
        Environment-specific logic is defined in the `reset_env` method. Typically, this function
        should not be overridden.

        Returns the initial observation and the initial state of the environment.

        **Arguments:**

        - `key`: JAX PRNG key.
        """
        obs, state = self.reset_env(key)
        return obs, state

    @abstractmethod
    def step_env(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[Real[Array, "..."]]
    ) -> Tuple[TimeStep, TEnvState]:
        """
        Defines the environment-specific step logic. I.e. here the state of the environment is updated
        according to the transition function.

        Returns a [`TimeStep`](.#timestep) object (observation, reward, terminated, truncated, info) and the new state.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `state`: Current state of the environment.
        - `action`: Action to take in the environment.
        """
        pass

    @abstractmethod
    def reset_env(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        """
        Defines the environment-specific reset logic.

        Returns the initial observation and the initial state of the environment.

        **Arguments:**

        - `key`: JAX PRNG key.
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space | PyTree[Space]:
        """
        Defines the space of valid actions for the environment.
        For multi-agent environments, this should be a PyTree of spaces.
        See [`jaxnasium.spaces`](Spaces.md) for more information on how to define (composite) action spaces.
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space | PyTree[Space]:
        """
        Defines the space of possible observations from the environment.
        For multi-agent environments, this should be a PyTree of spaces.
        See [`jaxnasium.spaces`](Spaces.md) for more information on how to define (composite) observation spaces.
        """
        pass

    def auto_reset(
        self, key: PRNGKeyArray, timestep_step: TimeStep, state_step: TEnvState
    ) -> Tuple[TimeStep, TEnvState]:
        """
        Auto-resets the environment when the episode is terminated or truncated.

        Given a step timestep and state, this function will auto-reset the environment
        and return the new timestep and state when the episode is terminated or truncated.
        Inserts the original observation in info to bootstrap correctly on truncated episodes.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `timestep_step`: The timestep returned by the `step_env` method.
        - `state_step`: The state returned by the `step_env` method.

        **Returns:**
        A tuple of the new timestep and state with the state and observation reset to a new
        initial state and observation when the episode is terminated or truncated.
        The original observation is inserted in info to bootstrap correctly on truncated episodes.
        """

        obs_step, reward, terminated, truncated, info = timestep_step

        assert jax.tree.structure(terminated) == jax.tree.structure(truncated)
        done = jax.tree.map(jnp.logical_or, terminated, truncated)
        done = jnp.all(jnp.array(jax.tree.leaves(done)))  # jax.tree.all does not work
        obs_reset, state_reset = self.reset(key)

        # Replace state and obs based on done
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )
        obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_reset, obs_step)

        # Insert the original observation in info to bootstrap correctly
        try:  # removing possible action mask to lower the memory footprint
            obs_step = jax.tree.map(
                lambda o: o.observation,
                obs_step,
                is_leaf=lambda x: isinstance(x, AgentObservation),
            )
        except Exception:
            pass
        info[ORIGINAL_OBSERVATION_KEY] = obs_step

        return TimeStep(obs, reward, terminated, truncated, info), state

    def sample_action(self, key: PRNGKeyArray) -> PyTree[Real[Array, "..."]]:
        """
        Convenience method to sample a random action from the environment's action space.
        While one could use `self.action_space.sample(key)`, this method additionally works on composite action spaces.
        """
        structure = jax.tree.structure(self.action_space)
        keys = jax.random.split(key, structure.num_leaves)
        keys = jax.tree.unflatten(structure, keys)
        return jax.tree.map(lambda space, k: space.sample(k), self.action_space, keys)

    def sample_observation(self, key: PRNGKeyArray) -> TObservation:  # pyright: ignore[reportInvalidTypeVarUse]
        """
        Convenience method to sample a random observation from the environment's observation space.
        While one could use `self.observation_space.sample(key)`, this method additionally works
        on composite observation spaces.
        """
        structure = jax.tree.structure(self.observation_space)
        keys = jax.random.split(key, structure.num_leaves)
        keys = jax.tree.unflatten(structure, keys)
        return jax.tree.map(
            lambda space, k: space.sample(k), self.observation_space, keys
        )

    @property
    def multi_agent(self) -> bool:
        """Indicates if the environment is a multi-agent environment.

        Infers this via the `_multi_agent` property. If not set, assumes single-agent.
        """
        if hasattr(self, "_multi_agent"):
            return self._multi_agent
        return False

    @property
    def agent_structure(self) -> PyTreeDef:
        """
        Returns the structure of the agent space. In single-agent environments this is
        simply a PyTreeDef(*).
        However, for multi-agent environments this is a PyTreeDef((*, x num_agents)).
        """
        if not self.multi_agent:
            return jax.tree.structure(0)
        _, agent_structure = eqx.tree_flatten_one_level(self.action_space)
        return agent_structure
