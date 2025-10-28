import logging
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, Real

import jaxnasium as jym

from ._environment import (
    ORIGINAL_OBSERVATION_KEY,
    AgentObservation,
    Environment,
    TEnvState,
    TimeStep,
    TObservation,
)
from ._spaces import Discrete, MultiDiscrete, Space

logger = logging.getLogger(__name__)


def is_wrapped(wrapped_env: Environment, wrapper_class: type | str) -> bool:
    """
    Check if the environment is wrapped with a specific wrapper class.
    """
    current_env = wrapped_env
    while isinstance(current_env, Wrapper):
        if isinstance(wrapper_class, str):  # Handle string class names
            if current_env.__class__.__name__ == wrapper_class:
                return True
        else:  # Handle class type inputs
            if isinstance(current_env, wrapper_class):
                return True
        current_env = current_env._env
    return False


def remove_wrapper(wrapped_env: Environment, wrapper_class: type) -> Environment:
    """
    Remove a specific wrapper class from the environment.
    """
    current_env = wrapped_env
    while isinstance(current_env, Wrapper):
        if isinstance(current_env, wrapper_class):
            return current_env._env
        current_env = current_env._env
    return wrapped_env


def _partition_obs_and_masks(
    observation_tree: PyTree[TObservation], multi_agent: bool
) -> Tuple[PyTree, PyTree]:
    """
    Seperates a PyTree of observations of type `AgentObservation` into two trees:
    one with the observations and one with the masks.
    If the observation is not of type `AgentObservation`, the second tree will only
    contain `None` values.
    This is used such that wrappers can act on the observations only when action masks
    are present.

    Useage:
    ```python
    (observation, ...), env_state = self._env.step/reset(...)
    obs, masks = self.partition_obs_and_masks(observation)
    obs = ... # do something with obs ...
    observation = eqx.combine(obs, masks)
    ```

    **Arguments:**

    - `observation_tree`: (PyTree of) observations to be partitioned.
    - `multi_agent`: Whether the environment is multi-agent or not.

    """
    observations = [observation_tree]
    if multi_agent:
        observations, _ = eqx.tree_flatten_one_level(observation_tree)
    if all(not isinstance(o, AgentObservation) for o in observations):
        filter_spec = True
    elif all(isinstance(o, AgentObservation) for o in observations):
        filter_spec = AgentObservation(observation=True, action_mask=False)
        filter_spec = jax.tree.map(
            lambda _: filter_spec,
            observation_tree,
            is_leaf=lambda x: isinstance(x, AgentObservation),
        )
    else:
        raise ValueError(
            "Observations for all agents must be either AgentObservation or not."
        )
    return eqx.partition(observation_tree, filter_spec=filter_spec)


class Wrapper(Environment):
    """Base class for all wrappers."""

    _env: Environment

    def reset_env(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        return self._env.reset_env(key)

    def step_env(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[Real[Array, "..."]]
    ) -> Tuple[TimeStep, TEnvState]:
        return self._env.step_env(key, state, action)

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, Any]:  # pyright: ignore[reportInvalidTypeVarUse]
        return self._env.reset(key)

    def step(
        self, key: PRNGKeyArray, state: Any, action: PyTree[Real[Array, "..."]]
    ) -> Tuple[TimeStep, Any]:
        return self._env.step(key, state, action)

    @property
    def action_space(self) -> Space | PyTree[Space]:
        return self._env.action_space

    @property
    def observation_space(self) -> Space | PyTree[Space]:
        return self._env.observation_space

    @property
    def multi_agent(self) -> bool:
        return getattr(self, "_multi_agent", getattr(self._env, "multi_agent", False))

    def __getattr__(self, name):
        return getattr(self._env, name)


class VecEnvWrapper(Wrapper):
    """
    Wrapper to vectorize environments.
    Simply calls `jax.vmap` on the `reset` and `step` methods of the environment.
    The number of environmnents is determined by the leading axis of the
    inputs to the `reset` and `step` methods, as if you would call `jax.vmap` directly.

    We use a wrapper instead of `jax.vmap` in each algorithm directly to control where
    the vectorization happens. This allows other wrappers to act on the vectorized
    environment, e.g. `NormalizeVecObsWrapper` and `NormalizeVecRewardWrapper`.
    """

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, Any]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, state = jax.vmap(self._env.reset)(key)
        return obs, state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[Real[Array, "..."]]
    ) -> Tuple[TimeStep, TEnvState]:
        timestep, state = jax.vmap(self._env.step)(key, state, action)
        return timestep, state


class LogEnvState(eqx.Module):
    env_state: TEnvState  # pyright: ignore[reportGeneralTypeIssues]
    episode_returns: float | Array
    episode_lengths: int | Array
    returned_episode_returns: float | Array
    returned_episode_lengths: int | Array
    timestep: int | Array = 0


class LogWrapper(Wrapper):
    """
    Log the episode returns and lengths. Modeled after the LogWrapper in
    [PureJaxRL](https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/purejaxrl/wrappers.py#L73).

    This wrapper inserts episode returns and lengths into the `info` dictionary of the
    `TimeStep` object. The `returned_episode_returns` and `returned_episode_lengths`
    are the returns and lengths of the last completed episode.

    After collecting a trajectory of `n` steps and collecting all the info dicts,
    the episode returns may be collected via:
    ```python
    return_values = jax.tree.map(
        lambda x: x[data["returned_episode"]], data["returned_episode_returns"]
    )
    ```

    **Arguments:**

    - `_env`: Environment to wrap.
    """

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, LogEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        structure = self._env.agent_structure
        initial_vals = jnp.zeros(structure.num_leaves).squeeze()
        initial_timestep = 0
        if is_wrapped(self._env, VecEnvWrapper):
            vec_count = jax.tree.leaves(obs)[0].shape[0]
            initial_vals = jnp.zeros((vec_count, structure.num_leaves)).squeeze()
            initial_timestep = jnp.zeros((vec_count,)).squeeze()
        if initial_vals.ndim == 0:
            initial_vals = jnp.expand_dims(initial_vals, axis=0)
        initial_returns = jax.tree.unflatten(structure, initial_vals.T)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=initial_returns,
            episode_lengths=initial_vals,
            returned_episode_returns=initial_returns,
            returned_episode_lengths=initial_vals,
            timestep=initial_timestep,
        )
        return obs, state

    def step(
        self, key: PRNGKeyArray, state: LogEnvState, action: PyTree[int | float | Array]
    ) -> Tuple[TimeStep, LogEnvState]:
        timestep, env_state = self._env.step(key, state.env_state, action)

        terminated, truncated = timestep.terminated, timestep.truncated
        assert jax.tree.structure(terminated) == jax.tree.structure(truncated)
        done = jax.tree.map(jnp.logical_or, terminated, truncated)
        done = jnp.all(jnp.array(jax.tree.leaves(done)))  # jax.tree.all does not work
        new_episode_return = jax.tree.map(
            lambda _r, r: (_r + r), state.episode_returns, timestep.reward
        )
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jax.tree.map(
                lambda n_r: (n_r * (1 - done)).squeeze(), new_episode_return
            ),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=jax.tree.map(
                lambda r, n_r: (r * (1 - done) + n_r * done).squeeze(),
                state.returned_episode_returns,
                new_episode_return,
            ),
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info = timestep.info
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return timestep._replace(info=info), state

    def _flat_reward(self, rewards: float | PyTree[float]):
        return jnp.array(jax.tree.leaves(rewards)).squeeze()


class NormalizeVecObsState(eqx.Module):
    env_state: TEnvState  # pyright: ignore[reportGeneralTypeIssues]
    mean: Float[Array, "..."]
    var: Float[Array, "..."]
    count: float


class NormalizeVecObsWrapper(Wrapper):
    """
    Normalize the observations of the environment via running mean and variance.
    This wrapper acts on vectorized environments and in turn should be wrapped within
    a `VecEnvWrapper`.

    **Arguments:**

    - `_env`: Environment to wrap.
    """

    def __check_init__(self):
        if not is_wrapped(self._env, VecEnvWrapper):
            raise ValueError(
                "NormalizeVecReward wrapper must wrapped around a `VecEnvWrapper`.\n"
                " Please wrap the environment with `VecEnvWrapper` first."
            )

    def update_state_and_get_obs(self, obs, state: NormalizeVecObsState):
        batch_mean = jax.tree.map(lambda o: jnp.mean(o, axis=0), obs)
        batch_var = jax.tree.map(lambda o: jnp.var(o, axis=0), obs)
        batch_count = jax.tree.leaves(obs)[0].shape[0]

        delta = jax.tree.map(lambda m, b: b - m, batch_mean, state.mean)
        tot_count = state.count + batch_count
        new_mean = jax.tree.map(
            lambda m, d: m + d * batch_count / tot_count,
            state.mean,
            delta,
        )

        m_a = jax.tree.map(lambda v: v * state.count, state.var)
        m_b = jax.tree.map(lambda v: v * batch_count, batch_var)
        M2 = jax.tree.map(
            lambda a, b, d: a
            + b
            + jnp.square(d) * state.count * batch_count / tot_count,
            m_a,
            m_b,
            delta,
        )
        new_var = jax.tree.map(lambda m: m / tot_count, M2)
        new_count = tot_count
        new_state = NormalizeVecObsState(
            env_state=state.env_state, mean=new_mean, var=new_var, count=new_count
        )

        normalized_obs = jax.tree.map(
            lambda o, m, v: (o - m) / jnp.sqrt(v + 1e-8), obs, new_mean, new_var
        )
        return normalized_obs, new_state

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, NormalizeVecObsState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        obs, masks = _partition_obs_and_masks(obs, self._env.multi_agent)
        state = NormalizeVecObsState(
            env_state=env_state,
            mean=jax.tree.map(jnp.zeros_like, obs),
            var=jax.tree.map(jnp.ones_like, obs),
            count=1e-4,
        )
        normalized_obs, state = self.update_state_and_get_obs(obs, state)
        normalized_obs = eqx.combine(normalized_obs, masks)
        return normalized_obs, state

    def step(
        self,
        key: PRNGKeyArray,
        state: NormalizeVecObsState,
        action: PyTree[int | float | Array],
    ) -> Tuple[TimeStep, NormalizeVecObsState]:
        timestep, env_state = self._env.step(key, state.env_state, action)
        obs = timestep.observation
        obs, masks = _partition_obs_and_masks(obs, self._env.multi_agent)
        state = replace(state, env_state=env_state)
        normalized_obs, state = self.update_state_and_get_obs(obs, state)
        normalized_obs = eqx.combine(normalized_obs, masks)
        return timestep._replace(observation=normalized_obs), state


class NormalizeVecRewState(eqx.Module):
    env_state: TEnvState  # pyright: ignore[reportGeneralTypeIssues]
    mean: Float[Array, "..."]
    var: Float[Array, "..."]
    count: float
    return_val: Float[Array, "..."]


class NormalizeVecRewardWrapper(Wrapper):
    """
    Normalize the rewards of the environment via running mean and variance.
    This wrapper acts on vectorized environments and in turn should be wrapped within
    a `VecEnvWrapper`.

    **Arguments:**

    - `_env`: Environment to wrap.
    - `gamma`: Discount factor for the rewards.
    """

    gamma: float = 0.99

    def __check_init__(self):
        if not is_wrapped(self._env, VecEnvWrapper):
            raise ValueError(
                "NormalizeVecReward wrapper must wrapped around a `VecEnvWrapper`.\n"
                " Please wrap the environment with `VecEnvWrapper` first."
            )

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, NormalizeVecRewState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        batch_count = jax.tree.leaves(obs)[0].shape[0]
        num_agents = self._env.agent_structure.num_leaves
        state = NormalizeVecRewState(
            env_state=env_state,
            mean=jnp.zeros(num_agents).squeeze(),
            var=jnp.ones(num_agents).squeeze(),
            count=1e-4,
            return_val=jnp.zeros((num_agents, batch_count)).squeeze(),
        )

        return obs, state

    def step(
        self,
        key: PRNGKeyArray,
        state: NormalizeVecRewState,
        action: PyTree[int | float | Array],
    ) -> Tuple[TimeStep, NormalizeVecRewState]:
        (obs, reward, terminated, truncated, info), env_state = self._env.step(
            key, state.env_state, action
        )

        # get the rewards as a single matrix -- reconstruct later
        reward, reward_structure = jax.tree.flatten(reward)
        reward = jnp.array(reward).squeeze()
        done = jnp.logical_or(terminated, truncated)  # TODO ?
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=-1)
        batch_var = jnp.var(return_val, axis=-1)
        batch_count = jax.tree.leaves(obs)[0].shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewState(
            env_state=env_state,
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
        )

        if np.any(self._env.multi_agent):  # type: ignore[reportGeneralTypeIssues]
            reward = reward / jnp.sqrt(jnp.expand_dims(state.var, axis=-1) + 1e-8)
            reward = jax.tree.unflatten(reward_structure, reward)
        else:
            reward = reward / jnp.sqrt(state.var + 1e-8)

        return TimeStep(obs, reward, terminated, truncated, info), state


class FlattenObservationWrapper(Wrapper):
    """Flatten the observations of the environment.

    Flattens each observation in the environment to a single vector.
    When the observation is a PyTree of arrays, it flattens each array
    and returns the same PyTree structure with the flattened arrays.

    **Arguments:**

    - `_env`: Environment to wrap.
    """

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        obs, masks = _partition_obs_and_masks(obs, self._env.multi_agent)
        obs = jax.tree.map(lambda x: jnp.reshape(x, -1), obs)
        obs = eqx.combine(obs, masks)
        return obs, env_state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[int | float | Array]
    ) -> Tuple[TimeStep, TEnvState]:
        timestep, env_state = self._env.step(key, state, action)
        obs, masks = _partition_obs_and_masks(
            timestep.observation, self._env.multi_agent
        )
        obs = jax.tree.map(lambda x: jnp.reshape(x, -1), obs)
        # if not isinstance(obs, jnp.ndarray):
        #     obs = jnp.concatenate(obs)
        obs = eqx.combine(obs, masks)
        timestep = timestep._replace(observation=obs)
        try:
            info = timestep.info
            info[ORIGINAL_OBSERVATION_KEY] = jax.tree.map(
                lambda x: jnp.reshape(x, -1), info[ORIGINAL_OBSERVATION_KEY]
            )
            timestep._replace(
                info=info,
            )
        except Exception:
            pass
        return timestep, env_state

    @property
    def observation_space(self) -> Space:
        obs_space = self._env.observation_space

        def get_flat_shape(space):
            if not hasattr(space, "shape") or space.shape == ():
                return space
            _space = deepcopy(space)
            flat_space_shape = int(np.prod(np.array(space.shape)))
            try:
                _space.shape = (flat_space_shape,)
            except AttributeError:  # Gymnasium envs have no setter on shape
                _space._shape = (flat_space_shape,)

            # Also flatten the .low and .high attributes if they exist
            if hasattr(_space, "low") and hasattr(_space, "high"):
                _space.low = jnp.reshape(_space.low, (-1,))
                _space.high = jnp.reshape(_space.high, (-1,))

            if hasattr(_space, "nvec"):
                _space.nvec = jnp.reshape(_space.nvec, (-1,))

            return _space

        return jax.tree.map(get_flat_shape, obs_space)


class TransformRewardWrapper(Wrapper):
    """
    Transform the rewards of the environment using a given function.

    **Arguments:**

    - `_env`: Environment to wrap.
    - `transform_fn`: Function to transform the rewards.
    """

    transform_fn: Callable

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[int | float | Array]
    ) -> Tuple[TimeStep, TEnvState]:
        timestep, env_state = self._env.step(key, state, action)
        transformed_reward = jax.tree.map(self.transform_fn, timestep.reward)
        return timestep._replace(reward=transformed_reward), env_state


class ScaleRewardWrapper(TransformRewardWrapper):
    """
    Scale the rewards of the environment by a given factor.

    **Arguments:**

    - `_env`: Environment to wrap.
    - `scale`: Factor to scale the rewards by.
    """

    scale: float

    def __init__(self, env: Environment, scale: float = 1.0):
        self._env = env
        self.scale = scale
        self.transform_fn = lambda r: r * scale


class DiscreteActionWrapper(Wrapper):
    """
    Wrapper to convert continuous actions to discrete actions.

    **Arguments:**

    - `_env`: Environment to wrap.
    - `num_actions`: Number of discrete actions to convert to.
    """

    num_actions: int

    def step(
        self,
        key: PRNGKeyArray,
        state: TEnvState,
        action: int | Int[Array, " num_actions"],
    ) -> Tuple[TimeStep, TEnvState]:
        # Convert the (multi)discrete action back to a continuous action
        original_action_space = self.original_action_space
        assert hasattr(original_action_space, "low") and hasattr(
            original_action_space, "high"
        ), (
            "Original action space must have 'low' and 'high' attributes. Is this a continuous action space?"
        )
        action = original_action_space.low + (action / (self.num_actions - 1)) * (  # type: ignore
            original_action_space.high - original_action_space.low  # type: ignore
        )
        return self._env.step(key, state, action)

    @property
    def action_space(self) -> Discrete | MultiDiscrete:
        def convert_to_discrete_or_multi_discrete(space):
            assert hasattr(space, "shape")
            if space.shape == () or space.shape == (1,):
                return Discrete(self.num_actions)
            elif len(space.shape) == 1:
                return MultiDiscrete(nvec=np.array([self.num_actions] * space.shape[0]))
            else:
                raise ValueError(
                    f"Action space of shape {space.shape} is not supported for DiscreteActionWrapper."
                    "Please raise an issue on GitHub if you need this feature."
                )

        return jax.tree.map(
            convert_to_discrete_or_multi_discrete, self._env.action_space
        )

    @property
    def original_action_space(self) -> Space:
        """
        Return the original action space of the environment.
        This is useful for algorithms that need to know the original action space.
        """
        return self._env.action_space


class MetaParamsWrapper(Wrapper):
    def reset(self, key, params: dict):  # pyright: ignore[reportIncompatibleMethodOverride]
        env = self._env
        for k in params:
            if not hasattr(self._env, k):
                raise ValueError(
                    f"Trying to map over {k}, but environment {k} not found in {self._env}."
                )
            env = eqx.tree_at(lambda env: getattr(env, k), self._env, params[k])
        return env.reset(key)

    def step(self, key, state, action, params: dict):  # pyright: ignore[reportIncompatibleMethodOverride]
        env = self._env
        for k in params:
            if not hasattr(self._env, k):
                raise ValueError(
                    f"Trying to map over {k}, but environment {k} not found in {self._env}."
                )
            env = eqx.tree_at(lambda env: getattr(env, k), self._env, params[k])
        return env.step(key, state, action)


class FlattenActionSpaceWrapper(Wrapper):
    """Wrapper to convert (PyTrees of) (multi-)discrete action spaces to a single
    discrete action space. This grows the action space (significantly for large action spaces),
    but allows to use algorithms that only support discrete action spaces.

    First flattens each MultiDiscrete action space to a single discrete action space,
    then combines possibly remaining discrete action spaces to a single discrete action space.

    **Arguments:**

    - `_env`: Environment to wrap.
    """

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: int
    ) -> Tuple[TimeStep, TEnvState]:
        # Converts the single discrete action back to the original PyTree of (multi-)discrete actions

        def from_single_discrete_space(action_space: PyTree[Space], action: int):
            """Converts a single discrete action to a (multi-)discrete action space."""

            original_actions = []
            spaces, space_structure = jax.tree.flatten(action_space)
            for space in spaces:
                if hasattr(space, "n"):
                    original_actions.append(action % space.n)
                    action = action // space.n
                elif hasattr(space, "nvec"):
                    _actions = []
                    for n in space.nvec:
                        _actions.append(action % n)
                        action = action // n
                    original_actions.append(jnp.array(_actions))
                else:
                    raise ValueError(
                        f"Cannot flatten space: {space}. Only (Multi-)Discrete spaces are supported."
                    )

            return jax.tree.unflatten(space_structure, original_actions)

        # Skip if action space did not change
        if hasattr(self.original_action_space, "n"):
            return self._env.step(key, state, action)

        if self.multi_agent:
            action = jym.tree.map_one_level(
                lambda sp, a: from_single_discrete_space(sp, a),
                self.original_action_space,
                action,
            )
            return self._env.step(key, state, action)

        action = from_single_discrete_space(self.original_action_space, action)
        return self._env.step(key, state, action)

    @property
    def action_space(self) -> Discrete:
        def to_single_discrete_space(spaces):
            """Combines a PyTree of (multi-)discrete spaces to a single discrete space."""

            n_values = []
            spaces = jax.tree.leaves(spaces)
            for space in spaces:
                if hasattr(space, "n"):
                    n_values.append(int(space.n))
                elif hasattr(space, "nvec"):
                    n_values.append(int(np.prod(np.array(space.nvec))))
                else:
                    raise ValueError(
                        f"Cannot flatten space: {space}. Only (Multi-)Discrete spaces are supported."
                    )

            combined_num_actions = int(np.prod(np.array(n_values)))
            logger.info(
                f"Flattened action space from: {spaces} to single space of {combined_num_actions} actions."
            )
            return Discrete(combined_num_actions)

        if self.multi_agent:
            return jym.tree.map_one_level(
                to_single_discrete_space, self._env.action_space
            )

        return to_single_discrete_space(self._env.action_space)

    @property
    def original_action_space(self) -> Space:
        """Return the original action space of the environment."""
        return self._env.action_space
