import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from functools import partial, reduce
from typing import Any, Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, Real

import jaxnasium as jym
from jaxnasium._environment import (
    ORIGINAL_OBSERVATION_KEY,
    Environment,
    TEnvState,
    TimeStep,
    TObservation,
)
from jaxnasium._spaces import Box, Discrete, MultiDiscrete, Space
from jaxnasium._types import AgentObservation

from ._util import partition_obs_and_masks

logger = logging.getLogger(__name__)


class Wrapper(Environment):
    """Base class for all wrappers."""

    _env: Environment

    def __check_init__(self):
        logger.info(f"Wrapping environment with {self.__class__.__name__}")

    def reset_env(self, key: PRNGKeyArray) -> tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        return self._env.reset_env(key)

    def step_env(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[Real[Array, "..."]]
    ) -> tuple[TimeStep, TEnvState]:
        return self._env.step_env(key, state, action)

    def reset(self, key: PRNGKeyArray) -> tuple[TObservation, Any]:  # pyright: ignore[reportInvalidTypeVarUse]
        return self._env.reset(key)

    def step(
        self, key: PRNGKeyArray, state: Any, action: PyTree[Real[Array, "..."]]
    ) -> tuple[TimeStep, Any]:
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
    """Remove every `wrapper_class` from the environment, keeping the rest of the stack."""

    def rebuild(env: Environment) -> Environment:
        if not isinstance(env, Wrapper):
            return env
        if isinstance(env, wrapper_class):
            return rebuild(env._env)  # drop this one, keep looking further in
        inner = env._env
        replacement = rebuild(inner)
        if replacement is inner:
            return env
        return eqx.tree_at(
            lambda e: e._env, env, replacement, is_leaf=lambda x: x is inner
        )

    return rebuild(wrapped_env)


def unwrap_to(wrapped_env: Environment, wrapper_class: type) -> Environment:
    """The environment inside the outermost `wrapper_class`, dropping everything around it."""
    current_env = wrapped_env
    while isinstance(current_env, Wrapper):
        if isinstance(current_env, wrapper_class):
            return current_env._env
        current_env = current_env._env
    return wrapped_env


def insert_wrapper(
    wrapped_env: Environment,
    new_wrapper: Callable[[Environment], Environment],
    *,
    inner_wrapper: type,
) -> Environment:
    """Insert a wrapper into a stack of wrappers at the position of a given inner wrapper."""

    def rewrap(env: Environment) -> Environment:
        if not isinstance(env, Wrapper):
            return env
        inner = env._env
        replacement = (
            new_wrapper(inner) if isinstance(env, inner_wrapper) else rewrap(inner)
        )
        if replacement is inner:
            return env
        return eqx.tree_at(
            lambda e: e._env, env, replacement, is_leaf=lambda x: x is inner
        )

    return rewrap(wrapped_env)


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

    def reset(self, key: PRNGKeyArray) -> tuple[TObservation, Any]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, state = jax.vmap(self._env.reset)(key)
        return obs, state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[Real[Array, "..."]]
    ) -> tuple[TimeStep, TEnvState]:
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

    def reset(self, key: PRNGKeyArray) -> tuple[TObservation, LogEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)

        # Infer the reward shape from the environment step function:
        reward_template = jax.eval_shape(
            lambda s: self._env.step(key, s, self._env.sample_action(key))[0].reward,
            env_state,
        )

        initial_returns = jym.tree.zeros_like(reward_template)
        initial_lengths = jym.tree.zeros_like(reward_template, dtype=jnp.int32)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=initial_returns,
            episode_lengths=initial_lengths,
            returned_episode_returns=initial_returns,
            returned_episode_lengths=initial_lengths,
            timestep=0,
        )

        return obs, state

    def step(
        self, key: PRNGKeyArray, state: LogEnvState, action: PyTree[int | float | Array]
    ) -> tuple[TimeStep, LogEnvState]:
        timestep, env_state = self._env.step(key, state.env_state, action)

        terminated, truncated = timestep.terminated, timestep.truncated
        assert jax.tree.structure(terminated) == jax.tree.structure(truncated)

        done = jax.tree.map(jnp.logical_or, terminated, truncated)
        done = reduce(jnp.logical_and, jax.tree.leaves(done))

        new_episode_return = jym.tree.add(state.episode_returns, timestep.reward)
        new_episode_length = jym.tree.add(state.episode_lengths, 1)

        state = LogEnvState(
            env_state=env_state,
            episode_returns=jym.tree.mul(new_episode_return, (1 - done)),  # done reset
            episode_lengths=jym.tree.mul(new_episode_length, (1 - done)),  # done reset
            # If done, set new episode return (/length); else, keep old episode return (/length)
            returned_episode_returns=jym.tree.add(
                jym.tree.mul(state.returned_episode_returns, (1 - done)),
                jym.tree.mul(new_episode_return, done),
            ),
            returned_episode_lengths=jym.tree.add(
                jym.tree.mul(state.returned_episode_lengths, (1 - done)),
                jym.tree.mul(new_episode_length, done),
            ),
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

        delta = jax.tree.map(lambda m, b: b - m, state.mean, batch_mean)
        tot_count = state.count + batch_count
        new_mean = jax.tree.map(
            lambda m, d: m + d * batch_count / tot_count,
            state.mean,
            delta,
        )

        m_a = jax.tree.map(lambda v: v * state.count, state.var)
        m_b = jax.tree.map(lambda v: v * batch_count, batch_var)
        M2 = jax.tree.map(
            lambda a, b, d: (
                a + b + jnp.square(d) * state.count * batch_count / tot_count
            ),
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

    def reset(self, key: PRNGKeyArray) -> tuple[TObservation, NormalizeVecObsState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        obs, masks = partition_obs_and_masks(obs, self._env.multi_agent)
        obs_template = jax.tree.map(lambda o: o[0], obs)
        state = NormalizeVecObsState(
            env_state=env_state,
            mean=jax.tree.map(jnp.zeros_like, obs_template),
            var=jax.tree.map(jnp.ones_like, obs_template),
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
    ) -> tuple[TimeStep, NormalizeVecObsState]:
        timestep, env_state = self._env.step(key, state.env_state, action)
        obs = timestep.observation
        obs, masks = partition_obs_and_masks(obs, self._env.multi_agent)
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

    def reset(self, key: PRNGKeyArray) -> tuple[TObservation, NormalizeVecRewState]:  # pyright: ignore[reportInvalidTypeVarUse]
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
    ) -> tuple[TimeStep, NormalizeVecRewState]:
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

    def reset(self, key: PRNGKeyArray) -> tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        obs, masks = partition_obs_and_masks(obs, self._env.multi_agent)
        obs = jax.tree.map(lambda x: jnp.reshape(x, -1), obs)
        obs = eqx.combine(obs, masks)
        return obs, env_state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[int | float | Array]
    ) -> tuple[TimeStep, TEnvState]:
        timestep, env_state = self._env.step(key, state, action)
        obs, masks = partition_obs_and_masks(
            timestep.observation, self._env.multi_agent
        )
        obs = jax.tree.map(lambda x: jnp.reshape(x, -1), obs)
        obs = eqx.combine(obs, masks)
        timestep = timestep._replace(observation=obs)
        try:
            # Same above: flatten the observation itself but leave any action mask alone.
            info = timestep.info
            terminal_obs, terminal_masks = partition_obs_and_masks(
                info[ORIGINAL_OBSERVATION_KEY], self._env.multi_agent
            )
            terminal_obs = jax.tree.map(lambda x: jnp.reshape(x, -1), terminal_obs)
            info[ORIGINAL_OBSERVATION_KEY] = eqx.combine(terminal_obs, terminal_masks)
            timestep = timestep._replace(info=info)
        except Exception:
            pass
        return timestep, env_state

    @property
    def observation_space(self) -> Space:
        obs_space = self._env.observation_space
        obs, masks = partition_obs_and_masks(obs_space, self._env.multi_agent)

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
                _space.low = np.reshape(_space.low, (-1,))
                _space.high = np.reshape(_space.high, (-1,))

            if hasattr(_space, "nvec"):
                _space.nvec = np.reshape(_space.nvec, (-1,))

            return _space

        flat_obs = jax.tree.map(get_flat_shape, obs)
        obs_space = eqx.combine(flat_obs, masks)
        return obs_space


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
    ) -> tuple[TimeStep, TEnvState]:
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
        action: PyTree[int | Int[Array, " num_actions"]],
    ) -> tuple[TimeStep, TEnvState]:
        def convert_to_continuous(space, discrete_action):
            assert hasattr(space, "low") and hasattr(space, "high"), (
                "Original action space must have 'low' and 'high' attributes. Is this a continuous action space?"
            )
            low = jnp.broadcast_to(space.low, space.shape)
            high = jnp.broadcast_to(space.high, space.shape)
            return low + (discrete_action / (self.num_actions - 1)) * (high - low)

        action = jax.tree.map(convert_to_continuous, self.original_action_space, action)
        return self._env.step(key, state, action)

    @property
    def action_space(self) -> PyTree[Discrete | MultiDiscrete]:
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
    def original_action_space(self) -> Space | PyTree[Space]:
        """
        Return the original action space of the environment.
        This is useful for algorithms that need to know the original action space.
        """
        return self._env.action_space


class MetaParamsWrapper(Wrapper):
    def reset(self, key, params: dict):  # pyright: ignore[reportIncompatibleMethodOverride]
        env = self._env
        for k, value in params.items():
            if not hasattr(self._env, k):
                raise ValueError(
                    f"Trying to map over {k}, but environment {k} not found in {self._env}."
                )
            env = eqx.tree_at(lambda env, _k=k: getattr(env, _k), env, value)
        return env.reset(key)

    def step(self, key, state, action, params: dict):  # pyright: ignore[reportIncompatibleMethodOverride]
        env = self._env
        for k, value in params.items():
            if not hasattr(self._env, k):
                raise ValueError(
                    f"Trying to map over {k}, but environment {k} not found in {self._env}."
                )
            env = eqx.tree_at(lambda env, _k=k: getattr(env, _k), env, value)
        return env.step(key, state, action)


@overload
def _actions_per_dimension(space: Space, error: Literal[True]) -> list[int]: ...


@overload
def _actions_per_dimension(space: Space, error: Literal[False]) -> list[int] | None: ...


def _actions_per_dimension(space: Space, error: bool = True) -> list[int] | None:
    """Number of actions along each dimension of a (multi-)discrete space, else `None`.

    `Discrete(n)` has the single dimension `[n]`; `MultiDiscrete(nvec)` has one per entry of `nvec` [n, n, n].

    Raises an error if `error` is `True` and the space is not (multi-)discrete.
    """
    if hasattr(space, "n"):  # Discrete
        return [int(space.n)]  # type: ignore
    if hasattr(space, "nvec"):  # MultiDiscrete
        return [int(n) for n in np.atleast_1d(space.nvec)]  # type: ignore
    if error:
        raise ValueError(f"Space {space} is not (multi-)discrete.")
    return None


def _to_single_discrete_space(spaces):
    """Combines a PyTree of (multi-)discrete spaces to a single discrete space."""

    spaces = jax.tree.leaves(spaces)
    n_values = [int(np.prod(_actions_per_dimension(s, error=True))) for s in spaces]

    combined_num_actions = int(np.prod(np.array(n_values)))
    logger.info(
        f"Flattened action space from: {spaces} to single space of {combined_num_actions} actions."
    )
    return Discrete(combined_num_actions)


def _from_single_discrete_space(target_action_space: PyTree[Space], action: int):
    """Converts a single discrete action to a (multi-)discrete action space."""

    original_actions = []
    spaces, space_structure = jax.tree.flatten(target_action_space)
    for space in spaces:
        actions = []
        for n in _actions_per_dimension(space, error=True):
            actions.append(action % n)
            action = action // n
        # a `Discrete` leaf is a scalar; a `MultiDiscrete` stays as array
        original_actions.append(actions[0] if space.shape == () else jnp.array(actions))

    return jax.tree.unflatten(space_structure, original_actions)


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
    ) -> tuple[TimeStep, TEnvState]:
        # Converts the single discrete action back to the original PyTree of (multi-)discrete actions

        # Skip if action space did not change
        if hasattr(self.original_action_space, "n"):
            return self._env.step(key, state, action)

        if self.multi_agent:
            action = jym.tree.map_one_level(
                lambda sp, a: _from_single_discrete_space(sp, a),
                self.original_action_space,
                action,
            )
            return self._env.step(key, state, action)

        action = _from_single_discrete_space(self.original_action_space, action)
        return self._env.step(key, state, action)

    @property
    def action_space(self) -> Discrete:
        if self.multi_agent:
            return jym.tree.map_one_level(
                _to_single_discrete_space, self._env.action_space
            )

        return _to_single_discrete_space(self._env.action_space)

    @property
    def original_action_space(self) -> Space:
        """Return the original action space of the environment."""
        return self._env.action_space


def _stackable_groups(space_tree: PyTree[Space]) -> list[list[int]]:
    """We group action spaces into groups of homogeneous spaces, such that these
    can be stacked into a single `MultiDiscrete` space and ungrouped into
    the original space later.

    We cannot just flatten and unflatten, because we want to
    flatten -> group homogeneous -> stack in MultiDiscrete.
    So we need to keep track of the original indices of the groupings.
    """
    groups: list[list[int]] = []
    by_num_actions: dict[int, list[int]] = {}
    for index, space in enumerate(jax.tree.leaves(space_tree)):
        dimensions = _actions_per_dimension(space, error=False)
        if dimensions is None or len(set(dimensions)) != 1:  # cont or heterogeneous
            groups.append([index])
            continue
        group = by_num_actions.setdefault(dimensions[0], [])
        if not group:
            groups.append(group)  # first member
        group.append(index)
    return groups


def _stacks_anything(space_tree: PyTree[Space]) -> bool:
    """Checks if there are any groups in the action space that can be stacked.
    Else, we typically just ignore operations."""
    return any(len(group) > 1 for group in _stackable_groups(space_tree))


def _stack_action_space(space_tree: PyTree[Space]) -> PyTree[Space]:
    """Merge each group of `space_tree`'s leaves into a single `MultiDiscrete`."""
    spaces = jax.tree.leaves(space_tree)

    stacked = []
    for group in _stackable_groups(space_tree):
        members = [spaces[index] for index in group]
        if len(group) == 1:
            stacked.append(members[0])
            continue
        # lay every member's dimensions end to end in one `nvec`
        nvec = [n for s in members for n in _actions_per_dimension(s, error=True)]
        stacked.append(MultiDiscrete(nvec=np.array(nvec), dtype=members[0].dtype))

    return stacked[0] if len(stacked) == 1 else tuple(stacked)


def _stack_action_masks(space_tree: PyTree[Space], mask_tree: PyTree) -> PyTree[Array]:
    masks = jax.tree.leaves(mask_tree)

    stacked = []
    for group in _stackable_groups(space_tree):
        members = [masks[index] for index in group]
        if len(group) == 1:
            stacked.append(members[0])
            continue
        rows = [jnp.reshape(m, (-1, m.shape[-1])) for m in members]
        stacked.append(jnp.concatenate(rows, axis=0))

    return stacked[0] if len(stacked) == 1 else tuple(stacked)


def _stack_mask_spaces(space_tree: PyTree[Space], _=None) -> PyTree[Box]:
    """The action-mask *spaces* mirroring the stacked action space."""

    def mask_space(space):
        dimensions = _actions_per_dimension(space, error=False)
        # continuous will now also get a mask; but algorithms should just ignore it
        shape = (*space.shape, dimensions[0]) if dimensions else space.shape
        return Box(
            low=np.zeros(shape, dtype=bool),
            high=np.ones(shape, dtype=bool),
            shape=shape,
            dtype=bool,
        )

    return jax.tree.map(mask_space, _stack_action_space(space_tree))


def _unstack_action(space_tree: PyTree[Space], action) -> PyTree[Array]:
    """Split a stacked action back into the the original environment's action pytree."""
    spaces, structure = jax.tree.flatten(space_tree)
    groups = _stackable_groups(space_tree)
    # a single group was returned bare rather than in a tuple, so re-wrap it
    per_group = list(action) if len(groups) > 1 else [action]

    actions: list[Any] = [None] * len(spaces)
    for group, group_action in zip(groups, per_group):
        if len(group) == 1:
            actions[group[0]] = group_action
            continue
        offset = 0
        for index in group:
            space = spaces[index]
            if space.shape == ():  # Discrete
                chunk = group_action[..., offset]
                offset += 1
            else:  # MultiDiscrete
                size = space.shape[0]
                chunk = group_action[..., offset : offset + size]
                offset += size
            actions[index] = chunk.astype(space.dtype)

    return jax.tree.unflatten(structure, actions)


class StackActionSpaceWrapper(Wrapper):
    """Wrapper to stack homogeneous discrete action spaces into a single `MultiDiscrete`.

    As an example:
    ```python
    (Discrete(20), Discrete(20), Discrete(20))  # -> MultiDiscrete([20, 20, 20])
    (Discrete(10), Discrete(10), Discrete(10), Discrete(5)) -> (MultiDiscrete([10, 10, 10]), Discrete(5))
    ```

    This may be useful because Jaxnasium algorithms will vmap homogenous MultiDiscrete spaces, while
    homogeneous Discrete spaces in a pytree are simply jax.tree.mapped over. Combining them into
    a single MultiDiscrete may then allow for more performance during runtime and compilation.
    Noteably, the environment could also define their action space as a single MultiDiscrete,
    but this wrapper allows for a more flexible definition of the action space, while preserving
    performance benefits.

    Continuous spaces are not stacked, and are passed through as a no-op.
    NOTE: continuous could be stacked as well, it is currently just not implemented.

    **Arguments:**

    - `_env`: Environment to wrap.
    """

    def _per_agent(self, fn: Callable, *trees):
        if self.multi_agent:
            return jym.tree.map_one_level(fn, *trees)
        return fn(*trees)

    def _replace_masks(self, observation, action_space, *, new_mask_fn: Callable):
        """Apply `new_mask(action_space, mask)` to every mask (where present)"""

        if not isinstance(observation, AgentObservation):
            return observation
        if observation.action_mask is None:
            return observation
        if not _stacks_anything(action_space):
            return observation
        return observation.replace(
            action_mask=new_mask_fn(action_space, observation.action_mask)
        )

    def __check_init__(self):
        stacks = self._per_agent(_stacks_anything, self._env.action_space)
        if not any(jax.tree.leaves(stacks)):
            logger.warning(
                f"{self.__class__.__name__} left the action space unchanged "
                f"({self._env.action_space}): no two of its leaves are discrete "
                "over the same number of actions. The wrapper is a no-op."
            )

    def reset(self, key: PRNGKeyArray) -> tuple[TObservation, Any]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        replace_mask_fn = partial(self._replace_masks, new_mask_fn=_stack_action_masks)
        obs = self._per_agent(replace_mask_fn, obs, self._env.action_space)
        return obs, env_state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[Real[Array, "..."]]
    ) -> tuple[TimeStep, TEnvState]:
        def restore(action_space, agent_action):
            if not _stacks_anything(action_space):
                return agent_action  # unchanged space -> unchanged action
            return _unstack_action(action_space, agent_action)

        action = self._per_agent(restore, self.original_action_space, action)
        timestep, env_state = self._env.step(key, state, action)
        replace_mask_fn = partial(self._replace_masks, new_mask_fn=_stack_action_masks)
        observation = self._per_agent(
            replace_mask_fn, timestep.observation, self._env.action_space
        )
        return timestep._replace(observation=observation), env_state

    @property
    def action_space(self) -> Space | PyTree[Space]:
        def stack(action_space):
            if not _stacks_anything(action_space):
                return action_space  # No changes
            return _stack_action_space(action_space)

        return self._per_agent(stack, self._env.action_space)

    @property
    def observation_space(self) -> Space | PyTree[Space]:
        # Have to adjust the mask if present for every agent:
        replace_mask_fn = partial(self._replace_masks, new_mask_fn=_stack_mask_spaces)
        return self._per_agent(
            replace_mask_fn, self._env.observation_space, self._env.action_space
        )

    @property
    def original_action_space(self) -> Space | PyTree[Space]:
        """Return the original action space of the environment."""
        return self._env.action_space
