from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import (
    Environment,
    LogWrapper,
    Space,
    VecEnvWrapper,
    insert_wrapper,
    is_wrapped,
    unwrap_to,
)

logger = logging.getLogger(__name__)


class RLAlgorithm(eqx.Module):
    """Base class for reinforcement learning algorithms in JAXnasium.

    This object can also be refered to as the "trainer" of an `RLAgent`. It contains the
    hyperparameters and collective methods such as `train` and `evaluate`. The trainable state
    lives in the `RLAgent`, which `init_agent` and `train` return.

    The philosophy of Jaxnasium algorithms is to maintain close to single-file implementations. All training
    logic is therefor implemented in the files themselves, and are designed for single-agent settings.
    This base class provides only a default evaluation loop, environment compatibility checks, and
    automatic multi-agent transformation capabilities.

    Key Features:
    - Multi-agent support: Automatic transformation of single-agent algorithms to multi-agent
    - Environment compatibility: Built-in checks and warnings for environment compatibility
    - Evaluation: Standardized evaluation interface for comparing algorithm performance

    Attributes:
        multi_agent: Whether this algorithm instance operates in multi-agent mode
        auto_upgrade_multi_agent: Whether to automatically upgrade single-agent methods to multi-agent
        log_function: Logging function to use ("simple", "tqdm", or custom callable)
        log_interval: Interval for logging (as fraction of total steps or absolute number)
        reduce_metrics_fn: Reduces one training iteration's metrics to what `train`
            returns alongside the agent. This function is called *after* log function on the
            batch of iteration metrics. Defaults to `"mean"`, averaging the metrics across
            the iteration. Set to `None` to return the full batch of metrics.
    """

    multi_agent: bool = eqx.field(static=True, default=False, kw_only=True)
    auto_upgrade_multi_agent: bool = eqx.field(static=True, default=True, kw_only=True)
    log_function: Callable | Literal["simple", "tqdm"] | None = eqx.field(
        static=True, default="simple", kw_only=True
    )
    log_interval: int | float = eqx.field(static=True, default=0.05, kw_only=True)
    reduce_metrics_fn: Callable | Literal["mean"] | None = eqx.field(
        static=True, default="mean", kw_only=True
    )

    @abstractmethod
    def train(
        self, key: PRNGKeyArray, env: Environment, agent: Any = None
    ) -> tuple[RLAgent, PyTree[Float[Array, " iterations"]]]:
        """Runs the training loop, returning the trained agent and its training metrics.

        `agent=None` builds a fresh agent; passing an existing agent continues training it.
        """
        ...

    @abstractmethod
    def init_agent(self, key: PRNGKeyArray, env: Environment) -> RLAgent:
        """Builds and returns an agent for `env`."""
        ...

    def evaluate(
        self,
        key: PRNGKeyArray,
        agent: RLAgent,
        env: Environment,
        num_eval_episodes: int = 10,
    ) -> Float[Array, " num_eval_episodes"]:
        """`num_eval_episodes` over the environment with the provided agent.

        Can also be called through an agent via `agent.evaluate(key, env, ...)`.
        """
        if is_wrapped(env, VecEnvWrapper):
            # Cannot vectorize because terminations may occur at different times
            # use jax.vmap(agent.evaluate) if you can ensure episodes are of equal length
            env = unwrap_to(env, VecEnvWrapper)
        elif is_wrapped(env, "GymnasiumWrapper") and (
            getattr(env, "_internal_num_envs", None) is not None
        ):
            raise ValueError(
                "Cannot evaluate on a vectorized GymnasiumWrapper. Please recreate the "
                "environment without the `VectorEnv` wrapper for evaluation."
            )

        def eval_episode(key, _) -> tuple[PRNGKeyArray, PyTree[float]]:
            def step_env(carry):
                episode_reward, rng, obs, env_state, done = carry
                rng, action_key, step_key = jax.random.split(rng, 3)

                action = agent.get_action(action_key, obs, deterministic=True)
                (obs, reward, terminated, truncated, _info), env_state = env.step(
                    step_key, env_state, action
                )
                done = jax.tree.map(jnp.logical_or, terminated, truncated)
                done = jnp.all(jnp.array(jax.tree.leaves(done)))
                episode_reward = jym.tree.add(episode_reward, reward)
                return (episode_reward, rng, obs, env_state, done)

            key, reset_key = jax.random.split(key)
            obs, env_state = env.reset(reset_key)
            done = False

            reward_structure = jax.eval_shape(
                lambda s, a: env.step(reset_key, s, a)[0].reward,
                env_state,
                env.sample_action(reset_key),
            )
            episode_reward = jym.tree.zeros_like(reward_structure, dtype=float)

            episode_reward, key, obs, env_state, done = jax.lax.while_loop(
                lambda carry: jnp.logical_not(carry[-1]),
                step_env,
                (episode_reward, key, obs, env_state, done),
            )

            return key, episode_reward

        _, episode_rewards = jax.lax.scan(
            eval_episode, key, jnp.arange(num_eval_episodes)
        )

        return episode_rewards

    def __check_env__(self, env: Environment, vectorized: bool = False) -> Environment:
        """
        Some validation checks on the current environment and its compatibility with the current
        algorithm setup.
        Additionally wraps the environment in a `VecEnvWrapper` if it is not already wrapped
        and `vectorized` is True.
        """
        if env.multi_agent:

            def first_level_structure(tree):
                return jax.tree.structure(tree, is_leaf=lambda x: x is not tree)

            first_level_action_space = first_level_structure(env.action_space)
            first_level_observation_space = first_level_structure(env.observation_space)
            assert first_level_action_space == first_level_observation_space, (
                "Action space and observation space must have the same first-level structure for multi-agent environments."
            )
            dummy_action = env.sample_action(jax.random.PRNGKey(0))
            _obs, state = env.reset(jax.random.PRNGKey(0))
            timestep, _ = env.step(jax.random.PRNGKey(0), state, dummy_action)
            first_level_obs = first_level_structure(timestep.observation)
            first_level_action = first_level_structure(dummy_action)
            first_level_reward = first_level_structure(timestep.reward)
            assert first_level_obs == first_level_reward == first_level_action, (
                "Observation, reward, terminated, and truncated must have the same first-level structure for multi-agent environments."
            )
        if is_wrapped(env, "JumanjiWrapper"):
            logger.warning(
                "Some Jumanji environments rely on specific action masking logic "
                "that may not be compatible with this algorithm. "
                "If this is the case, training will crash during compilation."
            )
        if is_wrapped(env, "JaxMARLWrapper"):  # noqa: SIM102
            if getattr(env, "name", None) == "coin_game":
                logger.warning(
                    "Coin game is currently not supported due to an inconsistent API"
                )
        if is_wrapped(env, "NormalizeVecObsWrapper") and getattr(
            self, "normalize_obs", False
        ):
            warnings.warn(
                "Using both environment-side normalization (NormalizeVecObsWrapper) and algorithm-side normalization."
                "This likely leads to incorrect results. We recommend only using algorithm-side normalization, "
                "as it allows for easier checkpointing and resuming training."
            )
        if is_wrapped(env, "NormalizeVecRewardWrapper") and getattr(
            self, "normalize_reward", False
        ):
            warnings.warn(
                "Using both environment-side normalization (NormalizeVecRewardWrapper) and algorithm-side normalization."
                "This likely leads to incorrect results. We recommend only using algorithm-side normalization, "
                "as it allows for easier checkpointing and resuming training."
            )
        if not is_wrapped(env, LogWrapper) and (
            self.reduce_metrics_fn == "mean" or self.log_function == "simple"
        ):
            logger.warning(
                f"Adding a `LogWrapper` to the environment, as {type(self).__name__} has "
                "set 'reduce_metrics_fn=mean' or 'log_function=simple', which require "
                "episode statistics from a `LogWrapper`."
            )
            if is_wrapped(env, VecEnvWrapper):
                env = insert_wrapper(env, LogWrapper, inner_wrapper=VecEnvWrapper)
            else:
                env = LogWrapper(env)

        if vectorized and not is_wrapped(env, VecEnvWrapper):
            if is_wrapped(env, "GymnasiumWrapper"):
                num_envs = getattr(self, "num_envs", None)
                env_num_envs = getattr(env, "_internal_num_envs", None)
                if env_num_envs is None:
                    raise ValueError(
                        f"Gymnasium envs have to be vectorized on the Gymnasium side for training."
                        f" Wrap it using a `gymnasium.vector.VectorEnv` of {num_envs} environments."
                    )
                if env_num_envs != num_envs:
                    raise ValueError(
                        f"{type(self).__name__} is configured with num_envs={num_envs}, "
                        f"but the environment provides a batch of {env_num_envs}. "
                    )
            else:
                env = VecEnvWrapper(env)

        return env

    @staticmethod
    def load(file_path: str) -> RLAgent:
        """Convenience method for `RLAgent.load(file_path)`.

        Returns the algorithm stored in `file_path` which was saved with `save`
        on an `RLAgent`, trainer included. Requires `jaxon` to be installed, which is not installed by default.
        """
        return RLAgent.load(file_path)


def per_agent(method):
    """Decorator to mark an RLAgent method as per-agent; only applicable in multi-agent settings.
    This is purely documentation, as it is the default behavior.
    """
    method.__per_agent__ = True
    return method


def collective(method):
    """Decorator to mark an RLAgent method as collective; only applicable in multi-agent settings.
    This means that, in multi-agent environments, this method will be called on the entire pytree
    of agents, rather than mapped to each agent individually.

    In MA, this means that `self` of this method refers to the `MultiAgentWrapper` of all agents.
    This is in contrast to `@per_agent`, where the method is written as a single-agent method
    and `self` refers to a single `RLAgent` object.
    """
    method.__per_agent__ = False
    return method


class HackuinoxModule(type(eqx.Module)):
    # Temporary name.
    # Here, we override the regular __call__ of equinox modules to allow __new__ methods of
    # modules to return something other than an instance of the module class (e.g. a MultiAgentWrapper of multiple instances).
    # Normally, Equinox will bookkeep what classes should be created and are being created (presumably to keep track of what should be frozen).

    def __call__(cls, *args, **kwargs):
        # Subclasses with a __new_wrapped__ classmethod can use it to intercept the regular __call__
        # and return something other than an instance of the class (e.g. a MultiAgentWrapper of multiple instances).
        # The __new_wrapped__ method should make sure infinite recursion is avoided
        dispatch = getattr(cls, "__new_wrapped__", None)
        if dispatch is not None:
            dispatched_result = dispatch(*args, **kwargs)
            if dispatched_result:
                return dispatched_result
        return super().__call__(*args, **kwargs)


class RLAgent(eqx.Module, metaclass=HackuinoxModule):
    """Trainable state, along with the trainer that produced it (hyperparameters and training logic)."""

    trainer: eqx.AbstractVar[Any]
    "The hyperparameters this agent was built with, and which its updates read."

    @abstractmethod
    def __init__(self, key: PRNGKeyArray, env: Environment, trainer: RLAlgorithm):
        pass

    def replace(self, **updates) -> Self:
        keys, values = zip(*updates.items())
        return eqx.tree_at(lambda c: [c.__dict__[key] for key in keys], self, values)

    def with_hyperparams(self, **hyperparams) -> Self:
        """Overrides this agent's trainer hyperparameters."""
        return self.replace(trainer=replace(self.trainer, **hyperparams))

    @abstractmethod
    def get_action(self, *args, **kwargs) -> Any: ...

    @collective
    def train(
        self, key: PRNGKeyArray, env: Environment, **hyperparams
    ) -> tuple[Self, PyTree[Float[Array, " iterations"]]]:
        """Continues training this agent, returning it with its training metrics."""
        self = self.with_hyperparams(**hyperparams)
        return self.trainer.train(key, env, agent=self)

    @collective
    def evaluate(
        self, key: PRNGKeyArray, env: Environment, num_eval_episodes: int = 10
    ) -> Float[Array, " num_eval_episodes"]:
        return self.trainer.evaluate(key, self, env, num_eval_episodes)

    @collective
    def save(self, file_path: str):
        """Save the current state (along with the trainer) to `file_path`. This uses `jaxon`
        internally to save the agent. `jaxon` is not installed by default, so must be installed manually.

        Alternatively, serialization can be done like any other eqx.Module as described here:
        https://docs.kidger.site/equinox/examples/serialisation/
        """
        from jaxnasium.algorithms.core import save_agent

        save_agent(file_path, self)

    @classmethod
    @collective
    def load(cls, file_path: str) -> Self:
        """Returns the agent stored in `file_path` which was saved with `save`, trainer included.
        Requires `jaxon` to be installed, which is not installed by default.
        """
        from jaxnasium.algorithms.core import load_agent

        return load_agent(file_path)

    @classmethod
    def __new_wrapped__(cls, key: PRNGKeyArray, env: Environment, trainer: RLAlgorithm):
        @dataclass
        class SingleAgentEnvView:
            env: Environment
            action_space: Space
            observation_space: Space

            def __getattr__(self, name):
                return getattr(self.env, name)

        auto_upgrade_multi_agent = getattr(trainer, "auto_upgrade_multi_agent", False)
        if getattr(env, "multi_agent", False) and auto_upgrade_multi_agent:
            from jaxnasium.algorithms.core._multi_agent import (
                MultiAgentWrapper,
                map_multi_agent,
                to_per_agent,
            )

            # `map_multi_agent` infers the agent structure from the first non-key argument
            # As such, we create a per-agent environment (with each environment having the obs/action space of a single agent)
            obs_spaces, agent_structure = eqx.tree_flatten_one_level(
                env.observation_space
            )
            action_spaces = eqx.tree_flatten_one_level(env.action_space)[0]
            envs = [
                SingleAgentEnvView(env, a, o) for a, o in zip(action_spaces, obs_spaces)
            ]
            envs = jax.tree.unflatten(agent_structure, envs)

            # Also create a per-agent trainer that may have per agent hyperparemeters.
            # I.e. with `gamma={"agent_0": 0.9, ...}`, each agent's trainer will have its own `gamma` value.
            trainer_args = {}
            for k, value in trainer.__dict__.items():
                if k == "auto_upgrade_multi_agent":
                    continue  # prevent infinite recursion
                trainer_args[k] = to_per_agent(value, agent_structure)
            trainers = [
                type(trainer)(
                    auto_upgrade_multi_agent=False,  # prevent infinite recursion
                    **{
                        key: eqx.tree_flatten_one_level(trainer_args[key])[0][i]
                        for key in trainer_args
                    },
                )
                for i in range(agent_structure.num_leaves)
            ]
            trainers = jax.tree.unflatten(agent_structure, trainers)

            return MultiAgentWrapper(
                # don't vmap during construction
                map_multi_agent(
                    lambda k, e, t: cls(k, e, t), key, envs, trainers, vmap=False
                ),
                trainer=trainer,  # the wrapper keeps the full trainer
            )
        return None  # continue with regular __call__
