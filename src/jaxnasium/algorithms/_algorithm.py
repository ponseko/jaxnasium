import copy
import inspect
import logging
import types
import warnings
from abc import abstractmethod
from dataclasses import replace
from typing import Any, Callable, List, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import (
    Environment,
    VecEnvWrapper,
    is_wrapped,
    remove_wrapper,
)
from jaxnasium.algorithms.utils import transform_multi_agent

logger = logging.getLogger(__name__)

DEFAULT_PER_AGENT_FUNCTIONS = [
    "get_action",
    "get_value",
    "_update_agent_state",
    "_make_agent_state",
    "_postprocess_rollout",
]


class RLAlgorithm(eqx.Module):
    """Base class for reinforcement learning algorithms in JAXnasium.

    This abstract base class provides a common interface for implementing RL algorithms
    in JAX. It supports both single-agent and multi-agent scenarios, with automatic
    multi-agent transformation capabilities.

    The philosophy of Jaxnasium algorithms is to maintain close to single-file implementations. All training
    logic is therefor implemented in the algorithms themselves, and are designed for single-agent settings.
    This base class provides only a default evaluation loop, environment compatibility checks, and
    automatic multi-agent transformation capabilities.

    Key Features:
    - Multi-agent support: Automatic transformation of single-agent algorithms to multi-agent
    - Environment compatibility: Built-in checks and warnings for environment compatibility
    - Evaluation: Standardized evaluation interface for comparing algorithm performance

    Attributes:
        agent_state: Abstract variable representing the algorithm's internal state (PyTree of modules)
        multi_agent: Whether this algorithm instance operates in multi-agent mode
        auto_upgrade_multi_agent: Whether to automatically upgrade single-agent methods to multi-agent
        actor_kwargs: Additional keyword arguments for actor networks
        critic_kwargs: Additional keyword arguments for critic networks
        log_function: Logging function to use ("simple", "tqdm", or custom callable)
        log_interval: Interval for logging (as fraction of total steps or absolute number)

    """

    state: eqx.AbstractVar[PyTree[eqx.Module]]
    "Trainable state of the algorithm, usually containing the networks, optimizer state and optional normalization running statistics."

    multi_agent: bool = eqx.field(static=True, default=False)
    auto_upgrade_multi_agent: bool = eqx.field(static=True, default=True)
    actor_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)
    critic_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)
    log_function: Optional[Callable | Literal["simple", "tqdm"]] = eqx.field(
        static=True, default="simple"
    )
    log_interval: int | float = eqx.field(static=True, default=0.05)

    @property
    def is_initialized(self) -> bool:
        return self.state is not None

    def save_state(self, file_path: str):
        with open(file_path, "wb") as f:
            eqx.tree_serialise_leaves(f, self.state)

    def load_state(self, file_path: str) -> "RLAlgorithm":
        with open(file_path, "rb") as f:
            state = eqx.tree_deserialise_leaves(f, self.state)
        agent = replace(self, state=state)
        return agent

    @staticmethod
    @abstractmethod
    def get_action(
        key: PRNGKeyArray, state: Any, observation: PyTree, deterministic: bool
    ) -> Any:
        pass

    @abstractmethod
    def train(self, key: PRNGKeyArray, env: Environment) -> "RLAlgorithm":
        pass

    def evaluate(
        self, key: PRNGKeyArray, env: Environment, num_eval_episodes: int = 10
    ) -> Float[Array, " num_eval_episodes"]:
        assert self.is_initialized, (
            "Agent state is not initialized. Create one via e.g. train() or init_state()."
        )
        if is_wrapped(env, VecEnvWrapper):
            # Cannot vectorize because terminations may occur at different times
            # use jax.vmap(agent.evaluate) if you can ensure episodes are of equal length
            env = remove_wrapper(env, VecEnvWrapper)

        def eval_episode(key, _) -> Tuple[PRNGKeyArray, PyTree[float]]:
            def step_env(carry):
                episode_reward, rng, obs, env_state, done = carry
                rng, action_key, step_key = jax.random.split(rng, 3)

                action = self.get_action(
                    action_key, self.state, obs, deterministic=True
                )
                (obs, reward, terminated, truncated, info), env_state = env.step(
                    step_key, env_state, action
                )
                done = jax.tree.map(jnp.logical_or, terminated, truncated)
                done = jnp.all(jnp.array(jax.tree.leaves(done)))
                episode_reward += jym.tree.mean(reward)
                return (episode_reward, rng, obs, env_state, done)

            key, reset_key = jax.random.split(key)
            obs, env_state = env.reset(reset_key)
            done = False
            episode_reward = 0.0

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

    def __make_multi_agent__(
        self, *, upgrade_func_names: List[str] = DEFAULT_PER_AGENT_FUNCTIONS
    ):
        cls = self.__class__
        new_attrs: dict[str, object] = {}

        for name in upgrade_func_names:
            try:
                attr_obj = inspect.getattr_static(cls, name)
            except AttributeError:
                if (
                    name in DEFAULT_PER_AGENT_FUNCTIONS
                    and upgrade_func_names == DEFAULT_PER_AGENT_FUNCTIONS
                ):  # If algorithm is just using defaults, then we skip missing methods
                    # If upgrade_func_names is set, then we expect methods to be present.
                    continue
                raise AttributeError(f"Method {name!r} not found in {cls.__name__}. ")

            if isinstance(attr_obj, staticmethod):
                orig_fn: Callable = attr_obj.__func__
                new_attrs[name] = staticmethod(transform_multi_agent(orig_fn))

            elif callable(attr_obj) or callable(
                attr_obj.method
            ):  # instance or class method
                orig_fn: Callable = (
                    attr_obj if callable(attr_obj) else attr_obj.method
                )  # .method compatibility with older equinox versions
                new_attrs[name] = transform_multi_agent(orig_fn)

            else:
                raise TypeError(f"Attribute {name!r} is not a (static/class)method")

        NewCls = types.new_class(
            f"{cls.__name__}__MultiAgent", (cls,), {}, lambda ns: ns.update(new_attrs)
        )

        new_instance = copy.copy(self)  # keeps parameters unchanged
        new_instance = replace(new_instance, multi_agent=True)
        object.__setattr__(new_instance, "__class__", NewCls)  # safe: NewCls ⊂ cls
        return new_instance

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
            obs, state = env.reset(jax.random.PRNGKey(0))
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
        if is_wrapped(env, "JaxMARLWrapper"):
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
        if vectorized and not is_wrapped(env, VecEnvWrapper):
            logger.info("Wrapping environment in VecEnvWrapper")
            env = VecEnvWrapper(env)

        return env
