from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

import jaxnasium as jym
from jaxnasium import (
    AgentObservation,
    Box,
    Discrete,
    Environment,
    EnvState,
    MultiDiscrete,
    TimeStep,
)
from jaxnasium.algorithms.core import Transition

SEED = jax.random.PRNGKey(0)
DEFAULT_NUM_ACTIONS = 4  # Default discrete num actions


# --------------------------------------------------------------------------- #
# Proxy environment
# Essentially, we build a minimal enviroment with configurable observation and action spaces.
# Usefull for testing all kind of input and output spaces.
# --------------------------------------------------------------------------- #
class AnyInputOutputEnvState(EnvState):
    current_step: int
    state_remainder: Any


def _mean_of_leaves(tree: Any) -> Array:
    """Mean over every element of every leaf of a pytree (returns a scalar)."""
    leaves = jax.tree.leaves(tree)
    flat = [jnp.ravel(jnp.asarray(leaf)).astype(jnp.float32) for leaf in leaves]
    return jnp.mean(jnp.concatenate(flat))


class AnyInputOutputEnv(Environment):
    """A minimal environment defined purely by an observation and action space.

    The dynamics are trivial (the observation is constant).
    """

    observation_space_fn: Callable[[], Any] = eqx.field(static=True)
    action_space_fn: Callable[[], Any] = eqx.field(static=True)
    _multi_agent: bool = eqx.field(static=True, default=False)
    episode_length: int = eqx.field(static=True, default=100)

    @property
    def observation_space(self) -> Any:  # type: ignore[override]
        return self.observation_space_fn()

    @property
    def action_space(self) -> Any:  # type: ignore[override]
        return self.action_space_fn()

    def reset_env(self, key: PRNGKeyArray):
        obs = self.sample_observation(key)
        state = AnyInputOutputEnvState(current_step=0, state_remainder=obs)
        return obs, state

    def step_env(self, key: PRNGKeyArray, state: AnyInputOutputEnvState, action: Any):
        # Reward is derived from the action so that continuous/discrete actions of
        # any structure produce a (per-agent) scalar reward.
        if self._multi_agent:
            reward = jym.tree.map_one_level(_mean_of_leaves, action)
        else:
            reward = _mean_of_leaves(action)

        obs = self.sample_observation(key)
        new_state = AnyInputOutputEnvState(
            current_step=state.current_step + 1,
            state_remainder=obs,
        )
        truncated = new_state.current_step >= self.episode_length
        terminated = jnp.zeros((), dtype=bool)
        info = {"received_action": action}
        return TimeStep(obs, reward, terminated, truncated, info), new_state


# --------------------------------------------------------------------------- #
# Space helpers
# --------------------------------------------------------------------------- #
def _bool_box(shape: tuple[int, ...]) -> Box:
    return Box(
        low=np.zeros(shape, dtype=bool),
        high=np.ones(shape, dtype=bool),
        shape=shape,
        dtype=jnp.bool_,
    )


def _float_box(shape: tuple[int, ...], low: float = -1.0, high: float = 1.0) -> Box:
    return Box(low=low, high=high, shape=shape, dtype=jnp.float32)


def _matching_action_space(obs_space: Any) -> Any:
    """Pick a single-agent action space compatible with an action mask in the observation space."""
    if isinstance(obs_space, AgentObservation) and obs_space.action_mask is not None:
        shape = obs_space.action_mask.shape
        if len(shape) == 1:
            return Discrete(shape[0])
        if len(shape) == 2:  # homogeneous multi-discrete mask (num_heads, num_actions)
            return MultiDiscrete(np.array([shape[1]] * shape[0], dtype=np.int32))
        raise ValueError(f"Unsupported mask shape for a proxy action space: {shape}")
    return Discrete(DEFAULT_NUM_ACTIONS)


def matching_action_spaces(obs_space: Any) -> Any:
    """Derive per-agent action spaces from a (possibly multi-agent) observation space."""
    if isinstance(obs_space, AgentObservation):
        return _matching_action_space(obs_space)
    leaves, treedef = eqx.tree_flatten_one_level(obs_space)
    if len(leaves) == 1 and leaves[0] is obs_space:
        return _matching_action_space(obs_space)
    return jax.tree.unflatten(
        treedef, [_matching_action_space(leaf) for leaf in leaves]
    )


def mirror_agent_structure(space: Any, leaf_factory: Callable[[], Any]) -> Any:
    """Build a pytree with the same first-level (agent) structure as `space`,
    filling each agent slot with `leaf_factory()`."""
    leaves, treedef = eqx.tree_flatten_one_level(space)
    return jax.tree.unflatten(treedef, [leaf_factory() for _ in leaves])


# #
# Observation (input) spaces
#
# Single-agent cases:
def obs_box_scalar():
    """Scalar Box -> 1d processor with in_features=1."""
    return _float_box(())


def obs_box_vector():
    """1d Box -> 1d (Identity) processor."""
    return _float_box((6,))


def obs_discrete_scalar():
    """Discrete -> shape () -> 1d processor."""
    return Discrete(5)


def obs_multidiscrete():
    """MultiDiscrete observation -> 1d processor of length nvec."""
    return MultiDiscrete(np.array([4, 4, 4], dtype=np.int32))


def obs_box_image_hwc():
    """3d Box (H, W, C) -> CNN with channels-last."""
    return _float_box((6, 6, 3))


def obs_box_image_chw():
    """3d Box (C, H, W) -> CNN with channels-first."""
    return _float_box((3, 6, 6))


def obs_dict_flat():
    """Composite (single-agent) observation mixing a vector and a discrete."""
    return {"vector": _float_box((6,)), "flag": Discrete(4)}


def obs_dict_all_discrete():
    """Composite (single-agent) observation with only Discrete leaves."""
    return {
        "position": Discrete(10),
        "velocity": Discrete(10),
        "step": Discrete(100),
    }


def obs_dict_nested():
    """One level of nesting inside a composite observation."""
    return {
        "sensors": {"left": _float_box((4,)), "right": _float_box((3,))},
        "scalar": _float_box((6,)),
    }


def obs_dict_vector_and_image():
    """Composite observation mixing a 1d and a 3d (CNN) branch."""
    return {"vector": _float_box((6,)), "image": _float_box((6, 6, 3))}


def obs_tuple():
    """Tuple-structured (single-agent) composite observation."""
    return (_float_box((4,)), _float_box((3,)))


def obs_agent_observation_vector():
    """Masked observation: vector obs + 1d action mask (-> Discrete action)."""
    return AgentObservation(observation=_float_box((6,)), action_mask=_bool_box((4,)))


def obs_agent_observation_dict():
    """Masked observation whose observation is itself a composite pytree."""
    return AgentObservation(
        observation={"grid": _float_box((6,)), "scalar": _float_box(())},
        action_mask=_bool_box((4,)),
    )


def obs_agent_observation_multidiscrete_mask():
    """Masked observation with a 2d mask (-> homogeneous MultiDiscrete action)."""
    return AgentObservation(observation=_float_box((6,)), action_mask=_bool_box((3, 4)))


def obs_agent_observation_critic():
    """Local policy obs plus a larger critic (CTDE-style) observation."""
    return AgentObservation(
        observation=_float_box((6,)),
        critic_observation=_float_box((12,)),
    )


def obs_agent_observation_masked_critic():
    """Action mask + a distinct critic observation."""
    return AgentObservation(
        observation=_float_box((6,)),
        action_mask=_bool_box((4,)),
        critic_observation=_float_box((12,)),
    )


# Multi-agent cases: the first level of the pytree is the agent dimension.
def obs_ma_dict_homogeneous():
    return {"agent_0": _float_box((6,)), "agent_1": _float_box((6,))}


def obs_ma_dict_heterogeneous():
    """Agents with different observation shapes."""
    return {"agent_0": _float_box((6,)), "agent_1": _float_box((8,))}


def obs_ma_dict_discrete():
    return {"agent_0": Discrete(5), "agent_1": Discrete(5)}


def obs_ma_dict_image():
    return {"agent_0": _float_box((6, 6, 3)), "agent_1": _float_box((6, 6, 3))}


def obs_ma_tuple_players():
    """Tuple-structured (players) multi-agent observation."""
    return (_float_box((6,)), _float_box((6,)))


def obs_ma_list_multidiscrete():
    """List-structured multi-agent observation (one MultiDiscrete per agent)."""
    nvec = np.array([3, 3, 3], dtype=np.int32)
    return [MultiDiscrete(nvec) for _ in range(3)]


def obs_ma_heterogeneous_discrete_multidiscrete():
    """Per-agent heterogeneous observation types (Discrete vs MultiDiscrete)."""
    return {
        "agent_0": Discrete(2),
        "agent_1": MultiDiscrete(np.array([2, 3], dtype=np.int32)),
    }


def obs_ma_dict_nested_masked_agent_observation():
    return {
        "agent_0": AgentObservation(
            observation=_float_box((6,)), action_mask=_bool_box((4,))
        ),
        "agent_1": AgentObservation(
            observation=_float_box((6,)), action_mask=_bool_box((4,))
        ),
    }


def obs_ma_dict_critic_observation():
    """Homogeneous agents with a larger shared-style critic observation."""
    return {
        "agent_0": AgentObservation(
            observation=_float_box((6,)), critic_observation=_float_box((12,))
        ),
        "agent_1": AgentObservation(
            observation=_float_box((6,)), critic_observation=_float_box((12,))
        ),
    }


SINGLE_AGENT_OBS_SPACES: dict[str, Callable[[], Any]] = {
    "obs_box_scalar": obs_box_scalar,
    "obs_box_vector": obs_box_vector,
    "obs_discrete_scalar": obs_discrete_scalar,
    "obs_multidiscrete": obs_multidiscrete,
    "obs_box_image_hwc": obs_box_image_hwc,
    "obs_box_image_chw": obs_box_image_chw,
    "obs_dict_flat": obs_dict_flat,
    "obs_dict_all_discrete": obs_dict_all_discrete,
    "obs_dict_nested": obs_dict_nested,
    "obs_dict_vector_and_image": obs_dict_vector_and_image,
    "obs_tuple": obs_tuple,
    "obs_agent_observation_vector": obs_agent_observation_vector,
    "obs_agent_observation_dict": obs_agent_observation_dict,
    "obs_agent_observation_multidiscrete_mask": obs_agent_observation_multidiscrete_mask,
    "obs_agent_observation_critic": obs_agent_observation_critic,
    "obs_agent_observation_masked_critic": obs_agent_observation_masked_critic,
}

MULTI_AGENT_OBS_SPACES: dict[str, Callable[[], Any]] = {
    "obs_ma_dict_homogeneous": obs_ma_dict_homogeneous,
    "obs_ma_dict_heterogeneous": obs_ma_dict_heterogeneous,
    "obs_ma_dict_discrete": obs_ma_dict_discrete,
    "obs_ma_dict_image": obs_ma_dict_image,
    "obs_ma_tuple_players": obs_ma_tuple_players,
    "obs_ma_list_multidiscrete": obs_ma_list_multidiscrete,
    "obs_ma_heterogeneous_discrete_multidiscrete": obs_ma_heterogeneous_discrete_multidiscrete,
    "obs_ma_dict_nested_masked_agent_observation": obs_ma_dict_nested_masked_agent_observation,
    "obs_ma_dict_critic_observation": obs_ma_dict_critic_observation,
}


# Single-agent: exercise the different branches of `PyTreeOutputNetwork`.
def act_discrete():
    """Single-head categorical."""
    return Discrete(4)


def act_multidiscrete_homogeneous():
    """Multi-head (stacked/vmapped) categorical."""
    return MultiDiscrete(np.array([4, 4, 4], dtype=np.int32))


def act_multidiscrete_single_head():
    return MultiDiscrete(np.array([5], dtype=np.int32))


def act_box_scalar():
    """Scalar continuous head."""
    return _float_box(())


def act_box_vector():
    """Multi-dimensional continuous head."""
    return _float_box((3,))


def act_dict_discrete():
    """Composite (single-agent) discrete action -> Joint of categoricals."""
    return {"move": Discrete(3), "turn": Discrete(2)}


def act_tuple_discrete():
    return (Discrete(3), Discrete(3))


def act_dict_box():
    """Composite (single-agent) continuous action."""
    return {"throttle": _float_box((2,)), "steer": _float_box((3,))}


def act_dict_mixed():
    """Mixed discrete + continuous composite action."""
    return {"move": Discrete(3), "throttle": _float_box((1,))}


# Multi-agent: the first level of the pytree is the agent dimension.
def act_ma_dict_discrete():
    return {"agent_0": Discrete(4), "agent_1": Discrete(4)}


def act_ma_dict_box():
    return {"agent_0": _float_box((2,)), "agent_1": _float_box((2,))}


def act_ma_dict_heterogeneous_box():
    return {"agent_0": _float_box((2,)), "agent_1": _float_box((3,))}


def act_ma_dict_multidiscrete():
    return {
        "agent_0": MultiDiscrete(np.array([4, 4], dtype=np.int32)),
        "agent_1": MultiDiscrete(np.array([4, 4], dtype=np.int32)),
    }


def act_ma_tuple_discrete():
    return (Discrete(4), Discrete(4))


def act_ma_dict_composite():
    """Multi-agent where each agent's action is itself a composite pytree."""
    return {
        "agent_0": {"move": Discrete(3), "turn": Discrete(2)},
        "agent_1": {"move": Discrete(3), "turn": Discrete(2)},
    }


def act_ma_heterogeneous_composite():
    """Per-agent heterogeneous composite action trees."""
    return {
        "agent_0": {
            "move": Discrete(3),
            "turn": MultiDiscrete(np.array([2, 2], dtype=np.int32)),
        },
        "agent_1": {"move": Discrete(3), "turn": Discrete(2)},
    }


SINGLE_AGENT_ACT_SPACES: dict[str, Callable[[], Any]] = {
    "act_discrete": act_discrete,
    "act_multidiscrete_homogeneous": act_multidiscrete_homogeneous,
    "act_multidiscrete_single_head": act_multidiscrete_single_head,
    "act_box_scalar": act_box_scalar,
    "act_box_vector": act_box_vector,
    "act_dict_discrete": act_dict_discrete,
    "act_tuple_discrete": act_tuple_discrete,
    "act_dict_box": act_dict_box,
    "act_dict_mixed": act_dict_mixed,
}

MULTI_AGENT_ACT_SPACES: dict[str, Callable[[], Any]] = {
    "act_ma_dict_discrete": act_ma_dict_discrete,
    "act_ma_dict_box": act_ma_dict_box,
    "act_ma_dict_heterogeneous_box": act_ma_dict_heterogeneous_box,
    "act_ma_dict_multidiscrete": act_ma_dict_multidiscrete,
    "act_ma_tuple_discrete": act_ma_tuple_discrete,
    "act_ma_dict_composite": act_ma_dict_composite,
    "act_ma_heterogeneous_composite": act_ma_heterogeneous_composite,
}


def make_proxy_env(
    obs_space_fn: Callable[[], Any],
    action_space_fn: Callable[[], Any] | None = None,
    *,
    multi_agent: bool = False,
    episode_length: int = 100,
) -> AnyInputOutputEnv:
    """Build an `AnyInputOutputEnv` from an observation (and optional action) space.
    When no action space is given, a compatible one is derived from the observation space (respecting action masks).
    """
    if action_space_fn is None:
        action_space = matching_action_spaces(obs_space_fn())
        action_space_fn = lambda: action_space
    return AnyInputOutputEnv(
        observation_space_fn=obs_space_fn,
        action_space_fn=action_space_fn,
        _multi_agent=multi_agent,
        episode_length=episode_length,
    )


REPRESENTATIVE_ENVS: dict[str, AnyInputOutputEnv] = {
    # Single-agent
    "vector_discrete": make_proxy_env(obs_box_vector, act_discrete),
    "image_discrete": make_proxy_env(obs_box_image_hwc, act_discrete),
    "dict_box": make_proxy_env(obs_dict_flat, act_box_vector),
    "multidiscrete_multidiscrete": make_proxy_env(
        obs_multidiscrete, act_multidiscrete_homogeneous
    ),
    "masked_discrete": make_proxy_env(obs_agent_observation_vector),  # mask -> Discrete
    "critic_obs_discrete": make_proxy_env(obs_agent_observation_critic),
    # Multi-agent (first pytree level is the agent dimension)
    "ma_dict_discrete": make_proxy_env(
        obs_ma_dict_homogeneous, act_ma_dict_discrete, multi_agent=True
    ),
    "ma_heterogeneous_box": make_proxy_env(
        obs_ma_dict_heterogeneous, act_ma_dict_heterogeneous_box, multi_agent=True
    ),
}

DISCRETE_ACTION_ENV_NAMES = [
    "vector_discrete",
    "image_discrete",
    "multidiscrete_multidiscrete",
    "masked_discrete",
    "ma_dict_discrete",
]


def observation_arrays(tree: Any) -> list[Array]:
    """Leaves of an observation (or observation-space) pytree, ignoring action masks.

    `AgentObservation` nodes are reduced to their `.observation` (and
    `.critic_observation` when present) so that masks are not treated as
    observation leaves.
    """
    stripped = jax.tree.map(
        lambda o: (
            {"observation": o.observation, "critic": o.critic_observation}
            if isinstance(o, AgentObservation)
            else o
        ),
        tree,
        is_leaf=lambda x: isinstance(x, AgentObservation),
    )
    return jax.tree.leaves(stripped)


def rollout(
    env: AnyInputOutputEnv,
    key: PRNGKeyArray,
    *,
    num_steps: int,
    num_envs: int = 1,
) -> Transition:
    """Simple vectorized rollout on a Env, constructing a rollout of `Transition` of shape `(num_steps, num_envs, ...)`.
    Randomly sampled actions, as this will yield varying observations and rewards.
    """
    venv = jym.VecEnvWrapper(env)
    reset_key, scan_key = jax.random.split(key)
    obs, state = venv.reset(jax.random.split(reset_key, num_envs))

    def step(carry, step_key):
        obs, state = carry
        act_key, env_key = jax.random.split(step_key)
        action = jax.vmap(env.sample_action)(jax.random.split(act_key, num_envs))
        timestep, state = venv.step(jax.random.split(env_key, num_envs), state, action)
        transition = Transition(
            observation=obs,
            action=action,
            reward=timestep.reward,
            terminated=timestep.terminated,
            truncated=timestep.truncated,
        )
        return (timestep.observation, state), transition

    _, transitions = jax.lax.scan(
        step, (obs, state), jax.random.split(scan_key, num_steps)
    )
    return transitions
