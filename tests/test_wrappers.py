import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _proxy_test_envs import (
    DISCRETE_ACTION_ENV_NAMES,
    REPRESENTATIVE_ENVS,
    make_proxy_env,
    obs_box_vector,
    observation_arrays,
)
from jaxtyping import PRNGKeyArray

import jaxnasium as jym
from jaxnasium import Box, Discrete, Environment, MultiDiscrete

NUM_ENVS = 4
SEED = jax.random.PRNGKey(0)


def _make_base_env() -> Environment:
    return jym.make("CartPole-v1", wrappers=[])


def _make_norm_env() -> Environment:
    """Different environment because CartPole only outputs 1.0 rewards.
    Might as well use this for all tests here, but oh well."""
    return jym.make("Acrobot-v1", wrappers=[])


def _run_single_reset_step(env: Environment, seed: PRNGKeyArray = SEED):
    reset_key, step_key, action_key = jax.random.split(seed, 3)
    obs, state = env.reset(reset_key)
    action = env.sample_action(action_key)
    timestep, state = env.step(step_key, state, action)
    return obs, state, timestep


def _run_vec_reset_step(
    env: Environment, num_envs: int = NUM_ENVS, seed: PRNGKeyArray = SEED
):
    reset_key, step_key, action_key = jax.random.split(seed, 3)
    keys = jax.random.split(reset_key, num_envs)
    obs, state = env.reset(keys)
    actions = jax.vmap(env.action_space.sample)(jax.random.split(action_key, num_envs))
    step_keys = jax.random.split(step_key, num_envs)
    timestep, state = env.step(step_keys, state, actions)
    return obs, state, timestep


def _num_flat_actions(action_space) -> int:
    """Number of actions a (multi-)discrete (pytree of) space collapses into."""
    total = 1
    for space in jax.tree.leaves(action_space):
        if hasattr(space, "n"):
            total *= int(space.n)
        elif hasattr(space, "nvec"):
            total *= int(np.prod(np.asarray(space.nvec)))
        else:
            raise AssertionError(f"Expected a (multi-)discrete space, got {space}")
    return total


@pytest.mark.parametrize(
    "env", REPRESENTATIVE_ENVS.values(), ids=REPRESENTATIVE_ENVS.keys()
)
def test_flatten_observation_wrapper(env: Environment):
    wrapped = jym.FlattenObservationWrapper(env)

    orig_obs, _ = env.reset(SEED)
    flat_obs, _ = wrapped.reset(SEED)

    orig_leaves = observation_arrays(orig_obs)
    flat_leaves = observation_arrays(flat_obs)
    space_leaves = observation_arrays(wrapped.observation_space)

    assert len(orig_leaves) == len(flat_leaves) == len(space_leaves)
    for orig, flat, space in zip(orig_leaves, flat_leaves, space_leaves):
        assert flat.ndim <= 1  # every observation leaf is flattened
        assert flat.size == orig.size  # no elements are lost
        # the declared flat space matches the number of elements in the flat obs
        assert flat.size == int(np.prod(space.shape))


@pytest.mark.parametrize("name", DISCRETE_ACTION_ENV_NAMES)
def test_flatten_action_space_wrapper(name: str):
    env = REPRESENTATIVE_ENVS[name]
    wrapped = jym.FlattenActionSpaceWrapper(env)
    flat_space = wrapped.action_space

    if env.multi_agent:
        orig_per_agent, _ = eqx.tree_flatten_one_level(env.action_space)
        flat_per_agent, _ = eqx.tree_flatten_one_level(flat_space)
        for orig, flat in zip(orig_per_agent, flat_per_agent):
            assert isinstance(flat, Discrete)
            assert flat.n == _num_flat_actions(orig)
    else:
        assert isinstance(flat_space, Discrete)
        assert flat_space.n == _num_flat_actions(env.action_space)

    _, _, timestep = _run_single_reset_step(wrapped)
    assert jnp.all(jnp.isfinite(jnp.asarray(jax.tree.leaves(timestep.reward))))


def test_scale_reward_wrapper():
    base_env = _make_base_env()
    scale = 0.25
    env = jym.ScaleRewardWrapper(base_env, scale=scale)

    _, _base_state, base_timestep = _run_single_reset_step(base_env)
    _, _, wrapped_timestep = _run_single_reset_step(env)

    assert wrapped_timestep.reward == pytest.approx(base_timestep.reward * scale)


def test_transform_reward_wrapper():
    base_env = _make_base_env()
    env = jym.TransformRewardWrapper(
        base_env,
        transform_fn=lambda reward: reward + 1.0,
    )

    _, _, base_timestep = _run_single_reset_step(base_env)
    _, _, wrapped_timestep = _run_single_reset_step(env)

    assert wrapped_timestep.reward == pytest.approx(base_timestep.reward + 1.0)


def test_log_wrapper():
    env = jym.LogWrapper(_make_base_env())

    key = SEED
    obs, state = env.reset(key)
    assert obs.shape == env.observation_space.shape

    for _ in range(5):
        key, step_key, action_key = jax.random.split(key, 3)
        action = env.sample_action(action_key)
        timestep, state = env.step(step_key, state, action)
        assert "returned_episode_returns" in timestep.info
        assert "returned_episode_lengths" in timestep.info
        assert "returned_episode" in timestep.info
        assert "timestep" in timestep.info


def test_vec_env_wrapper():
    env = jym.VecEnvWrapper(_make_base_env())

    obs, _, _ = _run_vec_reset_step(env)

    assert obs.shape[0] == NUM_ENVS


def test_normalize_vec_obs_wrapper():
    raw_env = jym.VecEnvWrapper(_make_norm_env())
    norm_env = jym.NormalizeVecObsWrapper(jym.VecEnvWrapper(_make_norm_env()))

    keys = jax.random.split(SEED, NUM_ENVS)
    _raw_obs, raw_state = raw_env.reset(keys)
    norm_obs, norm_state = norm_env.reset(keys)

    assert norm_obs.shape[0] == NUM_ENVS
    assert jnp.allclose(jnp.mean(norm_obs, axis=0), 0.0, atol=1e-2)

    actions = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
    raw_timestep, _ = raw_env.step(keys, raw_state, actions)
    norm_timestep, _ = norm_env.step(keys, norm_state, actions)

    assert norm_timestep.observation.shape[0] == NUM_ENVS
    assert not jnp.allclose(raw_timestep.observation, norm_timestep.observation)


def test_normalize_vec_reward_wrapper():
    raw_env = jym.VecEnvWrapper(_make_norm_env())
    norm_env = jym.NormalizeVecRewardWrapper(
        jym.VecEnvWrapper(_make_norm_env()), gamma=0.99
    )

    keys = jax.random.split(SEED, NUM_ENVS)
    actions = jnp.array([0, 1, 2, 0])

    _, raw_state = raw_env.reset(keys)
    _, norm_state = norm_env.reset(keys)

    raw_timestep, _ = raw_env.step(keys, raw_state, actions)
    norm_timestep, _ = norm_env.step(keys, norm_state, actions)

    raw_rewards = raw_timestep.reward
    assert jnp.all(raw_rewards == -1.0)
    assert jnp.all(jnp.isfinite(norm_timestep.reward))
    assert not jnp.allclose(norm_timestep.reward, raw_rewards)


def test_discrete_action_wrapper():
    num_actions = 11
    base_env = make_proxy_env(
        obs_box_vector, lambda: Box(low=-2.0, high=5.0, shape=(), dtype=jnp.float32)
    )
    env = jym.DiscreteActionWrapper(base_env, num_actions=num_actions)

    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == num_actions
    assert env.original_action_space == base_env.action_space

    # sample an action from the new env
    discrete_action = jnp.ones_like(env.sample_action(SEED))
    discrete_action = discrete_action * (num_actions // 2)  # middle action

    low = base_env.action_space.low
    high = base_env.action_space.high
    middle = (low + high) / 2
    continuous_action = jnp.ones_like(env.original_action_space.sample(SEED))
    continuous_action = continuous_action * middle  # middle action

    # reward is the average action, so this should be roughly equal in the middle action
    _, state_base = base_env.reset(SEED)
    _, state_discrete = env.reset(SEED)
    timestep_base, state_base = base_env.step(SEED, state_base, continuous_action)
    timestep_discrete, state_discrete = env.step(SEED, state_discrete, discrete_action)
    # breakpoint()
    assert jnp.allclose(timestep_base.reward, timestep_discrete.reward)


def test_discrete_action_wrapper_vector_continuous():
    num_actions = 5
    base_env = make_proxy_env(
        obs_box_vector,
        lambda: Box(low=-2.0, high=5.0, shape=((3,)), dtype=jnp.float32),
    )
    env = jym.DiscreteActionWrapper(base_env, num_actions=num_actions)

    assert isinstance(env.action_space, MultiDiscrete)
    assert jnp.array_equal(
        env.action_space.nvec, np.array([num_actions, num_actions, num_actions])
    )

    # sample an action from the new env
    discrete_action = jym.tree.ones_like(env.sample_action(SEED))
    discrete_action = jym.tree.mul(discrete_action, (num_actions // 2))  # middle action

    low = base_env.action_space.low
    high = base_env.action_space.high
    middle = (low + high) / 2
    continuous_action = jym.tree.ones_like(env.original_action_space.sample(SEED))
    continuous_action = jym.tree.mul(continuous_action, middle)  # middle action

    # reward is the average action, so this should be roughly equal in the middle action
    _, state_base = base_env.reset(SEED)
    _, state_discrete = env.reset(SEED)
    timestep_base, state_base = base_env.step(SEED, state_base, continuous_action)
    timestep_discrete, state_discrete = env.step(SEED, state_discrete, discrete_action)
    assert jnp.allclose(timestep_base.reward, timestep_discrete.reward)


def test_discrete_action_wrapper_pytree_continuous():
    num_actions = 5
    action_space = {
        "steer": Box(low=-2.0, high=5.0, shape=(), dtype=jnp.float32),
        "thrust": Box(low=-1.0, high=1.0, shape=(3,), dtype=jnp.float32),
    }
    base_env = make_proxy_env(obs_box_vector, lambda: action_space)
    env = jym.DiscreteActionWrapper(base_env, num_actions=num_actions)

    assert isinstance(env.action_space["steer"], Discrete)
    assert env.action_space["steer"].n == num_actions
    assert isinstance(env.action_space["thrust"], MultiDiscrete)
    assert jnp.array_equal(env.action_space["thrust"].nvec, np.array([num_actions] * 3))

    discrete_action = jym.tree.mul(
        jym.tree.ones_like(env.sample_action(SEED)), num_actions // 2
    )
    continuous_action = jax.tree.map(
        lambda space: jnp.broadcast_to((space.low + space.high) / 2, space.shape),
        env.original_action_space,
    )

    _, state_base = base_env.reset(SEED)
    _, state_discrete = env.reset(SEED)
    timestep_base, _ = base_env.step(SEED, state_base, continuous_action)
    timestep_discrete, _ = env.step(SEED, state_discrete, discrete_action)
    assert jnp.allclose(timestep_base.reward, timestep_discrete.reward)


def test_is_wrapped_and_remove_wrapper():
    env = jym.ScaleRewardWrapper(
        jym.FlattenObservationWrapper(_make_base_env()), scale=2.0
    )

    assert jym.is_wrapped(env, jym.FlattenObservationWrapper)
    assert jym.is_wrapped(env, "ScaleRewardWrapper")
    assert not jym.is_wrapped(env, jym.VecEnvWrapper)

    outer_removed = jym.remove_wrapper(env, jym.ScaleRewardWrapper)
    assert not jym.is_wrapped(outer_removed, jym.ScaleRewardWrapper)
    assert jym.is_wrapped(outer_removed, jym.FlattenObservationWrapper)

    # Removing an inner wrapper leaves the wrappers around it in place.
    inner_removed = jym.remove_wrapper(env, jym.FlattenObservationWrapper)
    assert not jym.is_wrapped(inner_removed, jym.FlattenObservationWrapper)
    assert jym.is_wrapped(inner_removed, jym.ScaleRewardWrapper)

    # `unwrap_to` drops all wrappers outside of it as well.
    unwrapped = jym.unwrap_to(env, jym.FlattenObservationWrapper)
    assert not jym.is_wrapped(unwrapped, jym.FlattenObservationWrapper)
    assert not jym.is_wrapped(unwrapped, jym.ScaleRewardWrapper)


def test_combined_wrappers():
    env = _make_base_env()
    env = jym.FlattenObservationWrapper(env)
    env = jym.ScaleRewardWrapper(env, scale=0.1)
    env = jym.LogWrapper(env)
    env = jym.VecEnvWrapper(env)
    env = jym.NormalizeVecRewardWrapper(env, gamma=0.99)
    env = jym.ScaleRewardWrapper(env, scale=2.0)
    env = jym.NormalizeVecObsWrapper(env)

    obs, state, timestep = _run_vec_reset_step(env)

    assert obs.shape[0] == NUM_ENVS
    assert jnp.all(jnp.isfinite(obs))
    assert jnp.all(jnp.isfinite(timestep.reward))
    assert "returned_episode_returns" in timestep.info
    assert "returned_episode_lengths" in timestep.info

    key = SEED
    for _ in range(3):
        key, step_key, action_key = jax.random.split(key, 3)
        actions = jax.vmap(env.action_space.sample)(
            jax.random.split(action_key, NUM_ENVS)
        )
        step_keys = jax.random.split(step_key, NUM_ENVS)
        timestep, state = env.step(step_keys, state, actions)
        assert jnp.all(jnp.isfinite(timestep.observation))
        assert jnp.all(jnp.isfinite(timestep.reward))


def _stack_action_env(action_space, obs_space=None, multi_agent=False):
    return make_proxy_env(
        obs_space or obs_box_vector, lambda: action_space, multi_agent=multi_agent
    )


def test_stack_action_space_wrapper_stacks_homogeneous_spaces():
    env = _stack_action_env(tuple(Discrete(20) for _ in range(4)))
    wrapped = jym.StackActionSpaceWrapper(env)

    space = wrapped.action_space
    assert isinstance(space, MultiDiscrete)
    assert list(np.asarray(space.nvec)) == [20, 20, 20, 20]

    # the wrapped environment still sees its own action structure
    _, _, timestep = _run_single_reset_step(wrapped)
    received = timestep.info["received_action"]
    assert jax.tree.structure(received) == jax.tree.structure(env.action_space)


def test_stack_action_space_wrapper_flattens_nesting():
    """Arbitrary nesting collapses onto one axis; `MultiDiscrete` leaves keep their dims."""
    env = _stack_action_env(
        {
            "move": (Discrete(3), Discrete(3)),
            "look": MultiDiscrete(np.array([3, 3])),
        }
    )
    wrapped = jym.StackActionSpaceWrapper(env)

    assert list(np.asarray(wrapped.action_space.nvec)) == [3, 3, 3, 3]  # type: ignore

    _, _, timestep = _run_single_reset_step(wrapped)
    received = timestep.info["received_action"]
    assert received["move"][0].shape == () and received["move"][1].shape == ()
    assert received["look"].shape == (2,)


def test_stack_action_space_wrapper_round_trips_actions():
    env = _stack_action_env(tuple(Discrete(9) for _ in range(3)))
    wrapped = jym.StackActionSpaceWrapper(env)

    _, state = wrapped.reset(SEED)
    timestep, _ = wrapped.step(SEED, state, jnp.array([3, 0, 8]))

    assert tuple(int(a) for a in timestep.info["received_action"]) == (3, 0, 8)


@pytest.mark.parametrize(
    "action_space",
    [
        Discrete(7),  # a single space: nothing to stack
        (Discrete(4), Discrete(9)),  # different number of actions
        (Box(low=-1.0, high=1.0, shape=(3,)),) * 2,  # not discrete
    ],
    ids=["single", "heterogeneous", "continuous"],
)
def test_stack_action_space_wrapper_passes_through_unstackable_spaces(action_space):
    env = _stack_action_env(action_space)
    wrapped = jym.StackActionSpaceWrapper(env)

    assert jax.tree.structure(wrapped.action_space) == jax.tree.structure(action_space)
    _, _, timestep = _run_single_reset_step(wrapped)
    received = timestep.info["received_action"]
    assert jax.tree.structure(received) == jax.tree.structure(action_space)


def test_stack_action_space_wrapper_multi_agent():
    action_space = {
        "agent_0": (Discrete(6), Discrete(6), Discrete(6)),
        "agent_1": (Discrete(6), Discrete(6), Discrete(6)),
    }
    obs_space = {"agent_0": obs_box_vector(), "agent_1": obs_box_vector()}
    env = _stack_action_env(action_space, lambda: obs_space, multi_agent=True)
    wrapped = jym.StackActionSpaceWrapper(env)

    space = wrapped.action_space
    assert set(space) == {"agent_0", "agent_1"}  # type: ignore
    for agent_space in space.values():  # type: ignore
        assert isinstance(agent_space, MultiDiscrete)
        assert list(np.asarray(agent_space.nvec)) == [6, 6, 6]

    _, _, timestep = _run_single_reset_step(wrapped)
    assert jax.tree.structure(timestep.info["received_action"]) == jax.tree.structure(
        action_space
    )


def test_stack_action_space_wrapper_stacks_action_masks():
    """Action mask must survive as well !"""
    mask_space = Box(
        low=np.zeros(4, bool), high=np.ones(4, bool), shape=(4,), dtype=bool
    )
    obs_space = jym.AgentObservation(
        observation=obs_box_vector(), action_mask=(mask_space, mask_space)
    )
    env = _stack_action_env((Discrete(4), Discrete(4)), lambda: obs_space)
    wrapped = jym.StackActionSpaceWrapper(env)

    assert wrapped.observation_space.action_mask.shape == (2, 4)  # type: ignore
    obs, _ = wrapped.reset(SEED)
    assert obs.action_mask.shape == (2, 4)


def test_stack_action_space_wrapper_leaves_passed_through_masks_alone():
    def mask_space(n):
        return Box(low=np.zeros(n, bool), high=np.ones(n, bool), shape=(n,), dtype=bool)

    obs_space = jym.AgentObservation(
        observation=obs_box_vector(),
        action_mask=(mask_space(4), mask_space(4), mask_space(4), mask_space(6)),
    )
    env = _stack_action_env(
        (Discrete(4), Discrete(4), Discrete(4), Discrete(6)), lambda: obs_space
    )
    wrapped = jym.StackActionSpaceWrapper(env)

    declared = jax.tree.map(lambda s: s.shape, wrapped.observation_space.action_mask)  # type: ignore
    obs, _ = wrapped.reset(SEED)
    actual = jax.tree.map(lambda m: m.shape, obs.action_mask)

    assert declared == ((3, 4), (6,))
    assert actual == declared
