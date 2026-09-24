import jax
import jax.numpy as jnp
import pytest
from _proxy_test_envs import REPRESENTATIVE_ENVS, observation_arrays, rollout

from jaxnasium import AgentObservation
from jaxnasium.algorithms.core import Normalizer
from jaxnasium.algorithms.core._normalization import RunningStatisticsState

SEED = jax.random.PRNGKey(0)
BATCH_SIZE = 512


@pytest.mark.parametrize(
    "env", REPRESENTATIVE_ENVS.values(), ids=REPRESENTATIVE_ENVS.keys()
)
def test_obs_normalization_standardizes_and_preserves_structure(env):
    obs_batch = jax.vmap(env.sample_observation)(jax.random.split(SEED, BATCH_SIZE))

    normalizer = Normalizer(
        obs_space=env.observation_space, normalize_obs=True, normalize_rew=False
    )
    normalizer = normalizer.update_obs(obs_batch)
    normalized = normalizer.normalize_obs(obs_batch)

    assert jax.tree.structure(normalized) == jax.tree.structure(obs_batch)

    # Each observation leaf is standardized to ~zero mean / unit std over the batch.
    for leaf in observation_arrays(normalized):
        leaf = leaf.astype(jnp.float32)
        assert jnp.allclose(jnp.mean(leaf, axis=0), 0.0, atol=1e-3)
        assert jnp.allclose(jnp.std(leaf, axis=0), 1.0, atol=1e-2)


def test_obs_normalization_preserves_action_mask():
    env = REPRESENTATIVE_ENVS["masked_discrete"]
    obs_batch = jax.vmap(env.sample_observation)(jax.random.split(SEED, BATCH_SIZE))

    normalizer = Normalizer(
        obs_space=env.observation_space, normalize_obs=True, normalize_rew=False
    )
    normalizer = normalizer.update_obs(obs_batch)
    normalized = normalizer.normalize_obs(obs_batch)

    assert isinstance(normalized, AgentObservation)
    assert normalized.action_mask is not None
    assert jnp.array_equal(normalized.action_mask, obs_batch.action_mask)
    assert not jnp.allclose(normalized.observation, obs_batch.observation)


def test_obs_normalization_normalizes_critic_observation():
    env = REPRESENTATIVE_ENVS["critic_obs_discrete"]
    obs_batch = jax.vmap(env.sample_observation)(jax.random.split(SEED, BATCH_SIZE))

    normalizer = Normalizer(
        obs_space=env.observation_space, normalize_obs=True, normalize_rew=False
    )
    normalizer = normalizer.update_obs(obs_batch)
    normalized = normalizer.normalize_obs(obs_batch)

    assert isinstance(normalized, AgentObservation)
    assert normalized.critic_observation is not None
    assert not jnp.allclose(normalized.observation, obs_batch.observation)
    assert not jnp.allclose(normalized.critic_observation, obs_batch.critic_observation)


def test_reward_normalization_runs_and_scales():
    env = REPRESENTATIVE_ENVS["vector_discrete"]
    num_steps, num_envs = 16, 4
    transition = rollout(env, SEED, num_steps=num_steps, num_envs=num_envs)

    normalizer = Normalizer(
        normalize_obs=False,
        normalize_rew=True,
        gamma=0.99,
        rew_shape=(num_envs,),
    )
    normalizer = normalizer.update(transition)
    normalized = normalizer.normalize_reward(transition.reward)

    assert normalized.shape == transition.reward.shape
    assert jnp.all(jnp.isfinite(normalized))
    assert not jnp.allclose(normalized, transition.reward)


def _constant_reward_normalizer(num_envs, gamma=0.99):
    return Normalizer(
        normalize_obs=False, normalize_rew=True, gamma=gamma, rew_shape=(num_envs,)
    )


def test_return_accumulator_runs_along_the_time_axis():
    num_steps, num_envs, gamma, iters = 8, 4, 0.99, 5
    reward = jnp.ones((num_steps, num_envs))
    done = jnp.zeros((num_steps, num_envs))

    normalizer = _constant_reward_normalizer(num_envs, gamma)
    for _ in range(iters):
        normalizer = normalizer.update_reward(reward, done)

    # Reference: plain python loop over every env step.
    returns, seen = jnp.zeros((num_envs,)), []
    for _ in range(iters):
        for t in range(num_steps):
            returns = returns * gamma + reward[t]
            seen.append(returns)
    seen = jnp.stack(seen)

    assert normalizer.returns is not None and normalizer.reward is not None
    assert jnp.allclose(normalizer.returns, returns, atol=1e-4)
    assert jnp.allclose(normalizer.reward.std, jnp.std(seen), rtol=1e-3)
    assert normalizer.reward.count == iters * num_steps * num_envs


def test_return_accumulator_resets_on_done():
    """A slot's accumulator resets after the step where the episode ended."""
    num_envs, gamma = 2, 0.99
    reward = jnp.ones((3, num_envs))
    # env 0 ends on step 1, env 1 never ends.
    done = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    normalizer = _constant_reward_normalizer(num_envs, gamma).update_reward(
        reward, done
    )
    expected = jnp.array([1.0, 1.0 + gamma + gamma**2])
    assert normalizer.returns is not None
    assert jnp.allclose(normalizer.returns, expected, atol=1e-5)


def test_constant_reward_does_not_explode_on_the_first_update():
    """A (near-)zero-variance batch must not be divided by the bare std floor."""
    num_envs = 4
    reward = jnp.ones((1, num_envs))  # single step -> no variance along time either
    done = jnp.zeros((1, num_envs))

    normalizer = _constant_reward_normalizer(num_envs).update_reward(reward, done)
    normalized = normalizer.normalize_reward(reward)

    clip = normalizer.clip_value_rew
    # if default is later set to None, we default to a large number
    assert jnp.all(jnp.abs(normalized) <= (clip or 1e8) + 1e-6)


def test_all_true_mask_matches_unmasked():
    batch = jax.random.normal(SEED, (128, 3))
    masked = RunningStatisticsState(jnp.zeros(3)).update(
        batch, mask=jnp.ones(128, dtype=bool)
    )
    plain = RunningStatisticsState(jnp.zeros(3)).update(batch)
    assert jnp.allclose(masked.mean, plain.mean, atol=1e-5)
    assert jnp.allclose(masked.std, plain.std, atol=1e-5)


def test_all_false_mask_is_a_noop_not_nan():
    """A rollout with no terminal steps yields an all-zero mask; 0/0 must not appear."""
    batch = jax.random.normal(SEED, (32, 3)) + 5.0
    stats = RunningStatisticsState(jnp.zeros(3)).update(
        batch, mask=jnp.zeros(32, dtype=bool)
    )
    assert jnp.all(jnp.isfinite(stats.mean)) and jnp.all(jnp.isfinite(stats.std))
    assert jnp.allclose(stats.mean, 0.0) and stats.count == 0


def test_normalized_outputs_are_clipped():
    dummy = jnp.zeros(2)
    data = jax.random.normal(SEED, (256, 2))
    outlier = jnp.array([[1e4, -1e4]])

    clipped = Normalizer(
        dummy_obs=dummy,
        normalize_obs=True,
        normalize_rew=False,
        clip_value_obs=10.0,
    ).update_obs(data)
    assert jnp.allclose(clipped.normalize_obs(outlier), jnp.array([[10.0, -10.0]]))

    unclipped = Normalizer(
        dummy_obs=dummy,
        normalize_obs=True,
        normalize_rew=False,
        clip_value_obs=None,
    ).update_obs(data)
    assert jnp.all(jnp.abs(unclipped.normalize_obs(outlier)) > 10.0)


def test_normalizer_is_noop_when_disabled():
    env = REPRESENTATIVE_ENVS["vector_discrete"]
    transition = rollout(env, SEED, num_steps=8, num_envs=2)

    normalizer = Normalizer(normalize_obs=False, normalize_rew=False)
    normalizer = normalizer.update(transition)

    assert jnp.array_equal(
        normalizer.normalize_obs(transition.observation), transition.observation
    )
    assert jnp.array_equal(
        normalizer.normalize_reward(transition.reward), transition.reward
    )


def test_running_statistics_matches_batch_moments():
    data = jax.random.normal(SEED, (1000, 4)) * 3.0 + 5.0

    stats = RunningStatisticsState(jnp.zeros(4))
    stats = stats.update(data)

    assert jnp.allclose(stats.mean, jnp.mean(data, axis=0), atol=1e-3)
    assert jnp.allclose(stats.std, jnp.std(data, axis=0), atol=1e-3)

    normalizer = Normalizer(
        dummy_obs=jnp.zeros(4), normalize_obs=True, normalize_rew=False
    ).update_obs(data)
    normalized = normalizer.normalize_obs(data)
    assert jnp.allclose(jnp.mean(normalized, axis=0), 0.0, atol=1e-3)
    assert jnp.allclose(jnp.std(normalized, axis=0), 1.0, atol=1e-2)


def test_running_statistics_incremental_matches_single_batch():
    data = jax.random.normal(SEED, (1000, 3))

    single = RunningStatisticsState(jnp.zeros(3)).update(data)
    incremental = RunningStatisticsState(jnp.zeros(3))
    for chunk in data.reshape(10, 100, 3):
        incremental = incremental.update(chunk)

    assert jnp.allclose(single.mean, incremental.mean, atol=1e-4)
    assert jnp.allclose(single.std, incremental.std, atol=1e-4)
