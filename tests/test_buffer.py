import jax
import jax.numpy as jnp
import pytest

from jaxnasium.algorithms.core import (
    PrioritizedTransitionBuffer,
    Transition,
    TransitionBuffer,
)

AGENTS = ("agent0", "agent1")
SHORT_LEN = 2
LONG_LEN = 4


def _make_single_and_multi_agent_transition(
    length: int,
    *,
    num_envs: int = 2,
    feature_dim: int = 3,
):
    time = jnp.arange(length)

    single_agent_transition = Transition(
        observation=jnp.ones((length, num_envs, feature_dim)) * time[:, None, None],
        action=jnp.ones((length, num_envs)) * time[:, None],
        reward=jnp.ones((length, num_envs)) * time[:, None],
        terminated=jnp.zeros((length, num_envs), dtype=bool),
        truncated=jnp.zeros((length, num_envs), dtype=bool),
    )

    multi_agent_transition = jax.tree.map(  # a copy per agent
        lambda x: single_agent_transition, {"agent0": 0, "agent1": 0}
    )

    return single_agent_transition, multi_agent_transition


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("with_replacement", [True, False])
def test_buffer_incremental_fill_samples_valid_region_sa(
    with_replacement: bool, n_steps: int
) -> None:
    """Test if samples stay within valid buffer range (single agent)."""
    max_size = 100
    chunk_size = max_size // 10
    transition, _ = _make_single_and_multi_agent_transition(max_size)

    buffer = TransitionBuffer(
        max_size=max_size,
        sample_batch_size=chunk_size,
        data_sample=transition,
        n_steps=n_steps,
    )
    for i in range(4):
        start_index = i * chunk_size
        chunk_transition = jax.tree.map(
            lambda x, _start_index=start_index: x[
                _start_index : _start_index + chunk_size
            ],
            transition,
        )
        buffer = buffer.insert(chunk_transition)

        for k in range(10):
            samples = buffer.sample(
                jax.random.PRNGKey(k), with_replacement=with_replacement
            )
            assert jnp.all(jnp.ravel(samples.reward) < start_index + chunk_size)


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("with_replacement", [True, False])
def test_buffer_incremental_fill_samples_valid_region_ma(
    with_replacement: bool, n_steps: int
) -> None:
    """Test if samples stay within valid buffer range (multi agent)."""
    max_size = 100
    chunk_size = max_size // 10
    _, transition = _make_single_and_multi_agent_transition(max_size)

    buffer = TransitionBuffer(
        max_size=max_size,
        sample_batch_size=chunk_size,
        data_sample=transition,
        n_steps=n_steps,
    )
    for i in range(4):
        start_index = i * chunk_size
        chunk_transition = jax.tree.map(
            lambda x, _start_index=start_index: x[
                _start_index : _start_index + chunk_size
            ],
            transition,
        )
        buffer = buffer.insert(chunk_transition)

        for k in range(10):
            samples = buffer.sample(
                jax.random.PRNGKey(k), with_replacement=with_replacement
            )
            assert jnp.all(
                jnp.ravel(samples["agent0"].reward) < start_index + chunk_size  # type: ignore
            )


def test_per_update_priorities_sa() -> None:
    transition, _ = _make_single_and_multi_agent_transition(4, num_envs=2)
    buffer = PrioritizedTransitionBuffer(
        max_size=8,
        sample_batch_size=3,
        data_sample=transition,
    )
    buffer = buffer.insert(transition)

    batch = buffer.sample(jax.random.PRNGKey(0))
    assert batch.PER_index is not None
    td_errors = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)
    buffer = buffer.update_priorities(batch.PER_index, td_errors)

    eps = 1e-6
    for flat_idx, td_error in zip(batch.PER_index, td_errors, strict=True):
        step = flat_idx // 2
        env = flat_idx % 2
        assert buffer.priorities[step, env] == pytest.approx(abs(td_error) + eps)


def test_per_update_priorities_ma() -> None:
    _, transition = _make_single_and_multi_agent_transition(4, num_envs=2)
    buffer = PrioritizedTransitionBuffer(
        max_size=8,
        sample_batch_size=3,
        data_sample=transition,
    )
    buffer = buffer.insert(transition)

    batch = buffer.sample(jax.random.PRNGKey(0))
    per_index = batch["agent0"].PER_index  # type: ignore
    assert per_index is not None
    assert jnp.array_equal(per_index, batch["agent1"].PER_index)  # type: ignore

    td_errors = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)
    buffer = buffer.update_priorities(per_index, td_errors)

    eps = 1e-6
    for flat_idx, td_error in zip(per_index, td_errors, strict=True):
        step = flat_idx // 2
        env = flat_idx % 2
        assert buffer.priorities[step, env] == pytest.approx(abs(td_error) + eps)


def test_valid_indices_infer_next_obs_drops_truncated_and_last_slot() -> None:
    length = 8
    num_envs = 2
    transition, _ = _make_single_and_multi_agent_transition(length, num_envs=num_envs)
    truncated = transition.truncated.at[3].set(True)
    truncated = truncated.at[5, 1].set(True)
    transition = transition.replace(truncated=truncated)

    buffer = TransitionBuffer(
        max_size=length * num_envs,
        sample_batch_size=4,
        data_sample=transition,
    )
    buffer = buffer.insert(transition)

    valid = buffer._get_flat_valid_start_indices(with_next_obs=True).reshape(
        length, num_envs
    )
    assert not jnp.any(valid[-1])
    assert not jnp.any(valid[3])
    assert not valid[5, 1]
    assert valid[5, 0]
    assert jnp.all(valid[0:3])
    assert jnp.all(valid[4])


def test_valid_indices_keep_truncated_when_next_obs_is_stored() -> None:
    # If we explicitely pass a next_observation in the transition; we just skip the whole inferring
    length = 6
    transition, _ = _make_single_and_multi_agent_transition(length)
    transition = transition.replace(
        truncated=transition.truncated.at[2].set(True),
        next_observation=transition.observation + 1.0,
    )
    buffer = TransitionBuffer(
        max_size=length * 2,
        sample_batch_size=4,
        data_sample=transition,
    )
    buffer = buffer.insert(transition)

    valid = buffer._get_flat_valid_start_indices(with_next_obs=True).reshape(length, 2)
    assert jnp.all(valid[2])
    assert jnp.all(valid[-1])


def test_valid_indices_n_step_infer_next_obs_drops_windows_with_truncation() -> None:
    length = 10
    n_steps = 3
    transition, _ = _make_single_and_multi_agent_transition(length)
    transition = transition.replace(truncated=transition.truncated.at[4].set(True))
    buffer = TransitionBuffer(
        max_size=length * 2,
        sample_batch_size=4,
        data_sample=transition,
        n_steps=n_steps,
    )
    buffer = buffer.insert(transition)

    valid = buffer._get_flat_valid_start_indices(with_next_obs=True).reshape(length, 2)
    assert not jnp.any(valid[2:5])
    assert jnp.all(valid[1])
    assert jnp.all(valid[5])
    assert not jnp.any(valid[-(n_steps):])


@pytest.mark.parametrize("n_steps", [1, 3])
def test_gather_infers_next_obs_from_following_slot(n_steps: int) -> None:
    length = 12
    transition, _ = _make_single_and_multi_agent_transition(length)
    buffer = TransitionBuffer(
        max_size=length * 2,
        sample_batch_size=8,
        data_sample=transition,
        n_steps=n_steps,
    )
    buffer = buffer.insert(transition)

    samples = buffer.sample(jax.random.PRNGKey(0), with_next_obs=True)
    assert samples.next_observation is not None
    assert jnp.all(samples.next_observation[..., 0] == samples.observation[..., 0] + 1)


def test_gather_uses_stored_next_obs_when_present() -> None:
    length = 8
    transition, _ = _make_single_and_multi_agent_transition(length)
    transition = transition.replace(next_observation=transition.observation + 100.0)
    buffer = TransitionBuffer(
        max_size=length * 2,
        sample_batch_size=8,
        data_sample=transition,
    )
    buffer = buffer.insert(transition)

    samples = buffer.sample(jax.random.PRNGKey(1), with_next_obs=True)
    assert samples.next_observation is not None
    assert jnp.allclose(samples.next_observation, samples.observation + 100.0)


def test_gather_without_next_obs_leaves_field_unset() -> None:
    length = 8
    transition, _ = _make_single_and_multi_agent_transition(length)
    buffer = TransitionBuffer(
        max_size=length * 2,
        sample_batch_size=8,
        data_sample=transition,
    )
    buffer = buffer.insert(transition)

    samples = buffer.sample(jax.random.PRNGKey(2), with_next_obs=False)
    assert samples.next_observation is None


def test_gather_infers_next_obs_multi_agent() -> None:
    length = 12
    _, transition = _make_single_and_multi_agent_transition(length)
    buffer = TransitionBuffer(
        max_size=length * 2,
        sample_batch_size=8,
        data_sample=transition,
    )
    buffer = buffer.insert(transition)

    samples = buffer.sample(jax.random.PRNGKey(0), with_next_obs=True)
    for agent in AGENTS:
        agent_batch = samples[agent]  # type: ignore[index]
        assert agent_batch.next_observation is not None
        assert jnp.all(
            agent_batch.next_observation[..., 0] == agent_batch.observation[..., 0] + 1
        )


def test_valid_indices_multi_agent_excludes_if_any_agent_truncated() -> None:
    length = 8
    _, multi = _make_single_and_multi_agent_transition(length)
    multi["agent1"] = multi["agent1"].replace(
        truncated=multi["agent1"].truncated.at[3].set(True)
    )
    buffer = TransitionBuffer(
        max_size=length * 2, sample_batch_size=4, data_sample=multi
    )
    buffer = buffer.insert(multi)

    valid = buffer._get_flat_valid_start_indices(with_next_obs=True).reshape(length, 2)
    assert not jnp.any(valid[3])
    assert jnp.all(valid[2])
