import jax
import jax.numpy as jnp
import pytest

from jaxnasium.algorithms.core import scan_callback

# One iteration of LogWrapper metrics: 1 step, 2 envs, only the first finished an episode.
FINISHED = jnp.array([[True, False]])
TIMESTEP = jnp.array([[3, 3]])

RETURNS_CASES = {
    "single_agent": (jnp.array([[1.5, 0.0]]), "episodic return=1.5"),
    "multi_agent": (
        {"agent_0": jnp.array([[1.5, 0.0]]), "agent_1": jnp.array([[2.5, 0.0]])},
        "episodic return={'agent_0': 1.5, 'agent_1': 2.5}",
    ),
}


@pytest.mark.parametrize(
    "returns, expected", RETURNS_CASES.values(), ids=RETURNS_CASES.keys()
)
def test_simple_logger_prints_finished_episode_returns(capsys, returns, expected):
    def body(carry, iteration):
        metrics = {
            "returned_episode": FINISHED,
            "returned_episode_returns": returns,
            "timestep": TIMESTEP,
        }
        return carry, metrics

    body = scan_callback(body, callback_fn="simple", callback_interval=1, n=1)
    jax.lax.scan(body, None, jnp.arange(1))
    jax.effects_barrier()

    out = capsys.readouterr().out
    assert f"global step=6, {expected}" in out
