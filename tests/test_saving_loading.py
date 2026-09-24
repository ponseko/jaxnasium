import json
import os

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _test_utils import get_valid_test_algs

import jaxnasium as jym
import jaxnasium.algorithms as jxalgs

pytestmark = pytest.mark.saving_loading


TEST_ENV = jym.make("CartPole-v1")


@pytest.mark.parametrize("test_alg_cls", get_valid_test_algs(TEST_ENV))
def test_equinox_style_serialisation(tmp_path, test_alg_cls: type[jxalgs.RLAlgorithm]):
    """https://docs.kidger.site/equinox/examples/serialisation/
    Default serialization via equinox (no `jaxon`). Still tested to work,
    requires rebuilding the skeleton.
    """
    env = TEST_ENV
    hyperparams = {"gamma": 0.5}  # only the JSON-able ones; the rest live in the code
    agent = test_alg_cls(**hyperparams).init_agent(jax.random.PRNGKey(1), env)  # type: ignore

    save_path = tmp_path / "test_saving_loading.eqx"
    with open(save_path, "wb") as f:
        f.write((json.dumps(hyperparams) + "\n").encode())
        eqx.tree_serialise_leaves(f, agent)

    with open(save_path, "rb") as f:
        loaded_hyperparams = json.loads(f.readline().decode())
        skeleton = test_alg_cls(**loaded_hyperparams).init_agent(  # type: ignore
            jax.random.PRNGKey(42), env
        )
        loaded = eqx.tree_deserialise_leaves(f, skeleton)

    assert loaded_hyperparams == hyperparams
    assert loaded.trainer.gamma == 0.5
    assert jnp.all(
        jym.tree.get_first(agent, "weight") == jym.tree.get_first(loaded, "weight")
    ), "Weights do not match after loading."

    loaded.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=2)
    loaded.train(jax.random.PRNGKey(2), env)

    os.remove(save_path)


@pytest.mark.parametrize("test_alg_cls", get_valid_test_algs(TEST_ENV))
def test_loading_without_a_skeleton(tmp_path, test_alg_cls: type[jxalgs.RLAlgorithm]):
    """A checkpoint reconstructs the agent outright -- nothing has to exist beforehand.

    `load_state` needs an agent to load into; `load_agent` does not.
    """
    env = TEST_ENV
    agent = test_alg_cls(gamma=0.5).init_agent(jax.random.PRNGKey(1), env)  # type: ignore
    save_path = tmp_path / "cold.jaxon"
    agent.save(save_path)

    cold = jxalgs.RLAgent.load(save_path)

    assert type(cold) is type(agent)
    assert cold.trainer.gamma == 0.5, "hyperparameters were not restored"
    assert jnp.all(
        jym.tree.get_first(cold, "weight") == jym.tree.get_first(agent, "weight")
    )
    # both of these work without ever building an agent from the config
    cold.evaluate(jax.random.PRNGKey(2), env, num_eval_episodes=2)
    cold.train(jax.random.PRNGKey(3), env)

    # Should also work through RLAlgorithm.load:
    cold = jxalgs.RLAlgorithm.load(save_path)

    assert type(cold) is type(agent)
    assert cold.trainer.gamma == 0.5, "hyperparameters were not restored"
    assert jnp.all(
        jym.tree.get_first(cold, "weight") == jym.tree.get_first(agent, "weight")
    )
    # both of these work without ever building an agent from the config
    cold.evaluate(jax.random.PRNGKey(2), env, num_eval_episodes=2)
    cold.train(jax.random.PRNGKey(3), env)


@pytest.mark.parametrize("test_alg_cls", get_valid_test_algs(TEST_ENV))
def test_loaded_agent_can_resume_training(
    tmp_path, test_alg_cls: type[jxalgs.RLAlgorithm]
):
    """Resuming exercises the optimizer state, which is the part of the checkpoint that
    has to stay structurally consistent with the parameters."""
    env = TEST_ENV
    agent = test_alg_cls().init_agent(jax.random.PRNGKey(1), env)  # type: ignore
    save_path = tmp_path / "resume.jaxon"
    agent.save(save_path)

    loaded = jxalgs.RLAgent.load(save_path)
    before = jym.tree.get_first(loaded, "weight")
    trained, _ = loaded.train(jax.random.PRNGKey(2), env)

    assert not jnp.allclose(before, jym.tree.get_first(trained, "weight")), (
        "resumed training changed nothing"
    )


@pytest.mark.parametrize("test_alg_cls", get_valid_test_algs(TEST_ENV))
def test_cloudpickle_saving(tmp_path, test_alg_cls: type[jxalgs.RLAlgorithm]):
    # Create a simple environment
    env = TEST_ENV

    agent = test_alg_cls()  # type: ignore
    agent = agent.init_agent(jax.random.PRNGKey(1), env)

    save_path = tmp_path / "test_cloudpickle_saving.pkl"
    with open(save_path, "wb") as f:
        cloudpickle.dump(agent, f)

    # Load the agent
    with open(save_path, "rb") as f:
        load_agent = cloudpickle.load(f)

    agent_weight = jym.tree.get_first(agent, "weight")
    load_agent_weight = jym.tree.get_first(load_agent, "weight")
    assert jnp.all(agent_weight == load_agent_weight), (
        "Weights do not match after loading."
    )

    # Check if the loaded agent can still train
    # load_agent.train(jax.random.PRNGKey(1), env)
    load_agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=2)

    # Remove the saved file after the test
    os.remove(save_path)
