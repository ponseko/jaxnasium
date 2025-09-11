import os

import _consts as TEST_CONSTS
import cloudpickle
import jax
import jax.numpy as jnp

import jaxnasium
import jaxnasium.algorithms


def test_saving_loading(tmp_path):
    # Create a simple environment
    env = jaxnasium.make("CartPole-v1")

    # Initialize the agent
    agent = jaxnasium.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)

    # # Train the agent
    # agent = agent.train(jax.random.PRNGKey(1), env)
    agent = agent.init_state(jax.random.PRNGKey(1), env)

    save_path = tmp_path / "test_saving_loading.eqx."
    agent.save_state(save_path)

    # Load the agent
    load_agent = jaxnasium.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)
    load_agent = load_agent.init_state(jax.random.PRNGKey(1), env)
    load_agent = load_agent.load_state(save_path)

    # Check if weights match (via some arbitary layer)
    assert jnp.all(
        agent.state.actor.mlp.layers[0].weight
        == load_agent.state.actor.mlp.layers[0].weight
    ), "Weights do not match after loading."
    assert jnp.all(
        agent.state.critic.mlp.layers[1].weight
        == load_agent.state.critic.mlp.layers[1].weight
    ), "Weights do not match after loading."

    # Check if the loaded agent can still train
    load_agent.train(jax.random.PRNGKey(1), env)
    load_agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=10)

    # Remove the saved file after the test
    os.remove(save_path)


def test_cloudpickle_saving(tmp_path):
    # Create a simple environment
    env = jaxnasium.make("CartPole-v1")

    # Initialize the agent
    agent = jaxnasium.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)

    # # Train the agent
    # agent = agent.train(jax.random.PRNGKey(1), env)
    agent = agent.init_state(jax.random.PRNGKey(1), env)

    save_path = tmp_path / "test_cloudpickle_saving.pkl"
    with open(save_path, "wb") as f:
        cloudpickle.dump(agent, f)

    # Load the agent
    with open(save_path, "rb") as f:
        load_agent: jaxnasium.algorithms.PPO = cloudpickle.load(f)

    # Check if weights match
    assert jnp.all(
        agent.state.actor.mlp.layers[0].weight
        == load_agent.state.actor.mlp.layers[0].weight
    ), "Weights do not match after loading."
    assert jnp.all(
        agent.state.critic.mlp.layers[1].weight
        == load_agent.state.critic.mlp.layers[1].weight
    ), "Weights do not match after loading."

    # Check if the loaded agent can still train
    load_agent.train(jax.random.PRNGKey(1), env)
    load_agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=10)

    # Remove the saved file after the test
    os.remove(save_path)
