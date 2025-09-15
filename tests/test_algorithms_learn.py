import _consts as TEST_CONSTS
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxnasium as jym
from jaxnasium.algorithms import SAC


@pytest.mark.parametrize("alg", TEST_CONSTS.DISCRETE_ALGS)
def test_discrete_is_learning(alg):
    # Confirm learning behavior on CartPole w/ default parameters
    env = jym.make("CartPole-v1")
    seed = jax.random.PRNGKey(1)
    seed1, seed2 = jax.random.split(seed)
    agent = alg(total_timesteps=1_000_000, log_function=None)
    agent = agent.train(seed1, env)

    rewards = agent.evaluate(seed2, env, num_eval_episodes=50)
    avg_reward = jnp.mean(rewards)
    assert avg_reward > 200, (
        f"Average reward too low: {avg_reward}. Training may have failed."
        f"Average reward: {avg_reward}, "
        f"Rewards array: {rewards}, "
        f"Rewards type: {type(rewards)}, "
        f"Rewards shape: {getattr(rewards, 'shape', 'no shape')}"
    )


@pytest.mark.parametrize("alg", TEST_CONSTS.CONTINUOUS_ALGS)
def test_continuous_is_learning(alg):
    # Confirm learning behavior on Pendulum w/ default parameters
    env = jym.make("Pendulum-v1")
    seed = jax.random.PRNGKey(0)
    seed1, seed2 = jax.random.split(seed)
    if alg == SAC:
        agent = alg(**TEST_CONSTS.SAC_CONTINUOUS_CONFIG)
    else:
        agent = alg(total_timesteps=500_000, log_function=None)
    agent = agent.train(seed1, env)

    rewards = agent.evaluate(seed2, env, num_eval_episodes=50)
    avg_reward = np.mean(rewards)
    assert avg_reward > -400, (
        f"Average reward too low: {avg_reward}. Training may have failed."
    )
