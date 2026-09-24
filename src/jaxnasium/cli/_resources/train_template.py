import jax
from jaxtyping import PRNGKeyArray

import jaxnasium as jym
from jaxnasium.algorithms import PPO as Algo

# jym.enable_compilation_cache() # optional


def do_random_evaluation(
    key: PRNGKeyArray, env: jym.Environment, num_repitions: int = 10
):
    """Perform some random steps to set a baseline for the environment."""
    rewards = 0.0
    for _ in range(num_repitions):
        reset_key, key = jax.random.split(key)
        obs, env_state = env.reset(reset_key)
        while True:
            sample_key, step_key, key = jax.random.split(key)
            action = env.action_space.sample(sample_key)
            (obs, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )
            rewards += reward
            if terminated or truncated:
                break
    return rewards / num_repitions


if __name__ == "__main__":
    env = jym.make("CartPole-v1")
    env = jym.LogWrapper(env)
    rng = jax.random.PRNGKey(0)

    random_rewards = do_random_evaluation(rng, env)
    print(f"Random Agent average reward: {random_rewards}")

    # RL Training
    agent = Algo(total_timesteps=100000)
    train = jym.precompile(agent.train, rng, env)
    agent, metrics = train()

    print(f"Agent average reward: {agent.evaluate(rng, env)}")

    # Changing network architecture:
    # from jaxnasium.algorithms.architectures import BroNet
    # agent = Algo(critic_kwargs={"body": BroNet.with_params(depth=2, width_size=256)})
    #
