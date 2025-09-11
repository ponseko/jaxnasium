import jax
from jaxtyping import PRNGKeyArray

import jaxnasium as jym
from jaxnasium.algorithms import PPO

# from <PROJECTNAME> import ExampleEnv


def do_random_evaluation(
    key: PRNGKeyArray, env: jym.Environment, num_repitions: int = 10
):
    """Perform some random steps to set a baseline for the environment."""
    rewards = 0.0
    for _ in range(num_repitions):
        obs, env_state = env.reset(key)
        while True:
            key, key = jax.random.split(key)
            action = env.action_space.sample(key)
            (obs, reward, terminated, truncated, info), env_state = env.step(
                key, env_state, action
            )
            rewards += reward
            if terminated or truncated:
                break
    return rewards / num_repitions


if __name__ == "__main__":
    env = ExampleEnv()  # noqa: F821 # type: ignore[reportUndefinedVariable]
    env = jym.LogWrapper(env)
    rng = jax.random.PRNGKey(0)

    random_rewards = do_random_evaluation(rng, env)
    print(f"Random Agent average reward: {random_rewards}")

    # RL Training with PPO
    agent = PPO(
        total_timesteps=50000,
        num_steps=64,
        learning_rate=2.5e-3,
        ent_coef=0.0,
        num_epochs=1,
    )
    agent = agent.train(rng, env)
