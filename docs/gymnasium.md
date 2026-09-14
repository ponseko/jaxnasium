# Using Gymnasium environments

Jaxnasium's algorithms are written for end-to-end RL training in pure JAX ([PureJaxRL](https://github.com/luchris429/purejaxrl)). However, this requires environment step and reset functions to be written in JAX and be fully JIT-compatible. While this can greatly improve training times, many environments are not (yet) ported over to JAX. 

Jaxnasium provides a `GymnasiumWrapper` to use gymnasium environments within its training pipeline. It calls back into Python on every step via `jax.experimental.io_callback`, so the rest of the training loop stays compiled. Each step is then a Python roundtrip, so throughput will naturally be lower compared to native JAX environments, but this wrapper opens the door for many more environments to be used with the same algorithms.

```python
import gymnasium as gym
import jax
from gymnasium.vector import AutoresetMode

from jaxnasium import GymnasiumWrapper
from jaxnasium.algorithms import PPO

venv = gym.make_vec(
    "CartPole-v1",
    num_envs=2,
    vectorization_mode="sync",
    vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
)
env = GymnasiumWrapper(venv)

ppo = PPO(num_envs=2, total_timesteps=100_000)
agent, metrics = ppo.train(jax.random.key(0), env)

eval_env = GymnasiumWrapper(gym.make("CartPole-v1"))
returns = agent.evaluate(jax.random.key(1), eval_env, num_eval_episodes=10)
```

## Training

During training, Jaxnasium algorithms expect a leading `num_envs` batch axis, typically obtained from
[`VecEnvWrapper`](api/Wrappers.md), which `jax.vmap`s the environment step and reset. Gymnasium environments
cannot rely on this wrapper, so have to do vectorization on the Gymnasium side, e.g. via [`gymnasium.make_vec`](https://gymnasium.farama.org/api/vector/):

```python
venv = gym.make_vec(
    "CartPole-v1",
    num_envs=2,
    vectorization_mode="sync",
    vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
)
env = GymnasiumWrapper(venv)

ppo = PPO(num_envs=2, total_timesteps=100_000)
agent, metrics = ppo.train(jax.random.key(0), env)
```

!!! notes
    - **`autoreset_mode=AutoresetMode.SAME_STEP` is preferred.** as it matches the Jaxnasium API.
      With Gymnasium's default mode (`NEXT_STEP`) or `DISABLED`, the `GymnasiumWrapper` resets finished 
      environments itself within the same step, which costs an extra reset call on every step in which an episode ends.
    - **An explicit `vectorization_mode`.** Some environments (e.g. `CartPole-v1`) ship
      their own vectorized implementation, which `make_vec` picks by default. Not all of these implementations
      allow individual environments to be reset manually, so the wrapper rejects it; set to `"sync"` (or
      `"async"`) instead.
    - **The algorithm's `num_envs` must equal the vector environment's.** The batch axis now comes
      from the environment, so training raises an error if they differ, or if the environment is
      not vectorized at all.

## Evaluation

`algorithm.evaluate()` runs one episode at a time (non-vectorized). A `VecEnvWrapper` is typically
removed automatically, but this is not the case for Gymnasium environments. As such, in contrast to training,
pass a plain Gymnasium environment to the `GymnasiumWrapper` instead.

```python
eval_env = GymnasiumWrapper(gym.make("CartPole-v1"))
returns = agent.evaluate(jax.random.key(1), eval_env, num_eval_episodes=10)
```

## EnvPool

Since [EnvPool](https://github.com/sail-sg/envpool) can create environments with the Gymnasium standard, the `GymnasiumWrapper` can trivially be used to speed up training times on EnvPool supported environments:

```python
import envpool

venv = envpool.make_gymnasium("CartPole-v1", num_envs=2, seed=0)
env = GymnasiumWrapper(venv)
```

EnvPool resets finished environments on the next step, so the wrapper resets them itself, as
described above. Seed the environments when creating them: EnvPool does not
take per-reset seeds into account, so runs are not seeded by the key passed to `reset`.


::: jaxnasium.GymnasiumWrapper
    options:
        members:
            - step
            - reset


## Limitations

!!! warning
    - **Info dictionaries are dropped**, since a callback must return a fixed structure of fixed
      shapes. Only `_TERMINAL_OBSERVATION` is provided.
    - **Only flat spaces** (`Box`, `Discrete` and `MultiDiscrete`) are supported.
    - **Training runs cannot be vmapped.** Running several seeds or hyperparameters at once with
      `jax.vmap` would make every copy step the same Python environments, so it raises an error.
    - Only a small subset of environments that adhere to the Gymnasium API have been tested.
