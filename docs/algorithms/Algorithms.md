# RL Algorithms

Jaxnasium provides a suite of reinforcement learning algorithms. Currently, a small set of algorithms are implemented. More may be added in the future, but the current objective is not to span a wide range of RL algorithms. 

## Algorithm Overview

Jaxnasium algorithms are primarily inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and [PureJaxRL](https://github.com/luchris429/purejaxrl) and therefore follow a near-single-file implementation philosophy. However, Jaxnasium algorithms are built in Equinox and follow a class-based design with a familiar [Stable-Baselines](https://github.com/DLR-RM/stable-baselines3)-like API.

Each algorithm is split into two objects:

- An **algorithm** (or **trainer**) (e.g. `PPO`), subclassing [`RLAlgorithm`][jaxnasium.algorithms.RLAlgorithm].
  It holds the hyperparameters and the training logic.
- An **agent** (e.g. `PPOAgent`), subclassing [`RLAgent`][jaxnasium.algorithms.RLAgent].
  It holds the trainable state (networks, optimizer state, normalizer) along with the
  trainer it was built from.

`trainer.train(key, env)` returns the trained agent and its training metrics. Since the agent
keeps a reference to its trainer, `agent.train(...)` and `agent.evaluate(...)` work directly.

## Available Algorithms

--8<-- "algorithms/_Algorithm-Table.md"

### Key Features Across All Algorithms

- **Automatic Multi-Agent Support**: All algorithms automatically transform to handle multi-agent environments
- **Flexible Action Spaces**: Support for discrete, continuous, and mixed action spaces. Algorithms deal with any composite (pytree of) spaces.
- **PureJaxRL Training**: Training logic, when used with a JIT-compatible environment, is fully JIT-compatible, allowing for extremely fast end-to-end training in JAX.
- **Modular Design**: Near-single-file implementations for easy understanding and modification
- **Configurable networks**: Swap in a different [architecture](networks/Architectures.md) without touching the algorithm.
- **Built-in Normalization**: Optional observation and reward normalization, checkpointed with the agent.
- **Checkpointing**: `agent.save(path)` / `RLAgent.load(path)`, trainer included.
- **Logging**: Optional logging during training built-in.

### Action and Observation Space Support Details

All algorithms in Jaxnasium support composite observation and action spaces through PyTree structures. When observation or action spaces are defined as PyTrees of spaces (e.g., dictionaries, tuples, or nested combinations), the algorithms automatically handle the structured data flow. The neural networks are designed to process PyTree inputs and outputs seamlessly. This design allows algorithms to work with complex environments without requiring manual preprocessing or postprocessing of the data.

### Multi-Agent Capabilities

Every algorithm in Jaxnasium includes automatic multi-agent support through function transformations. When you provide a multi-agent environment, the algorithm automatically:

- Builds one agent per entry of the first level of the environment's space PyTree, wrapped in a
  [`MultiAgentWrapper`][jaxnasium.algorithms.core.MultiAgentWrapper].
- Maps per-agent methods over that structure, so an algorithm can be written as a single-agent algorithm and handle multi-agent scenarios seamlessly.
- Supports homogeneous and heterogeneous agent scenarios. Homogeneous agents run in parallel under `jax.vmap`.

For more information, see the [Multi-Agent](Multi-Agent.md) documentation.

## Getting Started

Each algorithm follows a consistent interface:

```python
import jax
import jaxnasium as jym
from jaxnasium.algorithms import PPO

env = jym.make("CartPole-v1")
key = jax.random.PRNGKey(0)

# Configure the hyperparameters
algorithm = PPO(learning_rate_start=3e-4, total_timesteps=1_000_000, num_envs=8)

# Train, returning the trained agent and its per-iteration training metrics
agent, metrics = algorithm.train(key, env)

# Evaluate: jnp.array of shape (num_eval_episodes,)
returns = agent.evaluate(key, env, num_eval_episodes=10)

# Continue training, optionally with different hyperparameters
agent, more_metrics = agent.train(key, env, total_timesteps=100_000)
```

By default `metrics` is the mean return of the episodes that finished in each training
iteration, one scalar per iteration; iterations in which no episode finished yield `NaN`.
Pass `reduce_metrics_fn=None` to receive the full batch of metrics instead.

The algorithms are designed to work seamlessly with any Jaxnasium environment, automatically adapting to the environment's observation and action spaces, and scaling to multi-agent scenarios when needed.

## Customizing networks

Each algorithm exposes `actor_kwargs` / `critic_kwargs`, which are forwarded to the
[agent networks](networks/Networks.md). This is the entry point for using a different body
[architecture](networks/Architectures.md) or different output layers:

```python
from jaxnasium.algorithms import PPO
from jaxnasium.algorithms.architectures import BroNet

algorithm = PPO(critic_kwargs={"body": BroNet.with_params(depth=2, width_size=256)})
```
