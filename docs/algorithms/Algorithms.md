# RL Algorithms

Jaxnasium provides a suite of reinforcement learning algorithms. Currently, a small set of algorithms are implemented. More may be added in the future, but the current objective is not to span a wide range of RL algorithms. 

## Algorithm Overview

Jaxnasium algorithms are primarily inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and [PureJaxRL](https://github.com/luchris429/purejaxrl) and therefor follow a near-single-file implementation philosophy. However, Jaxnasium algorithms are built in Equinox and follow a class-based design with a familiar [Stable-Baselines](https://github.com/DLR-RM/stable-baselines3) API. 

All algorithms inherit from the [`RLAlgorithm`](https://github.com/ponseko/jymkit/blob/main/src/jymkit/algorithms/_algorithm.py) abstract base class, which primarily defines a minimal common interface for all algorithms, contains a standard evaluation loop and handles multi-agent support. All training logic is implemented in the algorithms themselves.

## Available Algorithms

--8<-- "algorithms/_Algorithm-Table.md"

### Key Features Across All Algorithms

- **Automatic Multi-Agent Support**: All algorithms automatically transform to handle multi-agent environments
- **Flexible Action Spaces**: Support for discrete, continuous, and mixed action spaces. Algorihms deal with any composite (pytree of) spaces.
- **PureJaxRL Training**: Training logic, when used with a JIT-compatible environment, is fully JIT-compatible, allowing for extremely fast end-to-end training in JAX.
- **Modular Design**: Near-single-file implementations for easy understanding and modification
- **Built-in Normalization**: Optional observation and reward normalization.
- **Logging**: Optional logging during training built-in.

### Action and Observation Space Support Details

All algorithms in Jaxnasium support composite observation and action spaces through PyTree structures. When observation or action spaces are defined as PyTrees of spaces (e.g., dictionaries, tuples, or nested combinations), the algorithms automatically handle the structured data flow. The neural networks are designed to process PyTree inputs and outputs seamlessly.This design allows algorithms to work with complex environments without requiring manual preprocessing or postprocessing of the data. 

### Multi-Agent Capabilities

Every algorithm in Jaxnasium includes automatic multi-agent support through function transformations. When you provide a multi-agent environment, the algorithm automatically:

- Transforms some methods to act on the first level of the PyTree structure of the inputs. Thereby performing per-agent operations.
- Algorithms can be designed as a single-agent algorithm, and handle multi-agent scenarios seamlessly.
- Homogeneous and heterogeneous agent scenarios are supported. Homogeneous agent operations may run in parallel. 

For more information, see the [Multi-Agent](../Multi-Agent/) documentation.

## Getting Started

Each algorithm follows a consistent interface:

```python
import jaxnasium as jym
from jaxnasium.algorithms import PPO

# Create algorithm instance
algorithm = PPO(
    learning_rate=3e-4,
    total_timesteps=1_000_000,
    num_envs=8
)

# Train on environment
trained_algorithm = algorithm.train(key, env)

# Evaluate
rewards = trained_algorithm.evaluate(key, env, num_episodes=10) # jnp.array of shape (num_episodes,)
```

The algorithms are designed to work seamlessly with any Jaxnasium environment, automatically adapting to the environment's observation and action spaces, and scaling to multi-agent scenarios when needed.
