# Multi-Agent RL

Multi-agent reinforcement learning is a **core pillar** of Jaxnasium's design philosophy. Rather than maintaining separate APIs for single-agent and multi-agent settings, Jaxnasium unifies both paradigms through **PyTrees** and function transformations.

## Expected PyTree Structures

Jaxnasium's Multi-Agent design is heavily based on JAX's PyTrees. [PyTrees](https://docs.jax.dev/en/latest/pytrees.html) allow us to (nested) data structures (like dictionaries, lists, tuples) in a way that JAX can efficiently process and transform. This becomes the perfect abstraction for multi-agent scenarios where we need to handle:

- **Observations**: `{"agent_0": obs_0, "agent_1": obs_1, ...}`
- **Actions**: `{"agent_0": action_0, "agent_1": action_1, ...}`
- **Rewards**: `{"agent_0": reward_0, "agent_1": reward_1, ...}`

Essentially, Jaxnasium enviroments with `multi_agent=True` are expected to have an `action_space` and `observation_space` that are PyTrees of spaces. The **first level** of the PyTree is the agent dimension. Similarly, the reward function and, optionally, the 
termination and truncation flag should also return PyTrees with the same first-level structure. This is similar to the API already set out by [JaxMARL](https://github.com/FLAIROx/JaxMARL), but Jaxnasium allows any PyTree structure of agents.

All elements below the first level of the PyTree can be arbitrary structures, including more nested PyTrees.

```python
# Homogeneous agents
env.action_space = {
    "agent_0": Discrete(2), "agent_1": Discrete(2)
}
env.observation_space = {
    "agent_0": Box(low=0, high=1, shape=(3,)), "agent_1": Box(low=0, high=1, shape=(3,))
}
reward = {
    "agent_0": -1, "agent_1": 1
}

# Heterogeneous agents
env.action_space = {
    "agent_0": Discrete(2), "agent_1": MultiDiscrete(2, 3)
}
env.observation_space = {
    "agent_0": Box(low=0, high=1, shape=(3,)), "agent_1": Box(low=0, high=1, shape=(8,))
}
reward = {
    "agent_0": -1, "agent_1": 1
}

# Heterogeneous agents in a list with nested PyTree actions
env.action_space = [
    {"position": Discrete(2), "velocity": Discrete(2)}, 
    {"action": MultiDiscrete(2, 3)}
]
env.observation_space = [
    {"xy": Box(low=0, high=1, shape=(2,)), "velocity": Box(low=0, high=1, shape=(1,))},
    Discrete(3)
]
reward = [
    -1, 1
]
```

!!! note "Enforcement"
    Jaxnasium Environments do not enforce this Multi-Agent structure. It is however recommended, and expected by the Jaxnasium algorithms.

### Why this is useful

The core idea here is that we can write single-agent algorithms that can easily transition to multi-agent settings via 
Jax's built-in PyTree operations.


## The `transform_multi_agent` Decorator

The `transform_multi_agent` decorator is the magic that makes single-agent algorithms automatically work with multi-agent environments.

### Core Mechanism

```python
@transform_multi_agent
def get_action(key, agent_state, observation):
    action_dist = agent_state.actor(observation)
    return action_dist.sample(seed=key)
```

When this function is called with multi-agent data:

```python
# Multi-agent inputs
agent_states = {"agent_0": state_0, "agent_1": state_1}
observations = {"agent_0": obs_0, "agent_1": obs_1}
key = jax.random.PRNGKey(42) # Key is (optionally) automatically split over the agents.

# The decorator automatically handles the transformation
actions = get_action(key, agent_states, observations)
# Result: {"agent_0": action_0, "agent_1": action_1}
```

- **Argument Structure**: The first argument of the function is assumed to have first-level PyTree structure of agents.
The remaining arguments that are not provided in `shared_argnames` are assumed to have the same first-level PyTree structure.
functions in `shared_argnames` will be shared across agents.

- **Key Splitting**: Optionally, PRNG keys can be provided as a single key, and will automatically be split accross the first-level PyTree structure of the first argument.

- **Automatic Shared Arguments Detection**: Optionally, rather than explicitly providing the `shared_argnames` argument, the decorator can automatically detect shared arguments based on the function signature. Arguments that do not have the same first-level PyTree structure as the first argument are assumed to be shared.

- **Homogeneous Agents**: Uses `jax.vmap` for maximum efficiency when all agents have identical structures
- **Heterogeneous Agents**: Uses `jax.tree.map` for flexible handling of different agent types

- **Automatic Transposition**: For `Transition` objects (replay buffer data), the decorator automatically transposes the data structure to be compatible with the function signature.

## The `__make_multi_agent__` Method

The `RLAlgorithm.__make_multi_agent__` method is the bridge that connects single-agent algorithms to multi-agent environments. It will apply the `transform_multi_agent` decorator to specified methods and return a new instance of the algorithm that is in multi-agent mode. By default, the transformed methods are:

- `get_action`
- `get_value`
- `_update_agent_state`
- `_make_agent_state`
- `_postprocess_rollout`

### Automatic Upgrade Process

When an algorithm encounters a multi-agent environment:

```python
def init_state(self, key: PRNGKeyArray, env: Environment) -> "PPO":
    if getattr(env, "multi_agent", False) and self.auto_upgrade_multi_agent:
        self = self.__make_multi_agent__()  # Automatic upgrade!
```

## PyTree Operations

The following functions are commonly used to handle multi-agent data:

### `map_one_level`
Maps a function over the first level of a PyTree structure:
```python
# Applies function to each agent's data
result = map_one_level(agent_function, agent_data)
```

### `stack` and `unstack`
Efficiently converts between agent-wise and batch-wise representations:
```python
# Convert agent-wise to batch-wise for vmap
stacked = stack(agent_data)  # {"agent_0": data_0, "agent_1": data_1} -> batched_data
result = jax.vmap(function)(stacked)
# Convert back to agent-wise
unstacked = unstack(result, structure=original_structure)
```

# Documentation

::: src.jaxnasium.algorithms.utils.transform_multi_agent
    options: 
        members: 
            -