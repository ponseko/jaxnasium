# Multi-Agent RL

Multi-agent reinforcement learning is a **core pillar** of Jaxnasium's design philosophy. Rather than maintaining separate APIs for single-agent and multi-agent settings, Jaxnasium unifies both paradigms through **PyTrees** and function transformations.

## Expected PyTree Structures

Jaxnasium's multi-agent design is heavily based on JAX's [PyTrees](https://docs.jax.dev/en/latest/pytrees.html), which let us express nested data structures (dictionaries, lists, tuples) in a way that JAX can efficiently process and transform. This becomes the perfect abstraction for multi-agent scenarios where we need to handle:

- **Observations**: `{"agent_0": obs_0, "agent_1": obs_1, ...}`
- **Actions**: `{"agent_0": action_0, "agent_1": action_1, ...}`
- **Rewards**: `{"agent_0": reward_0, "agent_1": reward_1, ...}`

Essentially, Jaxnasium environments with `multi_agent=True` are expected to have an `action_space` and `observation_space` that are PyTrees of spaces. The **first level** of the PyTree is the agent dimension. Similarly, the reward function and, optionally, the
termination and truncation flag should also return PyTrees with the same first-level structure. This is similar to the API already set out by [JaxMARL](https://github.com/FLAIROx/JaxMARL), but Jaxnasium allows any PyTree structure of agents.

All elements below the first level of the PyTree can be arbitrary structures, including more nested PyTrees.

```python
# Homogeneous agents
env.action_space = {"agent_0": Discrete(2), "agent_1": Discrete(2)}
env.observation_space = {
    "agent_0": Box(low=0, high=1, shape=(3,)),
    "agent_1": Box(low=0, high=1, shape=(3,)),
}
reward = {"agent_0": -1, "agent_1": 1}

# Heterogeneous agents
env.action_space = {"agent_0": Discrete(2), "agent_1": MultiDiscrete(2, 3)}
env.observation_space = {
    "agent_0": Box(low=0, high=1, shape=(3,)),
    "agent_1": Box(low=0, high=1, shape=(8,)),
}
reward = {"agent_0": -1, "agent_1": 1}

# Heterogeneous agents in a list with nested PyTree actions
env.action_space = [
    {"position": Discrete(2), "velocity": Discrete(2)},
    {"action": MultiDiscrete(2, 3)},
]
env.observation_space = [
    {"xy": Box(low=0, high=1, shape=(2,)), "velocity": Box(low=0, high=1, shape=(1,))},
    Discrete(3),
]
reward = [-1, 1]
```

!!! note "Enforcement"
    Jaxnasium Environments do not enforce this Multi-Agent structure. It is however recommended, and expected by the Jaxnasium algorithms.

### Why this is useful

The core idea here is that we can write single-agent algorithms that transition to
multi-agent settings via JAX's built-in PyTree operations.

## Automatic upgrade

Algorithms need no multi-agent-specific code. When an [`RLAgent`][jaxnasium.algorithms.RLAgent]
is constructed for an environment with `multi_agent=True` (and the trainer's
`auto_upgrade_multi_agent` is left at `True`), Jaxnasium instead builds *one agent per agent
in the environment* and returns them wrapped in a
[`MultiAgentWrapper`][jaxnasium.algorithms.core.MultiAgentWrapper]:

```python
import jaxnasium as jym
from jaxnasium.algorithms import DQN

env = jym.make("MPE_simple_spread_v3")  # multi-agent
agent, metrics = DQN().train(key, env)  # a MultiAgentWrapper of DQNAgents
```

Each per-agent trainer sees a single-agent view of the environment, holding just that agent's
observation and action space. Hyperparameters may be given per agent by passing a PyTree with
the agent structure, in which case each agent's trainer receives its own value:

```python
DQN(gamma={"agent_0": 0.9, "agent_1": 0.99})
```

Calls on the wrapper are then dispatched to every agent, and the results are returned in the
same structure. Methods marked with
[`@collective`][jaxnasium.algorithms._algorithm.collective] (such as `train`, `evaluate`, and
`save`) instead run once on the whole team; everything else is
[`@per_agent`][jaxnasium.algorithms._algorithm.per_agent] by default and is written as
single-agent code.

::: jaxnasium.algorithms.core.MultiAgentWrapper
    options:
        heading_level: 3
        members:
            - agents
            - trainer
            - with_hyperparams

## Mapping a function over agents

`map_multi_agent` is the transformation underlying all of the above. It applies a
single-agent function over the first level of each per-agent argument.

```python
from jaxnasium.algorithms.core._multi_agent import map_multi_agent


def get_action(agent, key, observation):
    return agent.get_action(key, observation)


agents = {"agent_0": agent_0, "agent_1": agent_1}
observations = {"agent_0": obs_0, "agent_1": obs_1}

# The agent structure is taken from `agents`, and the key is split over it automatically.
actions = map_multi_agent(get_action, agents, key, observations)
# {"agent_0": action_0, "agent_1": action_1}
```

Notable behaviours:

- **Agent structure**: inferred from the first non-key argument, or given explicitly via `agent_structure`.
- **Shared arguments**: arguments whose first level does not match the agent structure are broadcast to every agent.
- **Key splitting**: a single PRNG key is split across the agent structure.
- **Homogeneous agents**: mapped with `jax.vmap` for efficiency. Set the environment variable `JAXNASIUM_MULTI_AGENT_BATCH_SIZE` to cap how many agents are vmapped at once.
- **Heterogeneous agents**: mapped with `jax.tree.map`.
- **Containers**: [`Transition`][jaxnasium.algorithms.core.Transition] batches are transposed to a per-agent view and merged back afterwards, and `MultiAgentWrapper`s are unwrapped and re-wrapped.

::: jaxnasium.algorithms.core._multi_agent.map_multi_agent
    options:
        heading_level: 3

## PyTree helpers

The [`jaxnasium.tree`](../tree/Tree.md) module contains the pytree operations used to handle
multi-agent data, most notably
[`map_one_level`][jaxnasium.tree.map_one_level],
[`stack`][jaxnasium.tree.stack] and
[`unstack`][jaxnasium.tree.unstack].
