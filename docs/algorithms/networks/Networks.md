# Agent Networks

The networks the algorithms build. Each consists of three components:

1. An **observation processor** ([`PyTreeObsSpaceNetwork`](Layers.md)), which builds a
   network per observation space in the (possibly nested) observation PyTree. 1D spaces go
   through `architecture_1d`, 2D spaces through `architecture_2d`, and the outputs are
   concatenated into a single 1D vector.
2. A **body** (default [`SimBa`](Architectures.md)), a shared network processing that vector.
3. An **output processor** ([`PyTreeActionNetwork` / `PyTreeQValueNetwork`](Layers.md)),
   which builds a head per output space, choosing a discrete or continuous head based on
   the space, and returns results in the same PyTree structure as the action space.

Constructor arguments are reachable through an algorithm's `actor_kwargs` / `critic_kwargs`:

```python
from jaxnasium.algorithms import PPO
from jaxnasium.algorithms.architectures import MLP

algorithm = PPO(
    actor_kwargs={"body": MLP.with_params(hidden_sizes=(128, 128, 128))},
    critic_kwargs={"body": MLP.with_params(hidden_sizes=(128, 128, 128))},
)
```

::: jaxnasium.algorithms.ActorNetwork
    options:
        members:
            - __init__
            - __call__

::: jaxnasium.algorithms.ValueNetwork
    options:
        members:
            - __init__
            - __call__

::: jaxnasium.algorithms.QValueNetwork
    options:
        members:
            - __init__
            - __call__

`AdvantageNetwork` is an alias of `QValueNetwork`.

# Initialization

::: jaxnasium.algorithms.core.set_weight_bias