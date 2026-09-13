# Using Flax networks

Jaxnasium's algorithms, and many more components within the library, are written as [Equinox](https://docs.kidger.site/equinox/) modules. It only felt natural to write all the networks within the same ecosystem.

While Equinox is an amazing library definitely worth a try, [Flax](https://github.com/google/flax) has grown to be the bigger neural network library within the JAX ecosystem. Understandably, one may then favor and be more familiar with writing Flax architectures. Luckily, using Flax code within the Equinox ecosystem is not much of a hassle.

Below is an example of a small Equinox Module wrapping an NNX module, that can be used within existing [agent networks](Networks.md) and [algorithms](../Algorithms.md). Similar conversion from the older linen API can also be achieved; [this issue](https://github.com/patrick-kidger/equinox/issues/886#issuecomment-2433115652) is probably a good starting point.

```python
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PRNGKeyArray


class NNXModule(eqx.Module):
    """Exposes a stateless NNX network as an Equinox module."""

    # flax fields
    graphdef: nnx.GraphDef = eqx.field(static=True)
    params: nnx.State

    # custom fields for jaxnasium bookkeeping
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        *,
        key: PRNGKeyArray,
        nnx_module: Callable[..., nnx.Module],
    ):
        model = nnx_module(in_features, rngs=nnx.Rngs(key))
        self.in_features = in_features
        self.graphdef, self.params, rest = nnx.split(model, nnx.Param, ...)
        # We assume rest is empty here (stateless network). Else this example crashes.

        self.out_features = jax.eval_shape(
            lambda p: nnx.merge(self.graphdef, p)(jnp.zeros(in_features)),
            self.params,
        ).shape[-1]

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        return nnx.merge(self.graphdef, self.params)(x)

    @classmethod
    def with_nnx_network(cls, module: Callable[..., nnx.Module]):
        """Convenience function such that we have a constructor with 'in_features, *, key=...' as signature"""
        return eqx.Partial(cls, nnx_module=module)
```

Any NNX module whose constructor takes `in_features` and `rngs` can then be used as a body:

```python
import jaxnasium as jym
from jaxnasium.algorithms import PPO


class FlaxMLP(nnx.Module):
    def __init__(self, in_features, hidden_sizes=(64, 64), *, rngs: nnx.Rngs):
        sizes = (in_features, *hidden_sizes)
        self.layers = nnx.List(
            [nnx.Linear(a, b, rngs=rngs) for a, b in zip(sizes[:-1], sizes[1:])]
        )

    def __call__(self, x):
        for layer in self.layers:
            x = nnx.relu(layer(x))
        return x

body = NNXModule.with_nnx_network(FlaxMLP)

env = jym.make("CartPole-v1")
ppo = PPO(actor_kwargs={"body": body}, critic_kwargs={"body": body})
agent, metrics = ppo.train(jax.random.key(0), env)
```

The same factory works for `obs_architecture_1d` / `obs_architecture_2d`.
Since the parameters are ordinary array leaves, gradients, optimizer updates, and [checkpointing](../core/Checkpointing.md) all work as usual.

!!! note
    - Flax is not a dependency and needs to be installed separately.
    - [`set_weight_bias`](Networks.md#initialization) only re-initializes
      `eqx.nn.Linear` and `eqx.nn.Conv` layers, so NNX layers keep their own initialization.
      You should add your own custom initialization to the nnx or wrapper module itself if desired.
