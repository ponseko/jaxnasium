# Compilation

End-to-end JAX training loops are compiled as one big program, which can take a while.
Jaxnasium provides two convenience helpers around JAX's own facilities: one to compile
ahead of time (with progress reporting), and one to persist compilation results across
processes.

```python
import jax
import jaxnasium as jym
from jaxnasium.algorithms import PPO

jym.enable_compilation_cache()

env = jym.make("CartPole-v1")
train = jym.precompile(PPO().train, jax.random.PRNGKey(0), env)
agent, metrics = train()
```

::: jaxnasium.precompile

::: jaxnasium.enable_compilation_cache
