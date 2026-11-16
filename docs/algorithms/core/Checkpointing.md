# Checkpointing

Agents are saved with their trainer, so a checkpoint contains everything needed to keep training or to evaluate:

```python
agent.save("ppo_cartpole.eqx")

from jaxnasium.algorithms import RLAgent

agent = RLAgent.load("ppo_cartpole.eqx")
agent, metrics = agent.train(key, env)
```

Agents are saved with [`jaxon`](https://github.com/42Ar/jaxon). Alternatively, they are
plain Equinox modules and can be serialized
[like any other](https://docs.kidger.site/equinox/examples/serialisation/).

## During training

To save the agent while it trains, add a callback.

```python
import equinox as eqx
import jax

from jaxnasium.algorithms import PPO
from jaxnasium.algorithms.core import Callback, print_returns


def save_checkpoint(key, runner_state, metrics, iteration, num_iterations):
    agent = runner_state[0]
    # Only arrays can be sent to the host; the rest (e.g. functions) is closed over
    arrays, static = eqx.partition(agent, eqx.is_array)

    def save(arrays, iteration):
        agent = eqx.combine(arrays, static)
        # removing callbacks since they may not be loadable elsewhere
        agent = agent.with_hyperparams(callbacks=())
        agent.save(f"ppo_{int(iteration)}.jaxon")

    jax.debug.callback(save, arrays, iteration)


trainer = PPO(callbacks=(Callback(print_returns), Callback(save_checkpoint, interval=0.1)))
agent, metrics = trainer.train(key, env)  # saves every 10% of training, and at the end
```

::: jaxnasium.algorithms.core.save_agent

::: jaxnasium.algorithms.core.load_agent
