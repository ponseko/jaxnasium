# Checkpointing

Agents are saved with their trainer, so a checkpoint contains everything needed to keep training or to evaluate:

```python
agent.save("ppo_cartpole.eqx")

from jaxnasium.algorithms import RLAgent

agent = RLAgent.load("ppo_cartpole.eqx")
agent, metrics = agent.train(key, env)
```

This requires [`jaxon`](https://pypi.org/project/jaxon/), which is not installed by default
(`pip install jaxon`). Alternatively, agents are plain Equinox modules and can be serialized
[like any other](https://docs.kidger.site/equinox/examples/serialisation/).

::: jaxnasium.algorithms.core.save_agent

::: jaxnasium.algorithms.core.load_agent
