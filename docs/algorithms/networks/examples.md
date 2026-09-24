# CTDE examples

## Altered critic inputs

[`AgentObservation`](../../api/Environment.md) has an optional `critic_observation`.
`ValueNetwork` / `QValueNetwork` use this when it is not None as their input. This may
then be used to trivially use a different observation for the critic during training.

The environment (or a wrapper) decides what the critic sees:

```python
AGENTS = ("agent_0", "agent_1")


class ExampleEnv(jym.Environment):
    _multi_agent: bool = eqx.field(static=True, default=True)

    @property
    def observation_space(self):
        return {
            a: jym.AgentObservation(
                observation=jym.Box(-1.0, 1.0, (1,)),
                critic_observation=jym.Box(-1.0, 1.0, (2,)),
            )
            for a in AGENTS
        }

    def step_env(self, positions):
        ...
        obs = {
            a: jym.AgentObservation(
                observation=positions[i][None],
                critic_observation=positions,
            )
            for i, a in enumerate(AGENTS)
        }
        return (obs, ...), state

```

As `AgentObservation` is a regular PyTree, an Algorithm' `Normalizer` automatically keeps 
separate running statistics for the two observations.

## Multi-Agent Shared critic parameter example

Below is an example of sharing critic parameters for SAC through a custom MultiAgentWrapper.
Each agent still has their own critic, but only the first is updated and synced to other agents
afterwards.

```python
def pool(batch: Transition) -> Transition:
    """Concatenate homogeneous agents into one batch."""
    n = batch.structure.num_leaves
    return batch.replace(
        observation=jym.tree.concatenate(batch.observation),
        next_observation=jym.tree.concatenate(batch.next_observation),
        action=jym.tree.concatenate(batch.action),
        reward=jym.tree.concatenate(batch.reward),
        terminated=jnp.repeat(batch.terminated, n, axis=0),
        truncated=jnp.repeat(batch.truncated, n, axis=0),
    )


class SharedCriticTeam(MultiAgentWrapper):
    """One critic update on a pooled batch from all agents, then copy weights to every agent."""

    @property
    def _first_agent(self):
        return eqx.tree_flatten_one_level(self.agents)[0][0]

    def _sync(self, first_agent):
        """ Sync the first agent's critics to all other agents. """
        def copy(agent):
            return agent.replace(
                critics=first_agent.critics,
                critics_target=first_agent.critics_target,
                optimizer_state={
                    **agent.optimizer_state,
                    "critics": first_agent.optimizer_state["critics"],
                },
            )

        return eqx.tree_at(
            lambda w: w.agents, self, jym.tree.map_one_level(copy, self.agents)
        )

    def update_critics_params(self, key, batch):
        """Override the default update_critics_params to use a pooled batch on a single agent
        Then sync the updated critic to all agents."""
        pooled_batch = pool(batch) # combine agents into one batch
        updated_first_agent = self._first_agent.update_critics_params(key, pooled_batch)
        return self._sync(updated_first_agent)
```

Wrap an already-initialised team and sync once so the copies start equal:

```python
team = SAC().init_agent(key, env) # is a MultiAgentWrapper
shared = SharedCritic(agents=team.agents, trainer=team.trainer) # replace with our custom MultiAgentWrapper
shared = shared._sync(shared._first_agent)
shared, metrics = shared.train(key, env)
```

