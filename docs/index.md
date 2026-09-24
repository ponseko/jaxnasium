# Jaxnasium

**A Lightweight Utility Library for JAX-based Reinforcement Learning Projects.**

Jaxnasium is not an environment suite, and not a framework that locks you in. Rather, it is your one-stop shop for your JAX RL code: a single environment API that existing suites are wrapped into, a set of general algorithms you can either import or copy into your project, and the
tooling to train, evaluate and sweep them.

1. 🕹️ **One environment API.** Import environments from Gymnax, Jumanji, Brax, Pgx, JaxMARL, xMinigrid, Navix or Craftax through `jym.make(...)`, automatically wrapped to a common standard.
2. 🤖 **Readable cross-suite algorithms.** Various algorithms that can operate on any of those environments in near-single-file philosophy, built in [Equinox](https://github.com/patrick-kidger/equinox) with a familiar [Stable-Baselines](https://github.com/DLR-RM/stable-baselines3)-like API and end-to-end JIT training in the spirit of [PureJaxRL](https://github.com/luchris429/purejaxrl).
3. 👥 **Multi-agent for free.** Single-agent algorithm code transparently upgrades to multi-agent environments through PyTrees and function transformations.
4. 📊 **Sweeps and evaluation.** Grid, random, Sobol and one-at-a-time searches over many seeds, locally or across a Slurm job array.
5. 🚀 **Project scaffolding.** Bootstrap a complete codebase with a single CLI command.

## Installation

```bash
pip install "jaxnasium[algs]"   # [algs] pulls in optax + distrax, needed for jaxnasium.algorithms
```

Third-party environment suites are *not* dependencies; install the ones you want to use
(e.g. `pip install gymnax`). For a brand-new project, let the [CLI](cli.md) set everything up for you:

```bash
uvx jaxnasium init <projectname>   # or: pipx run jaxnasium init <projectname>
cd <projectname> && uv run train.py
```

## Quickstart

```python
import jax
import jaxnasium as jym
from jaxnasium.algorithms import PPO

env = jym.make("CartPole-v1")
key = jax.random.PRNGKey(0)

algorithm = PPO(total_timesteps=500_000, learning_rate_start=2.5e-3)
agent, metrics = algorithm.train(key, env)
returns = agent.evaluate(key, env, num_eval_episodes=10)
```

`train` returns the trained agent alongside its training metrics: by default the mean
return of the episodes that finished in each training iteration, one scalar per iteration
(iterations without a finished episode yield `NaN`).

## 🏠 Environments

Jaxnasium does not aim to deliver a full environment suite. Instead, `jym.make(...)` imports
environments from existing suites (provided they are installed) and wraps them to the
Jaxnasium API standard:

```python
env = jym.make("Breakout-MinAtar")  # from Gymnax
env = jym.FlattenObservationWrapper(env)

algorithm = PPO(**some_good_hyperparameters)
agent, metrics = algorithm.train(jax.random.PRNGKey(0), env)

# > Using an environment from Gymnax via gymnax.make(Breakout-MinAtar).
# > Wrapping Gymnax environment with GymnaxWrapper
# >  Control this behavior by passing wrappers=[...] to jym.make
# > Wrapping environment in VecEnvWrapper
# > ... training results
```

By default `make` applies the adapter wrapper for the environment's library and a
[`LogWrapper`](api/Wrappers.md) for episode statistics; the algorithms add a
`VecEnvWrapper` themselves when training.

!!! info
    For convenience, Jaxnasium bundles the 5 [classic-control environments](https://gymnasium.farama.org/environments/classic_control/), which need no external dependencies.

See [Available Environments](api/Available-Environments.md) for a complete list of available environments.

### Environment API

The Jaxnasium API stays close to the *somewhat* established [Gymnax](https://github.com/RobertTLange/gymnax) API for the `reset()` and `step()` functions, but allows for truncated episodes in a manner closer to [Gymnasium](https://gymnasium.farama.org/).

```python
obs, env_state = env.reset(key)  # <-- Mirroring Gymnax

# env.step(): Gymnasium Timestep tuple with state information
(obs, reward, terminated, truncated, info), env_state = env.step(key, env_state, action)
```

Writing your own environment means subclassing [`jym.Environment`](api/Environment.md) and
implementing `step_env`, `reset_env`, `observation_space` and `action_space`.

## 🤖 Algorithms

Algorithms in `jaxnasium.algorithms` follow a near-single-file implementation philosophy. In contrast to implementations in [CleanRL](https://github.com/vwxyzjn/cleanrl) or [PureJaxRL](https://github.com/luchris429/purejaxrl), Jaxnasium algorithms are built in Equinox and follow a class-based design with a familiar [Stable-Baselines](https://github.com/DLR-RM/stable-baselines3) API.

Each algorithm is split into an **algorithm**, which holds the hyperparameters and the training
logic, and the **agent**, holding the trainable state along with the trainer that produced it.

```python
agent, metrics = PPO(**some_good_hyperparameters).train(key, env)

agent.get_action(key, obs)                          # act
agent, more_metrics = agent.train(key, env)         # keep training
agent.save("ppo_agent.eqx")                         # checkpoint, trainer included
```

Networks are configurable without touching the algorithm, observation and action spaces may
be arbitrary PyTrees of spaces, and observation/reward normalization is built in and
checkpointed with the agent. See [Algorithms](algorithms/Algorithms.md) for the details.

--8<-- "algorithms/_Algorithm-Table.md"

## 👥 Multi-agent

Multi-agent support is a **core pillar** of Jaxnasium's design rather than a separate API.
Multi-agent Environments simply expose a PyTree of spaces whose first level is the agent dimension, and
algorithms build one agent per entry automatically:

```python
env = jym.make("MPE_simple_spread_v3")  # a JaxMARL environment
agent, metrics = PPO().train(key, env)  # one PPO agent per agent in the environment

agent.get_action(key, obs)  # {"agent_0": ..., "agent_1": ..., ...}
```

Hyperparameters may even differ per agent, e.g. `PPO(gamma={"agent_0": 0.9, "agent_1": 0.99})`.
See [Multi-Agent](algorithms/Multi-Agent.md).

## 📊 Sweeps and evaluation

`jaxnasium.eval` turns any function into a parameter sweep, and provides a standard
train-and-evaluate trial for Jaxnasium algorithms. Jobs are addressed by index, so the same
script runs locally in a loop or as a Slurm job array:

```python
from jaxnasium.eval import AlgorithmEvaluation, GridSearch, SobolSearch, Sweep

sweep = Sweep(
    AlgorithmEvaluation(jax.random.split(key, 10), batch_size=5, save_path="results/"),
    GridSearch({"env": ["CartPole-v1", "Acrobot-v1"], "algorithm": ["PPO", "DQN"]}),
    SobolSearch({"learning_rate_start": (1e-4, 1e-2, "log")}, num_samples=32, seed=key),
)

len(sweep)          # 2 envs * 2 algorithms * 32 learning rates
result = sweep[0]()  # run one job
```

See [Sweeps](eval/Sweeps.md) and [Searches](eval/Searches.md).

## Where to go next

| | |
| --- | --- |
| [Environment](api/Environment.md) · [Spaces](api/Spaces.md) · [Wrappers](api/Wrappers.md) | Build or adapt an environment |
| [Algorithms](algorithms/Algorithms.md) · [Multi-Agent](algorithms/Multi-Agent.md) | Train agents |
| [Networks](algorithms/networks/Networks.md) · [Architectures](algorithms/networks/Architectures.md) | Swap in your own model |
| [Sweeps](eval/Sweeps.md) · [Searches](eval/Searches.md) | Run experiments at scale |
| [Compilation](api/Compilation.md) | Ahead-of-time compilation and caching |
| [Checkpointing](algorithms/core/Checkpointing.md) · [Tree utilities](tree/Tree.md) · [CLI](cli.md) | Everything else |