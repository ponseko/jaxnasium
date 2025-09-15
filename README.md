
# Jaxnasium: A Lightweight Utility Library for JAX-based RL Projects


[![PyPI version](https://badge.fury.io/py/jaxnasium.svg)](https://badge.fury.io/py/jaxnasium)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://ponseko.github.io/jaxnasium/)

Jaxnasium lets you

1. 🕹️ Import your favourite environments from various libraries with a single API and automatically wrap them to a common standard.
2. 🚀 Bootstrap new JAX RL projects with a single CLI command and get started instantly with a complete codebase.
3. 🤖 Jaxnasium comes equiped with standard **general** RL implementations based on a near-single-file philosophy. You can either import these as off-the-shelf algorithms or copy over the code and tweak them for your problem. These algorithms follow the ideas of [PureJaxRL](https://github.com/luchris429/purejaxrl) for extremely fast end-to-end RL training in JAX.

For more details, see the [📖 Documentation](https://ponseko.github.io/jaxnasium/).

## 🚀 Getting started

Jaxnasium lets you bootstrap your new reinforcement learning projects directly from the command line. As such, for new projects, the easiest way to get started is via [uv](https://docs.astral.sh/uv/getting-started/installation/):

> ```bash
> uvx jaxnasium <projectname>
> uv run example_train.py
> 
> # ... or via pipx
> pipx run jaxnasium <projectname>
> # activate a virtual environment in your preferred way, e.g. conda
> python example_train.py
> ```

This will set up a Python project folder structure with (optionally) an environment template and (optionally) algorithm code for you to tailor to your problem.

For existing projects, you can simply install Jaxnasium via `pip` and import the required functionality.

> ```bash
> pip install jaxnasium
> ```

> ```python
> import jax
> import jaxnasium as jym
> from jaxnasium.algorithms import PPO
> 
> env = jym.make("CartPole-v1")
> env = jaxnasium.LogWrapper(env)
> rng = jax.random.PRNGKey(0)
> agent = PPO(total_timesteps=5e5, learning_rate=2.5e-3)
> agent = agent.train(rng, env)
> ```

## 🏠 Environments

Jaxnasium is not aimed at delivering a full environment suite. However, it does come equipped with a `jym.make(...)` command to import environments from existing suites (provided that these are installed) and wrap them appropriately to the Jaxnasium API standard. For example, using environments from Gymnax:

```python
import jaxnasium as jym
from jaxnasium.algorithms import PPO
import jax

env = jym.make("Breakout-MinAtar")
env = jym.FlattenObservationWrapper(env)
env = jym.LogWrapper(env)

agent = PPO(**some_good_hyperparameters)
agent = agent.train(jax.random.PRNGKey(0), env)

# > Using an environment from Gymnax via gymnax.make(Breakout-MinAtar).
# > Wrapping Gymnax environment with GymnaxWrapper
# >  Disable this behavior by passing wrapper=False
# > Wrapping environment in VecEnvWrapper
# > ... training results
```

!!!info 
    For convenience, Jaxnasium does include the 5 [classic-control environments](https://gymnasium.farama.org/environments/classic_control/).

See the [Environments](./api/Available-Environments.md) page for a complete list of available environments.

### Environment API

The Jaxnasium API stays close to the *somewhat* established [Gymnax](https://github.com/RobertTLange/gymnax) API for the `reset()` and `step()` functions, but allows for truncated episodes in a manner closer to [Gymnasium](https://gymnasium.farama.org/).

```python
env = jym.make(...)

obs, env_state = env.reset(key) # <-- Mirroring Gymnax

# env.step(): Gymnasium Timestep tuple with state information
(obs, reward, terminated, truncated, info), env_state = env.step(key, state, action)
```

## 🤖 Algorithms

Algorithms in `jaxnasium.algorithms` are built following a near-single-file implementation philosophy in mind. In contrast to implementations in [CleanRL](https://github.com/vwxyzjn/cleanrl) or [PureJaxRL](https://github.com/luchris429/purejaxrl), Jaxnasium algorithms are built in Equinox and follow a class-based design with a familiar [Stable-Baselines](https://github.com/DLR-RM/stable-baselines3) API. 

```python
from jaxnasium.algorithms import PPO
import jax

env = ...
agent = PPO(**some_good_hyperparameters)
agent = agent.train(jax.random.PRNGKey(0), env)
```

See the [Algorithms](./algorithms/Algorithms.md) for more details on the included algorithms..

## Available Algorithms

| Algorithm | Multi-Agent<sup>1</sup> | Observation Spaces | Action Spaces | Composite (nested) Spaces<sup>2</sup> |
|-----------|-------------------------|-------------------|---------------|--------------------------------------|
| **PPO**   | ✅                      | `Box`, `Discrete`, `MultiDiscrete` | `Box`, `Discrete`, `MultiDiscrete` | ✅ |
| **DQN**   | ✅                      | `Box`, `Discrete`, `MultiDiscrete` | `Discrete`, `MultiDiscrete`<sup>3</sup> | ✅ |
| **PQN**   | ✅                      | `Box`, `Discrete`, `MultiDiscrete` | `Discrete`, `MultiDiscrete`<sup>3</sup> | ✅ |
| **SAC**   | ✅                      | `Box`, `Discrete`, `MultiDiscrete` | `Box`, `Discrete`, `MultiDiscrete` | ✅ |

<sup>1</sup> All algorithms support automatic multi-agent transformation through the `auto_upgrade_multi_agent` parameter. See [Multi-Agent documentation](https://ponseko.github.io/jaxnasium/algorithms/Multi-Agent/) for more information.

<sup>2</sup> Algorithms support composite (nested) spaces. See [Spaces documentation](https://ponseko.github.io/jaxnasium/api/Spaces/) for more information.

<sup>3</sup> MultiDiscrete action spaces in PQN and DQN are only supported when flattening to a Discrete action space. E.g. via the `FlattenActionSpaceWrapper`.

