# Available Environments

Jaxnasium doesn't bundle a large number of environments directly. Instead, it relies on existing work and wraps these environments in various wrappers to conform to the Jaxnasium API. This approach allows users to leverage a wide array of established environments while maintaining the performance and flexibility offered by JAX.

## Native Environments

**Note:** The classic control environments are natively implemented ánd bundled in Jaxnasium convenience:

- `CartPole-v1`
- `MountainCar-v0` 
- `Acrobot-v1`
- `Pendulum-v1`
- `MountainCarContinuous-v0`

These environments are implemented directly in Jaxnasium and don't require external dependencies.

## External Environments using the Jaxnasium API

These environments run without wrappers.

## External Environment Libraries

Jaxnasium integrates with the following external environment libraries through wrapper adapters.
See the end of this page for a full list of available environments.

### [Gymnax](https://github.com/RobertTLange/gymnax)
JAX implementations of OpenAI's Gym environments, offering accelerated and parallelized rollouts. Includes classic control, bsuite, and MinAtar environments.

### [Jumanji](https://github.com/instadeepai/jumanji)
A suite of diverse, scalable reinforcement learning environments implemented in JAX by DeepMind. Focuses on combinatorial problems and general decision-making tasks.

### [Brax](https://github.com/google/brax)
A fast and flexible physics simulation engine for training and evaluating rigid body environments in JAX by Google.

### [Pgx](https://github.com/sotetsuk/pgx)
JAX implementations of various board games and classic environments, including chess, Go, shogi, and more.

### [JaxMARL](https://github.com/FLAIROx/JaxMARL)
Multi-agent reinforcement learning environments implemented in JAX, including MPE (Multi-Particle Environment) scenarios and other multi-agent tasks.

### [xMinigrid](https://github.com/dunnolab/xland-minigrid)
JAX implementation of MiniGrid environments, including XLand variants for procedural generation research.

### [Navix](https://github.com/epignatelli/navix)
JAX implementation of navigation environments, providing various gridworld navigation tasks.

### [Craftax](https://github.com/MichaelTMatthews/Craftax)
JAX implementation of Craftax environments, inspired by Minecraft-like crafting and survival tasks.

## Usage

To use any of these environments, simply call:

```python
import jaxnasium as jym

# Native environments
env = jym.make("CartPole-v1")

# External environments (requires installing the respective library)
env = jym.make("Breakout-MinAtar")  # Gymnax
env = jym.make("Game2048-v1")       # Jumanji
env = jym.make("ant")               # Brax
env = jym.make("chess")             # Pgx
```

**Note:** External environment libraries are not bundled as dependencies and need to be installed manually (e.g., via pip) before use.

## Complete List of Registered Environments

Below is the complete list of all registered environment strings available in Jaxnasium:

!!! note "Auto-generated List"
    This list is automatically generated from the Jaxnasium registry.

--8<-- "api/_Available-Environments-List.md"