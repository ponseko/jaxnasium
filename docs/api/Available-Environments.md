# Available Environments

Jaxnasium doesn't bundle a large number of environments directly. Instead, it relies on existing work and wraps these environments in various wrappers to conform to the Jaxnasium API. This approach allows users to leverage a wide array of established environments while maintaining the performance and flexibility offered by JAX.

## Native Environments

For convenience, the five [classic control environments](https://gymnasium.farama.org/environments/classic_control/)
are natively implemented and bundled in Jaxnasium, and require no external dependencies:

- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`
- `Pendulum-v1`
- `MountainCarContinuous-v0`

## External Environments using the Jaxnasium API

These environments run without wrappers.
#TODO

## External Environment Libraries

Jaxnasium integrates with the following external environment libraries through wrapper adapters.
See the end of this page for a full list of available environments.
These are not bundled as dependencies and need to be installed manually (e.g. via `pip`) before use.

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

### [Octax](https://github.com/riiswa/octax)
JAX CHIP-8 emulator environments (Brix, Pong, Tetris, and other classic games).

### [Craftax](https://github.com/MichaelTMatthews/Craftax)
JAX implementation of Craftax environments, inspired by Minecraft-like crafting and survival tasks.

## Usage

To create any of these environments, simply pass its id to `jaxnasium.make`:

```python
import jaxnasium as jym

# Native environments
env = jym.make("CartPole-v1")

# External environments (requires installing the respective library)
env = jym.make("Breakout-MinAtar")  # Gymnax
env = jym.make("Game2048-v1")  # Jumanji
env = jym.make("ant")  # Brax
env = jym.make("chess")  # Pgx
env = jym.make("octax:brix")  # Octax
```

Every id can also be given with an explicit provider prefix (e.g. `gymnax:Breakout-MinAtar`),
which is useful when the same name exists in more than one library.

By default, `make` applies the wrapper for the environment's library (translating it to the
Jaxnasium API) and a [`LogWrapper`](Wrappers.md). Pass `wrappers=[...]` to control this.

::: jaxnasium._registry.Registry
    options:
        heading_level: 3
        members:
            - make
            - register
            - register_alias
            - registered_envs
            - print_envs

## Complete List of Registered Environments

Below is the complete list of all registered environment ids available in Jaxnasium.

!!! note "Auto-generated List"
    This list is automatically generated from the Jaxnasium registry.

--8<-- "api/_Available-Environments-List.md"
