import difflib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Type

from ._environment import Environment
from ._wrappers import LogWrapper
from ._wrappers_ext import (
    BraxWrapper,
    GymnaxWrapper,
    JaxMARLWrapper,
    JumanjiWrapper,
    NavixWrapper,
    PgxWrapper,
    Wrapper,
    xMinigridWrapper,
)

logger = logging.getLogger(__name__)


def _wrap_env(
    env: Environment | Any, wrapper: Type[Wrapper], **wrapper_kwargs
) -> Environment:
    """Simply wraps an environment and outputs what happened to a logger"""
    logger.info(f"Wrapping environment with {wrapper.__name__}")
    return wrapper(env, **wrapper_kwargs)


@dataclass
class Registry:
    _environments: Dict[str, Type[Environment]] = field(default_factory=dict)
    _aliases: Dict[str, str] = field(default_factory=dict)

    def register(self, id: str, **kwargs):
        """Register an environment with the registry.

        **Arguments**:
            `id`: The environment ID (e.g., "CartPole-v1")
            `entry_point`: Optional entry point string for lazy loading
            `**kwargs`: currently unused
        """

        def decorator(env_class: Type[Environment]) -> Type[Environment]:
            self._environments[id] = env_class
            return env_class

        return decorator

    def register_alias(self, alias: str, target: str):
        """Register an alias for an environment.

        **Arguments**:
            `alias`: The alias to register
            `target`: The target environment ID
        """
        self._aliases[alias] = target

    def make(
        self,
        id: str,
        wrappers: List[Type[Wrapper] | Literal["external_lib_wrapper"]] = [
            "external_lib_wrapper",
            LogWrapper,
        ],
        **env_kwargs,
    ) -> Environment:
        """Create an environment instance.

        **Arguments**:
            `id`: The environment ID
            `wrappers`: List of wrappers to apply to the environment.
                - `external_lib_wrapper` (string): Wrapper for external libraries (e.g. Gymnax, Jumanji, Brax).
                    only used if a environment is loaded from a supported external library.
                - `LogWrapper` (class): Wrapper for logging the actions
            `**env_kwargs`: Environment constructor arguments
        """
        # Handle aliases
        assert id is not None, "Environment ID cannot be None"
        env = None
        if id in self._aliases:
            id = self._aliases[id]

        # Try direct registration first
        if id in self._environments:
            env_class = self._environments[id]
            env = env_class(**env_kwargs)

        # Try external package detection
        if ":" in id:
            package, env_name = id.split(":", 1)
            wrap = "external_lib_wrapper" in wrappers
            env = self._make_external(package, env_name, wrap=wrap, **env_kwargs)

        if env is not None:
            for wrapper in wrappers:
                if wrapper == "external_lib_wrapper":
                    continue  # applied in _make_external (if applicable)
                env = _wrap_env(env, wrapper)
            return env

        matches = difflib.get_close_matches(id, self.registered_envs, n=1, cutoff=0.6)
        if matches:
            raise ValueError(
                f"Environment {id} not found in registry. Did you mean {matches[0]}?"
            )
        else:
            raise ValueError(f"Environment {id} not found in registry")

    def _make_external(
        self, package: str, env_name: str, *, wrap: bool, **env_kwargs
    ) -> Environment:
        """Create an external environment with appropriate wrapper.

        **Arguments**:
            `package`: The package to use (e.g., "gymnax", "jumanji", "brax", ...)
            `env_name`: The environment name
            `**env_kwargs`: Environment constructor arguments
        """
        try:
            if package == "gymnax":
                import gymnax

                env, _ = gymnax.make(env_name, **env_kwargs)
                if wrap:
                    return _wrap_env(env, GymnaxWrapper)
                return env  # type: ignore
            elif package == "jumanji":
                import jumanji

                env = jumanji.make(env_name, **env_kwargs)  # type: ignore
                if wrap:
                    return _wrap_env(env, JumanjiWrapper)
                return env  # type: ignore
            elif package == "brax":
                import brax.envs

                env = brax.envs.get_environment(env_name, **env_kwargs)
                if wrap:
                    return _wrap_env(env, BraxWrapper)
                return env  # type: ignore
            elif package == "pgx":
                import pgx

                env = pgx.make(env_name, **env_kwargs)  # type: ignore
                if wrap:
                    return _wrap_env(env, PgxWrapper)
                return env  # type: ignore
            elif package == "jaxmarl":
                import jaxmarl

                env = jaxmarl.make(env_name, **env_kwargs)
                if wrap:
                    return _wrap_env(env, JaxMARLWrapper)
                return env  # type: ignore
            elif package == "xminigrid":
                import xminigrid

                env, env_params = xminigrid.make(env_name, **env_kwargs)
                if wrap:
                    return _wrap_env(env, xMinigridWrapper, _params=env_params)
                return env  # type: ignore
            elif package == "navix":
                import navix

                env = navix.make(env_name, **env_kwargs)
                if wrap:
                    return _wrap_env(env, NavixWrapper)
                return env  # type: ignore
            elif package == "craftax":
                from craftax import craftax_env

                env = craftax_env.make_craftax_env_from_name(
                    env_name, auto_reset=False, **env_kwargs
                )
                if wrap:
                    return _wrap_env(env, GymnaxWrapper)  # Uses Gymnax style API
                return env  # type: ignore
            else:
                raise ValueError(f"Unsupported/unknown external package: {package}")
        except ImportError as e:
            raise ImportError(
                f"Package {package} not installed. Please install manually via pip: {e}"
            )

    def get_env_class(self, id: str) -> Type[Environment]:
        """Get the environment class for an environment ID.

        **Arguments**:
            `id`: The environment ID
        """
        if id in self._aliases:
            id = self._aliases[id]

        if id in self._environments:
            return self._environments[id]

        if ":" in id:
            raise ValueError("Cannot get environment class for external environments")

        raise ValueError(f"Environment {id} not found in registry")

    @property
    def registered_envs(self) -> List[str]:
        """List all environments in the registry as a flat list."""
        return list(self._environments.keys()) + list(self._aliases.keys())

    def print_envs(self) -> None:
        """Pretty prints the available environments in the registry."""
        print("Available environments in Jaxnasium:")
        print("=" * 50)

        def format_block(envs, title, icon):
            if not envs:
                return ""
            envs_per_line = 3
            # Get max length across ALL environments for consistent column width
            all_envs = list(self._environments.keys()) + [
                alias for alias in self._aliases.keys()
            ]
            max_length = max(len(env) for env in all_envs) if all_envs else 0
            formatted = "\n".join(
                " ".join(
                    f"• {env}".ljust(max_length + 2)
                    for env in envs[i : i + envs_per_line]
                )
                for i in range(0, len(envs), envs_per_line)
            )
            return f"\n{icon} {title}:\n{formatted}"

        # Native environments
        if self._environments:
            native_envs = sorted(self._environments.keys())
            print(format_block(native_envs, "Native environments", "🔧"))

        # Group external environments by package
        external_packages = {}
        for alias, target in self._aliases.items():
            if ":" in target:
                package = target.split(":")[0]
                if package not in external_packages:
                    external_packages[package] = []
                external_packages[package].append(alias)

        # Print external environments grouped by package
        for package in sorted(external_packages.keys()):
            envs = sorted(external_packages[package])
            print(format_block(envs, f"{package.title()} environments", "📦"))

        print(f"\nTotal: {len(self._environments) + len(self._aliases)} environments")
        print(
            "Note that external libraries📦 are not bundled as dependencies and need to be installed manually (e.g. via pip)."
        )
        print("=" * 50)


registry = Registry()
make = registry.make

# Gymnax envs
# Classic control accessible only with "gymnax:" prefix as they are included in Jaxnasium
registry.register_alias("gymnax:CartPole-v1", "gymnax:CartPole-v1")
registry.register_alias("gymnax:Acrobot-v1", "gymnax:Acrobot-v1")
registry.register_alias("gymnax:Pendulum-v1", "gymnax:Pendulum-v1")
registry.register_alias("gymnax:MountainCar-v0", "gymnax:MountainCar-v0")
registry.register_alias(
    "gymnax:MountainCarContinuous-v0", "gymnax:MountainCarContinuous-v0"
)
registry.register_alias("Asterix-MinAtar", "gymnax:Asterix-MinAtar")
registry.register_alias("Breakout-MinAtar", "gymnax:Breakout-MinAtar")
registry.register_alias("Freeway-MinAtar", "gymnax:Freeway-MinAtar")
registry.register_alias("SpaceInvaders-MinAtar", "gymnax:SpaceInvaders-MinAtar")
registry.register_alias("DeepSea-bsuite", "gymnax:DeepSea-bsuite")
registry.register_alias("Catch-bsuite", "gymnax:Catch-bsuite")
registry.register_alias("MemoryChain-bsuite", "gymnax:MemoryChain-bsuite")
registry.register_alias("UmbrellaChain-bsuite", "gymnax:UmbrellaChain-bsuite")
registry.register_alias("DiscountingChain-bsuite", "gymnax:DiscountingChain-bsuite")
registry.register_alias("MNISTBandit-bsuite", "gymnax:MNISTBandit-bsuite")
registry.register_alias("SimpleBandit-bsuite", "gymnax:SimpleBandit-bsuite")
registry.register_alias("FourRooms-misc", "gymnax:FourRooms-misc")
registry.register_alias("MetaMaze-misc", "gymnax:MetaMaze-misc")
registry.register_alias("PointRobot-misc", "gymnax:PointRobot-misc")
registry.register_alias("BernoulliBandit-misc", "gymnax:BernoulliBandit-misc")
registry.register_alias("GaussianBandit-misc", "gymnax:GaussianBandit-misc")
registry.register_alias("Reacher-misc", "gymnax:Reacher-misc")
registry.register_alias("Swimmer-misc", "gymnax:Swimmer-misc")
registry.register_alias("Pong-misc", "gymnax:Pong-misc")

# Jumanji envs
registry.register_alias("Game2048-v1", "jumanji:Game2048-v1")
registry.register_alias("GraphColoring-v1", "jumanji:GraphColoring-v1")
registry.register_alias("Minesweeper-v0", "jumanji:Minesweeper-v0")
registry.register_alias("RubiksCube-v0", "jumanji:RubiksCube-v0")
registry.register_alias(
    "RubiksCube-partly-scrambled-v0", "jumanji:RubiksCube-partly-scrambled-v0"
)
registry.register_alias("SlidingTilePuzzle-v0", "jumanji:SlidingTilePuzzle-v0")
registry.register_alias("Sudoku-v0", "jumanji:Sudoku-v0")
registry.register_alias("Sudoku-very-easy-v0", "jumanji:Sudoku-very-easy-v0")
registry.register_alias("BinPack-v2", "jumanji:BinPack-v2")
registry.register_alias("FlatPack-v0", "jumanji:FlatPack-v0")
registry.register_alias("JobShop-v0", "jumanji:JobShop-v0")
registry.register_alias("Knapsack-v1", "jumanji:Knapsack-v1")
registry.register_alias("Tetris-v0", "jumanji:Tetris-v0")
registry.register_alias("Cleaner-v0", "jumanji:Cleaner-v0")
registry.register_alias("Connector-v2", "jumanji:Connector-v2")
registry.register_alias("CVRP-v1", "jumanji:CVRP-v1")
registry.register_alias("MultiCVRP-v0", "jumanji:MultiCVRP-v0")
registry.register_alias("Maze-v0", "jumanji:Maze-v0")
registry.register_alias("RobotWarehouse-v0", "jumanji:RobotWarehouse-v0")
registry.register_alias("Snake-v1", "jumanji:Snake-v1")
registry.register_alias("TSP-v1", "jumanji:TSP-v1")
registry.register_alias("MMST-v0", "jumanji:MMST-v0")
registry.register_alias("PacMan-v1", "jumanji:PacMan-v1")
registry.register_alias("Sokoban-v0", "jumanji:Sokoban-v0")
registry.register_alias("LevelBasedForaging-v0", "jumanji:LevelBasedForaging-v0")
registry.register_alias("SearchAndRescue-v0", "jumanji:SearchAndRescue-v0")

# Brax envs
registry.register_alias("ant", "brax:ant")
registry.register_alias("halfcheetah", "brax:halfcheetah")
registry.register_alias("hopper", "brax:hopper")
registry.register_alias("humanoid", "brax:humanoid")
registry.register_alias("humanoidstandup", "brax:humanoidstandup")
registry.register_alias("inverted_pendulum", "brax:inverted_pendulum")
registry.register_alias("inverted_double_pendulum", "brax:inverted_double_pendulum")
registry.register_alias("pusher", "brax:pusher")
registry.register_alias("reacher", "brax:reacher")
registry.register_alias("walker2d", "brax:walker2d")

# Pgx envs
registry.register_alias("2048", "pgx:2048")
registry.register_alias("animal_shogi", "pgx:animal_shogi")
registry.register_alias("backgammon", "pgx:backgammon")
registry.register_alias("chess", "pgx:chess")
registry.register_alias("connect_four", "pgx:connect_four")
registry.register_alias("gardner_chess", "pgx:gardner_chess")
registry.register_alias("go_9x9", "pgx:go_9x9")
registry.register_alias("go_19x19", "pgx:go_19x19")
registry.register_alias("hex", "pgx:hex")
registry.register_alias("kuhn_poker", "pgx:kuhn_poker")
registry.register_alias("leduc_holdem", "pgx:leduc_holdem")
registry.register_alias("minatar-asterix", "pgx:minatar-asterix")
registry.register_alias("minatar-breakout", "pgx:minatar-breakout")
registry.register_alias("minatar-freeway", "pgx:minatar-freeway")
registry.register_alias("minatar-seaquest", "pgx:minatar-seaquest")
registry.register_alias("minatar-space_invaders", "pgx:minatar-space_invaders")
registry.register_alias("othello", "pgx:othello")
registry.register_alias("shogi", "pgx:shogi")
registry.register_alias("sparrow_mahjong", "pgx:sparrow_mahjong")
registry.register_alias("tic_tac_toe", "pgx:tic_tac_toe")

# JaxMARL envs
registry.register_alias("MPE_simple_v3", "jaxmarl:MPE_simple_v3")
registry.register_alias("MPE_simple_tag_v3", "jaxmarl:MPE_simple_tag_v3")
registry.register_alias("MPE_simple_world_comm_v3", "jaxmarl:MPE_simple_world_comm_v3")
registry.register_alias("MPE_simple_spread_v3", "jaxmarl:MPE_simple_spread_v3")
registry.register_alias("MPE_simple_crypto_v3", "jaxmarl:MPE_simple_crypto_v3")
registry.register_alias(
    "MPE_simple_speaker_listener_v4", "jaxmarl:MPE_simple_speaker_listener_v4"
)
registry.register_alias("MPE_simple_push_v3", "jaxmarl:MPE_simple_push_v3")
registry.register_alias("MPE_simple_adversary_v3", "jaxmarl:MPE_simple_adversary_v3")
registry.register_alias("MPE_simple_reference_v3", "jaxmarl:MPE_simple_reference_v3")
registry.register_alias("MPE_simple_facmac_v1", "jaxmarl:MPE_simple_facmac_v1")
registry.register_alias("MPE_simple_facmac_3a_v1", "jaxmarl:MPE_simple_facmac_3a_v1")
registry.register_alias("MPE_simple_facmac_6a_v1", "jaxmarl:MPE_simple_facmac_6a_v1")
registry.register_alias("MPE_simple_facmac_9a_v1", "jaxmarl:MPE_simple_facmac_9a_v1")
registry.register_alias("switch_riddle", "jaxmarl:switch_riddle")
registry.register_alias("SMAX", "jaxmarl:SMAX")
registry.register_alias("HeuristicEnemySMAX", "jaxmarl:HeuristicEnemySMAX")
# registry.register_alias("LearnedPolicyEnemySMAX", "jaxmarl:LearnedPolicyEnemySMAX")
registry.register_alias("ant_4x2", "jaxmarl:ant_4x2")
registry.register_alias("halfcheetah_6x1", "jaxmarl:halfcheetah_6x1")
registry.register_alias("hopper_3x1", "jaxmarl:hopper_3x1")
registry.register_alias("humanoid_9|8", "jaxmarl:humanoid_9|8")
registry.register_alias("walker2d_2x3", "jaxmarl:walker2d_2x3")
registry.register_alias("storm", "jaxmarl:storm")
registry.register_alias("storm_2p", "jaxmarl:storm_2p")
registry.register_alias("storm_np", "jaxmarl:storm_np")
registry.register_alias("hanabi", "jaxmarl:hanabi")
registry.register_alias("overcooked", "jaxmarl:overcooked")
registry.register_alias("overcooked_v2", "jaxmarl:overcooked_v2")
registry.register_alias("coin_game", "jaxmarl:coin_game")
registry.register_alias("jaxnav", "jaxmarl:jaxnav")

# xminigrid envs (Xland MiniGrid)
registry.register_alias("XLand-MiniGrid-R1-9x9", "xminigrid:XLand-MiniGrid-R1-9x9")
registry.register_alias("XLand-MiniGrid-R1-11x11", "xminigrid:XLand-MiniGrid-R1-11x11")
registry.register_alias("XLand-MiniGrid-R1-13x13", "xminigrid:XLand-MiniGrid-R1-13x13")
registry.register_alias("XLand-MiniGrid-R1-15x15", "xminigrid:XLand-MiniGrid-R1-15x15")
registry.register_alias("XLand-MiniGrid-R1-17x17", "xminigrid:XLand-MiniGrid-R1-17x17")
registry.register_alias("XLand-MiniGrid-R2-9x9", "xminigrid:XLand-MiniGrid-R2-9x9")
registry.register_alias("XLand-MiniGrid-R2-11x11", "xminigrid:XLand-MiniGrid-R2-11x11")
registry.register_alias("XLand-MiniGrid-R2-13x13", "xminigrid:XLand-MiniGrid-R2-13x13")
registry.register_alias("XLand-MiniGrid-R2-15x15", "xminigrid:XLand-MiniGrid-R2-15x15")
registry.register_alias("XLand-MiniGrid-R2-17x17", "xminigrid:XLand-MiniGrid-R2-17x17")
registry.register_alias("XLand-MiniGrid-R4-9x9", "xminigrid:XLand-MiniGrid-R4-9x9")
registry.register_alias("XLand-MiniGrid-R4-11x11", "xminigrid:XLand-MiniGrid-R4-11x11")
registry.register_alias("XLand-MiniGrid-R4-13x13", "xminigrid:XLand-MiniGrid-R4-13x13")
registry.register_alias("XLand-MiniGrid-R4-15x15", "xminigrid:XLand-MiniGrid-R4-15x15")
registry.register_alias("XLand-MiniGrid-R4-17x17", "xminigrid:XLand-MiniGrid-R4-17x17")
registry.register_alias("XLand-MiniGrid-R6-13x13", "xminigrid:XLand-MiniGrid-R6-13x13")
registry.register_alias("XLand-MiniGrid-R6-17x17", "xminigrid:XLand-MiniGrid-R6-17x17")
registry.register_alias("XLand-MiniGrid-R6-19x19", "xminigrid:XLand-MiniGrid-R6-19x19")
registry.register_alias("XLand-MiniGrid-R9-16x16", "xminigrid:XLand-MiniGrid-R9-16x16")
registry.register_alias("XLand-MiniGrid-R9-19x19", "xminigrid:XLand-MiniGrid-R9-19x19")
registry.register_alias("XLand-MiniGrid-R9-25x25", "xminigrid:XLand-MiniGrid-R9-25x25")
registry.register_alias(
    "MiniGrid-BlockedUnlockPickUp", "xminigrid:MiniGrid-BlockedUnlockPickUp"
)
registry.register_alias("MiniGrid-DoorKey-5x5", "xminigrid:MiniGrid-DoorKey-5x5")
registry.register_alias("MiniGrid-DoorKey-6x6", "xminigrid:MiniGrid-DoorKey-6x6")
registry.register_alias("MiniGrid-DoorKey-8x8", "xminigrid:MiniGrid-DoorKey-8x8")
registry.register_alias("MiniGrid-DoorKey-16x16", "xminigrid:MiniGrid-DoorKey-16x16")
registry.register_alias("MiniGrid-Empty-5x5", "xminigrid:MiniGrid-Empty-5x5")
registry.register_alias("MiniGrid-Empty-6x6", "xminigrid:MiniGrid-Empty-6x6")
registry.register_alias("MiniGrid-Empty-8x8", "xminigrid:MiniGrid-Empty-8x8")
registry.register_alias("MiniGrid-Empty-16x16", "xminigrid:MiniGrid-Empty-16x16")
registry.register_alias(
    "MiniGrid-EmptyRandom-5x5", "xminigrid:MiniGrid-EmptyRandom-5x5"
)
registry.register_alias(
    "MiniGrid-EmptyRandom-6x6", "xminigrid:MiniGrid-EmptyRandom-6x6"
)
registry.register_alias(
    "MiniGrid-EmptyRandom-8x8", "xminigrid:MiniGrid-EmptyRandom-8x8"
)
registry.register_alias(
    "MiniGrid-EmptyRandom-16x16", "xminigrid:MiniGrid-EmptyRandom-16x16"
)
registry.register_alias("MiniGrid-FourRooms", "xminigrid:MiniGrid-FourRooms")
registry.register_alias("MiniGrid-LockedRoom", "xminigrid:MiniGrid-LockedRoom")
registry.register_alias("MiniGrid-MemoryS8", "xminigrid:MiniGrid-MemoryS8")
registry.register_alias("MiniGrid-MemoryS16", "xminigrid:MiniGrid-MemoryS16")
registry.register_alias("MiniGrid-MemoryS32", "xminigrid:MiniGrid-MemoryS32")
registry.register_alias("MiniGrid-MemoryS64", "xminigrid:MiniGrid-MemoryS64")
registry.register_alias("MiniGrid-MemoryS128", "xminigrid:MiniGrid-MemoryS128")
registry.register_alias("MiniGrid-Playground", "xminigrid:MiniGrid-Playground")
registry.register_alias("MiniGrid-Unlock", "xminigrid:MiniGrid-Unlock")
registry.register_alias("MiniGrid-UnlockPickUp", "xminigrid:MiniGrid-UnlockPickUp")

# Navix envs
registry.register_alias("Navix-Empty-5x5-v0", "navix:Navix-Empty-5x5-v0")
registry.register_alias("Navix-Empty-6x6-v0", "navix:Navix-Empty-6x6-v0")
registry.register_alias("Navix-Empty-8x8-v0", "navix:Navix-Empty-8x8-v0")
registry.register_alias("Navix-Empty-16x16-v0", "navix:Navix-Empty-16x16-v0")
registry.register_alias("Navix-Empty-Random-5x5-v0", "navix:Navix-Empty-Random-5x5-v0")
registry.register_alias("Navix-Empty-Random-6x6-v0", "navix:Navix-Empty-Random-6x6-v0")
registry.register_alias("Navix-Empty-Random-8x8-v0", "navix:Navix-Empty-Random-8x8-v0")
registry.register_alias(
    "Navix-Empty-Random-16x16-v0", "navix:Navix-Empty-Random-16x16-v0"
)
registry.register_alias("Navix-DoorKey-5x5-v0", "navix:Navix-DoorKey-5x5-v0")
registry.register_alias("Navix-DoorKey-6x6-v0", "navix:Navix-DoorKey-6x6-v0")
registry.register_alias("Navix-DoorKey-8x8-v0", "navix:Navix-DoorKey-8x8-v0")
registry.register_alias("Navix-DoorKey-16x16-v0", "navix:Navix-DoorKey-16x16-v0")
registry.register_alias(
    "Navix-DoorKey-Random-5x5-v0", "navix:Navix-DoorKey-Random-5x5-v0"
)
registry.register_alias(
    "Navix-DoorKey-Random-6x6-v0", "navix:Navix-DoorKey-Random-6x6-v0"
)
registry.register_alias(
    "Navix-DoorKey-Random-8x8-v0", "navix:Navix-DoorKey-Random-8x8-v0"
)
registry.register_alias(
    "Navix-DoorKey-Random-16x16-v0", "navix:Navix-DoorKey-Random-16x16-v0"
)
registry.register_alias("Navix-FourRooms-v0", "navix:Navix-FourRooms-v0")
registry.register_alias("Navix-KeyCorridorS3R1-v0", "navix:Navix-KeyCorridorS3R1-v0")
registry.register_alias("Navix-KeyCorridorS3R2-v0", "navix:Navix-KeyCorridorS3R2-v0")
registry.register_alias("Navix-KeyCorridorS3R3-v0", "navix:Navix-KeyCorridorS3R3-v0")
registry.register_alias("Navix-KeyCorridorS4R3-v0", "navix:Navix-KeyCorridorS4R3-v0")
registry.register_alias("Navix-KeyCorridorS5R3-v0", "navix:Navix-KeyCorridorS5R3-v0")
registry.register_alias("Navix-KeyCorridorS6R3-v0", "navix:Navix-KeyCorridorS6R3-v0")
registry.register_alias("Navix-LavaGapS5-v0", "navix:Navix-LavaGapS5-v0")
registry.register_alias("Navix-LavaGapS6-v0", "navix:Navix-LavaGapS6-v0")
registry.register_alias("Navix-LavaGapS7-v0", "navix:Navix-LavaGapS7-v0")
registry.register_alias(
    "Navix-SimpleCrossingS9N1-v0", "navix:Navix-SimpleCrossingS9N1-v0"
)
registry.register_alias(
    "Navix-SimpleCrossingS9N2-v0", "navix:Navix-SimpleCrossingS9N2-v0"
)
registry.register_alias(
    "Navix-SimpleCrossingS9N3-v0", "navix:Navix-SimpleCrossingS9N3-v0"
)
registry.register_alias(
    "Navix-SimpleCrossingS11N5-v0", "navix:Navix-SimpleCrossingS11N5-v0"
)
registry.register_alias(
    "Navix-Dynamic-Obstacles-5x5-v0", "navix:Navix-Dynamic-Obstacles-5x5-v0"
)
registry.register_alias(
    "Navix-Dynamic-Obstacles-5x5-Random-v0",
    "navix:Navix-Dynamic-Obstacles-5x5-Random-v0",
)
registry.register_alias(
    "Navix-Dynamic-Obstacles-6x6-v0", "navix:Navix-Dynamic-Obstacles-6x6-v0"
)
registry.register_alias(
    "Navix-Dynamic-Obstacles-6x6-Random-v0",
    "navix:Navix-Dynamic-Obstacles-6x6-Random-v0",
)
registry.register_alias(
    "Navix-Dynamic-Obstacles-8x8-v0", "navix:Navix-Dynamic-Obstacles-8x8-v0"
)
registry.register_alias(
    "Navix-Dynamic-Obstacles-16x16-v0", "navix:Navix-Dynamic-Obstacles-16x16-v0"
)
registry.register_alias("Navix-DistShift1-v0", "navix:Navix-DistShift1-v0")
registry.register_alias("Navix-DistShift2-v0", "navix:Navix-DistShift2-v0")
registry.register_alias("Navix-GoToDoor-5x5-v0", "navix:Navix-GoToDoor-5x5-v0")
registry.register_alias("Navix-GoToDoor-6x6-v0", "navix:Navix-GoToDoor-6x6-v0")
registry.register_alias("Navix-GoToDoor-8x8-v0", "navix:Navix-GoToDoor-8x8-v0")

# Craftax envs
registry.register_alias(
    "Craftax-Classic-Symbolic-v1", "craftax:Craftax-Classic-Symbolic-v1"
)
registry.register_alias(
    "Craftax-Classic-Pixels-v1", "craftax:Craftax-Classic-Pixels-v1"
)
registry.register_alias("Craftax-Symbolic-v1", "craftax:Craftax-Symbolic-v1")
registry.register_alias("Craftax-Pixels-v1", "craftax:Craftax-Pixels-v1")
# registry.register_alias(
#     "Craftax-Symbolic-AutoReset-v1", "craftax:Craftax-Symbolic-AutoReset-v1"
# )
# registry.register_alias(
#     "Craftax-Pixels-AutoReset-v1", "craftax:Craftax-Pixels-AutoReset-v1"
# )
