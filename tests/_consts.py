from jaxnasium.algorithms import DQN, PPO, PQN, SAC
from jaxnasium.algorithms.architectures import MLP
from jaxnasium.wrappers import Wrapper

DISCRETE_ALGS = [PPO, PQN, DQN, SAC]
CONTINUOUS_ALGS = [PPO, SAC]


LEARN_CONFIG_DISCRETE: dict[type, dict] = {
    PPO: {"total_timesteps": 100_000},
    PQN: {"total_timesteps": 100_000},
    DQN: {"total_timesteps": 100_000},
    SAC: {"total_timesteps": 100_000, "critics_num_updates": 8, "batch_size": 256},
}

LEARN_CONFIG_CONTINUOUS: dict[type, dict] = {
    PPO: {"total_timesteps": 100_000},
    SAC: {"total_timesteps": 100_000, "critics_num_updates": 8, "batch_size": 256},
}

# Optional wrappers applied before environment use.
ENV_WRAPPERS: dict[str, list[type[Wrapper]]] = {}

AGENT_MIN_CONFIG = {
    "num_envs": 1,
    "total_timesteps": 512,
    "warmup_steps": 16,
    "log_function": None,
    "normalize_observations": False,
    "normalize_rewards": False,
    "actor_kwargs": {"body": MLP.with_params(hidden_sizes=(8,))},
    "critic_kwargs": {"body": MLP.with_params(hidden_sizes=(8,))},
}


# Skipped due some bugs in the environments
SKIP_ENVS: dict[str, str] = {
    "SimpleBandit-bsuite": "bug on reset: https://github.com/RobertTLange/gymnax/issues/110",
    "storm": "Bug, diff obs space from rest of envs, but also does not seem to align with the step output (?)",
    "storm_2p": "Bug, diff obs space from rest of envs, but also does not seem to align with the step output (?)",
    "storm_np": "Bug, diff obs space from rest of envs, but also does not seem to align with the step output (?)",
    "Tetris-v0": "Uses invalid jnp.clip call",
    "NoisyStatelessMetaCartPole": "Not runnable in newer jax versions > 0.10",
}

# Skipped for some limatation in default configuration of algorithms
SKIP_AGENT_ENVS: dict[str, str] = {
    "BinPack-v2": "Action masking not supported due to actions being conditionally dependent",
    "Tetris-v0": "Action masking not supported due to actions being conditionally dependent",
    "Minesweeper-v0": "Action masking not supported due to actions being conditionally dependent",
    "Sudoku-v0": "Action masking not supported due to actions being conditionally dependent",
    "Sudoku-very-easy-v0": "Action masking not supported due to actions being conditionally dependent",
    "FlatPack-v0": "Action masking not supported due to actions being conditionally dependent",
    "RubiksCube-v0": "Heterogeous MultiDiscrete action space",
    "RubiksCube-partly-scrambled-v0": "Heterogeous MultiDiscrete action space",
}
