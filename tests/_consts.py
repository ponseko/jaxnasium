import jax

from jaxnasium.algorithms import DQN, PPO, PQN, SAC  # noqa: F401

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

DISCRETE_ALGS = [PPO, PQN, DQN]
CONTINUOUS_ALGS = [PPO]

PPO_MIN_CONFIG = {
    "num_envs": 2,
    "num_epochs": 1,
    "num_minibatches": 1,
    "total_timesteps": 1_000,
    "log_function": None,
}

SAC_CONTINUOUS_CONFIG = {
    "total_timesteps": 1_000_000,
    "num_envs": 8,
    "learning_rate": 0.003,
    "anneal_learning_rate": True,
    "update_every": 64,
    "batch_size": 512,
    "target_entropy_scale": 1.5,
    "anneal_entropy_scale": 0.1,
    "replay_buffer_size": 500_000,
    "normalize_rewards": True,
    "normalize_observations": False,
    "actor_kwargs": {
        "hidden_sizes": (128, 128),
    },
    "critic_kwargs": {
        "hidden_sizes": (128, 128),
    },
    "log_function": None,
}

CLASSIC_CONTROL_ENVS = [
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
]

GYMNAX_TEST_ENVS = [
    "gymnax:CartPole-v1",
    "gymnax:Acrobot-v1",
    "Breakout-MinAtar",
    "Catch-bsuite",
    "FourRooms-misc",
]
JUMANJI_TEST_ENVS = ["Snake-v1", "Game2048-v1", "Cleaner-v0", "Maze-v0"]
BRAX_TEST_ENVS = [
    "ant",
    "halfcheetah",
    "humanoid",
    "inverted_double_pendulum",
    "walker2d",
]
