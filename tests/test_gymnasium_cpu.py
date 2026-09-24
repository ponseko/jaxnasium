import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxnasium as jym
from jaxnasium.algorithms import PPO
from jaxnasium.wrappers import GymnasiumWrapper

gymnasium = pytest.importorskip("gymnasium")


NUM_ENVS = 2
SEED = jax.random.PRNGKey(0)


def _make_venv(
    env_id: str = "CartPole-v1", num_envs: int = NUM_ENVS, autoreset_mode=None
):
    return gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make(env_id)] * num_envs,
        autoreset_mode=autoreset_mode or gymnasium.vector.AutoresetMode.SAME_STEP,
    )


def _make_env(num_envs: int = NUM_ENVS) -> GymnasiumWrapper:
    return GymnasiumWrapper(_make_venv(num_envs=num_envs))


def _rollout(env: GymnasiumWrapper, length: int = 30):
    """A full rollout under jit, as an algorithm would run it."""

    def env_step(carry, _):
        key, _obs, state = carry
        key, action_key, step_key = jax.random.split(key, 3)
        action = env.sample_action(jax.random.split(action_key, env._internal_num_envs))  # type: ignore[reportOptionalMemberAccess]
        timestep, state = env.step(
            jax.random.split(step_key, env._internal_num_envs),  # type: ignore[reportOptionalMemberAccess]
            state,
            action,
        )
        return (key, timestep.observation, state), timestep

    obs, state = env.reset(jax.random.split(SEED, env._internal_num_envs))  # type: ignore[reportOptionalMemberAccess]
    return jax.jit(lambda: jax.lax.scan(env_step, (SEED, obs, state), None, length))()


def _fixed_action_rollout(venv, steps: int = 60):
    """Steps a wrapped vector env with a fixed action sequence, returning numpy arrays."""
    env = GymnasiumWrapper(venv)
    n = env._internal_num_envs
    actions = np.random.default_rng(0).integers(0, 2, size=(steps, n)).astype(np.int32)  # type: ignore[reportArgumentType]
    _obs, state = env.reset(jax.random.split(SEED, n))  # type: ignore[reportArgumentType]
    step = jax.jit(env.step)
    out = {"obs": [], "terminal_obs": [], "reward": [], "done": []}
    for t in range(steps):
        timestep, state = step(SEED, state, jnp.asarray(actions[t]))
        out["obs"].append(timestep.observation)
        out["terminal_obs"].append(timestep.info[jym.ORIGINAL_OBSERVATION_KEY])
        out["reward"].append(timestep.reward)
        out["done"].append(timestep.terminated | timestep.truncated)
    return {k: np.stack(v) for k, v in out.items()}


def test_vectorized_env_is_batched():
    env = _make_env()
    # Spaces describe a single environment; the batch axis comes from the environment.
    assert env.observation_space.shape == (4,)
    assert isinstance(env.action_space, jym.Discrete)

    keys = jax.random.split(SEED, NUM_ENVS)
    obs, state = env.reset(keys)
    assert obs.shape == (NUM_ENVS, 4) and obs.dtype == jnp.float32
    action = env.sample_action(keys)
    assert action.shape == (NUM_ENVS,)
    assert env.sample_observation(keys).shape == (NUM_ENVS, 4)

    timestep, state = jax.jit(env.step)(keys, state, action)
    assert timestep.observation.shape == (NUM_ENVS, 4)
    assert timestep.reward.shape == timestep.terminated.shape == (NUM_ENVS,)
    assert timestep.reward.dtype == jnp.float32
    assert timestep.terminated.dtype == timestep.truncated.dtype == jnp.bool_


def test_scan_rollout_auto_resets():
    _carry, timestep = _rollout(_make_env(), length=60)
    done = timestep.terminated | timestep.truncated
    assert done.any(), "a random CartPole policy should terminate within 60 steps"

    # On a done step the observation is already the one after the auto-reset, while
    # the info holds the terminal one.
    terminal_obs = timestep.info[jym.ORIGINAL_OBSERVATION_KEY]
    step, env_idx = (int(i) for i in jnp.argwhere(done)[0])
    assert not jnp.allclose(
        timestep.observation[step, env_idx], terminal_obs[step, env_idx]
    )
    # A reset CartPole starts near zero; a terminal one has a large pole angle
    assert abs(float(terminal_obs[step, env_idx][2])) > 0.2


def test_continuous_action_space():
    env = GymnasiumWrapper(_make_venv("Pendulum-v1"))
    assert isinstance(env.action_space, jym.Box)
    _carry, timestep = _rollout(env, length=5)
    assert timestep.observation.shape == (5, NUM_ENVS, 3)
    assert timestep.reward.shape == (5, NUM_ENVS)


def test_plain_environment():
    env = GymnasiumWrapper(gymnasium.make("CartPole-v1"))
    assert not env.is_vectorized

    def env_step(carry, _):
        key, _obs, state = carry
        key, action_key, step_key = jax.random.split(key, 3)
        timestep, state = env.step(step_key, state, env.sample_action(action_key))
        return (key, timestep.observation, state), timestep

    obs, state = env.reset(SEED)
    assert obs.shape == (4,)
    _carry, timestep = jax.jit(
        lambda: jax.lax.scan(env_step, (SEED, obs, state), None, 120)
    )()
    # No batch axis, and auto-resets done by the wrapper itself
    assert timestep.observation.shape == (120, 4) and timestep.reward.shape == (120,)
    done = timestep.terminated | timestep.truncated
    assert done.any(), "a random policy should terminate within 120 steps"
    step = int(jnp.argwhere(done)[0][0])
    terminal_obs = timestep.info[jym.ORIGINAL_OBSERVATION_KEY][step]
    assert not jnp.allclose(timestep.observation[step], terminal_obs)
    assert abs(float(terminal_obs[2])) > 0.2  # terminal pole angle, not a fresh reset


@pytest.mark.parametrize("mode", ["NEXT_STEP", "DISABLED"])
def test_other_autoreset_modes_match_same_step(mode):
    # The wrapper emulates SAME_STEP by resetting finished environments itself, so with the
    # same seeds and actions the trajectories must be identical.
    AutoresetMode = gymnasium.vector.AutoresetMode
    reference = _fixed_action_rollout(
        _make_venv(autoreset_mode=AutoresetMode.SAME_STEP)
    )
    emulated = _fixed_action_rollout(_make_venv(autoreset_mode=AutoresetMode[mode]))

    assert reference["done"].any(), "the rollout should include episode ends"
    for key in reference:
        np.testing.assert_array_equal(emulated[key], reference[key], err_msg=key)


def test_rejects_a_vector_env_that_ignores_masked_resets():
    # Gymnasium's own CartPoleVectorEnv (what `make_vec` picks by default) auto-resets on
    # the next step and resets *all* environments when given a reset mask.
    venv = gymnasium.make_vec("CartPole-v1", num_envs=NUM_ENVS)
    with pytest.raises(ValueError, match="cannot reset individual environments"):
        GymnasiumWrapper(venv)


def test_envpool():
    envpool = pytest.importorskip("envpool")
    # EnvPool resets on the next step and returns only the reset environments from a
    # masked reset, which exercises the other branch of the manual auto-reset.
    rollout = _fixed_action_rollout(
        envpool.make_gymnasium("CartPole-v1", num_envs=NUM_ENVS, seed=0), steps=200
    )
    done = rollout["done"]
    assert done.any()
    # A missed or doubled reset would show up as a zero reward (CartPole pays 1 per step)
    assert np.all(rollout["reward"] == 1.0)
    assert np.all(np.abs(rollout["obs"][done]) <= 0.05)  # fresh episodes after a reset
    terminal = rollout["terminal_obs"][done]
    assert np.all((np.abs(terminal[:, 0]) > 2.4) | (np.abs(terminal[:, 2]) > 0.2095))


def test_train_and_evaluate():
    env = _make_env()
    trainer = PPO(
        num_envs=NUM_ENVS, total_timesteps=256, num_steps=16, log_function=None
    )
    agent, metrics = trainer.train(SEED, env)
    assert jax.tree.leaves(metrics)

    # Evaluation needs an environment without a batch axis, as for a JAX environment.
    eval_env = GymnasiumWrapper(gymnasium.make("CartPole-v1"))
    returns = trainer.evaluate(SEED, agent, eval_env, num_eval_episodes=2)
    assert returns.shape == (2,)
    assert (returns > 0).all()


def test_check_env_requires_a_matching_batch():
    with pytest.raises(ValueError, match="num_envs"):
        PPO(
            num_envs=NUM_ENVS + 1, total_timesteps=128, log_function=None
        ).__check_env__(_make_env(), vectorized=True)
    with pytest.raises(ValueError, match="vectorized on the Gymnasium side"):
        PPO(num_envs=NUM_ENVS, total_timesteps=128, log_function=None).__check_env__(
            GymnasiumWrapper(gymnasium.make("CartPole-v1")), vectorized=True
        )


def test_evaluate_rejects_a_batched_environment():
    trainer = PPO(num_envs=NUM_ENVS, total_timesteps=256, log_function=None)
    agent = trainer.init_agent(SEED, _make_env())
    with pytest.raises(ValueError, match="vectorized GymnasiumWrapper"):
        trainer.evaluate(SEED, agent, _make_env(), num_eval_episodes=1)


def test_rejects_being_vectorized_again():
    with pytest.raises(ValueError, match="already vectorized"):
        jym.VecEnvWrapper(_make_env())
    with pytest.raises(ValueError, match="cannot be vectorized"):
        jym.VecEnvWrapper(GymnasiumWrapper(gymnasium.make("CartPole-v1")))


def test_vmapping_training_runs_raises():
    # Vmapped copies (e.g. seeds) would all step the same Python environments; the
    # ordered callbacks turn that into an error instead of silently wrong results.
    env = _make_env()
    trainer = PPO(num_envs=NUM_ENVS, total_timesteps=64, num_steps=8, log_function=None)
    with pytest.raises(ValueError, match="vmap"):
        jax.vmap(lambda k: trainer.train(k, env)[1])(jax.random.split(SEED, 2))


def test_dict_and_tuple_gymasium_space_conversion():
    ## uses a nested spaces of tuples and dicts
    class Env(gymnasium.Env):
        def __init__(self):
            self.observation_space = gymnasium.spaces.Box(
                -1, 1, shape=(1,), dtype=np.float32
            )
            self.action_space = gymnasium.spaces.Dict(
                {
                    "move": gymnasium.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                    "choice": gymnasium.spaces.Tuple(
                        (
                            gymnasium.spaces.Discrete(3),
                            gymnasium.spaces.MultiDiscrete([2, 2]),
                        )
                    ),
                }
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    space = GymnasiumWrapper(Env()).action_space
    assert isinstance(space["move"], jym.Box) and space["move"].shape == (2,)  # type: ignore[reportOptionalMemberAccess]
    discrete, multi = space["choice"]  # type: ignore[reportOptionalMemberAccess]
    assert isinstance(discrete, jym.Discrete) and int(discrete.n) == 3
    assert isinstance(multi, jym.MultiDiscrete)
    np.testing.assert_array_equal(multi.nvec, [2, 2])
