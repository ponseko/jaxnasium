from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _consts import AGENT_MIN_CONFIG, KNOWN_FAILURES, SKIP_AGENT_ENVS

import jaxnasium as jym
from jaxnasium.algorithms import DQN, PPO, PQN, SAC, RLAgent, RLAlgorithm
from jaxnasium.algorithms.core import Transition

SEED = jax.random.PRNGKey(0)


def get_skip_envs(env_reason_dict: dict[str, str]) -> dict[str, str]:
    """Returns the environments that are skipped due to bugs or limitations"""
    skip_envs = env_reason_dict.copy()
    for pattern, value in env_reason_dict.items():
        if "_SUITE_" in pattern:
            suite = pattern.split(":")[1]
            for env_id in _registry_env_ids(suite):
                skip_envs[env_id] = value
    return skip_envs


def registry_envs_for_package(package: str) -> list:
    """Sorted registry aliases of a package, as test params with id `package/env_id`.
    Known failures are strict xfails."""
    known_failures = get_skip_envs(KNOWN_FAILURES)
    params = []
    for env_id in _registry_env_ids(package):
        reason = known_failures.get(env_id)
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        params.append(pytest.param(env_id, id=f"{package}/{env_id}", marks=marks))
    return params


def _registry_env_ids(package: str) -> list[str]:
    """Return sorted registry aliases that resolve to an external package."""
    if package == "jaxnasium":
        return sorted(jym.registry._environments.keys())
    prefix = f"{package}:"
    return sorted(
        alias
        for alias, target in jym.registry._aliases.items()
        if target.startswith(prefix)
    )


def _shape_dtype(leaf) -> tuple[tuple[int, ...], Any]:
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return tuple(leaf.shape), leaf.dtype
    array = jnp.asarray(leaf)
    return array.shape, array.dtype


def _assert_equal_pytrees(
    reference, candidate, name: str = " ", *, strict_dtype: bool = True
) -> None:
    """Asserts equal structure, shapes and dtypes. With `strict_dtype=False` dtype mismatches only warn."""
    assert jax.tree.structure(reference) == jax.tree.structure(candidate), (
        f"Tree structure mismatch for {name}: {jax.tree.structure(reference)} != {jax.tree.structure(candidate)}"
    )
    dtype_mismatches = []
    for (path, ref_leaf), cand_leaf in zip(
        jax.tree_util.tree_leaves_with_path(reference),
        jax.tree.leaves(candidate),
        strict=True,
    ):
        ref_shape, ref_dtype = _shape_dtype(ref_leaf)
        cand_shape, cand_dtype = _shape_dtype(cand_leaf)
        if ref_dtype != cand_dtype:
            mismatch = (
                f"{jax.tree_util.keystr(path) or '<root>'}: {ref_dtype} != {cand_dtype}"
            )
            assert not strict_dtype, f"Tree dtype mismatch for {name}: {mismatch}"
            dtype_mismatches.append(mismatch)
        if (ref_shape == () and cand_shape == (1,)) or (
            ref_shape == (1,) and cand_shape == ()
        ):
            warnings.warn(
                f"Detected inconsistent scalar shape ({ref_shape} and {cand_shape})"
                f"This is often caused by a mismatch in the observation space and the actual"
                f"observation returned by step()/reset(). This will not necessarily cause issues,"
                f" but should likely be fixed in the environment implementation. "
            )
            continue
        assert ref_shape == cand_shape, (
            f"Tree shape mismatch for {name}: {ref_shape} != {cand_shape}"
        )
    if dtype_mismatches:
        warnings.warn(
            f"Declared dtype != actual dtype for {name} ({'; '.join(dtype_mismatches)}). "
            "This will not necessarily cause issues, but should likely be fixed in the "
            "environment's space definition."
        )


def _has_continuous_action_space(env: jym.Environment) -> bool:
    spaces = jax.tree.leaves(env.action_space)
    for space in spaces:
        if hasattr(space, "n") or hasattr(space, "nvec"):
            continue
        return True
    return False


def get_valid_test_algs(env: jym.Environment) -> list[type[RLAlgorithm]]:
    algs = [
        partial[PPO](PPO, **AGENT_MIN_CONFIG),
        partial[SAC](SAC, **AGENT_MIN_CONFIG, critics_num_updates=1),
    ]
    if not _has_continuous_action_space(env):
        NO_ACTOR_AGENT_MIN_CONFIG = {
            k: v for k, v in AGENT_MIN_CONFIG.items() if k != "actor_kwargs"
        }
        algs.extend(
            [
                partial[DQN](DQN, **NO_ACTOR_AGENT_MIN_CONFIG, num_updates=1),
                partial[PQN](PQN, **NO_ACTOR_AGENT_MIN_CONFIG),
            ]
        )
    return algs


def _make_dummy_update_batch(
    agent: RLAgent, env: jym.Environment, key: jax.Array
) -> Transition:
    obs_key, action_key, next_obs_key = jax.random.split(key, 3)

    observation = env.sample_observation(obs_key)
    next_observation = env.sample_observation(next_obs_key)
    action = agent.get_action(action_key, observation)

    if env.multi_agent:
        reward = jym.tree.map_one_level(lambda x: jnp.array(0.0), action)
    else:
        reward = jnp.array(0.0)

    batch = Transition(
        observation=observation,
        action=action,
        reward=reward,
        terminated=jym.tree.zeros_like(reward, dtype=bool),
        truncated=jym.tree.zeros_like(reward, dtype=bool),
        next_observation=next_observation,
        return_=jym.tree.zeros_like(reward),
        value=jym.tree.ones_like(reward),
        log_prob=jym.tree.zeros_like(reward),
        advantage=jym.tree.zeros_like(reward),
    )

    # the batch is now a single transition, for perhaps multiple agents
    # make it a batch:
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (4, *x.shape)), batch)

    return batch


def _run_agent_update(agent: Any, batch: Transition, key: jax.Array):
    if isinstance(agent.trainer, SAC):
        jax.eval_shape(agent.update_critics_params, key, batch)
        jax.eval_shape(agent.update_actor_params, key, batch)
    else:
        jax.eval_shape(agent.update_params, batch)


def _run_agent_on_env(
    env: jym.Environment,
    alg: RLAlgorithm,
    key: jax.Array,
    *,
    include_update: bool = True,
) -> Any:
    def _test(key: jax.Array):
        init_key, obs_key, action_key, update_key = jax.random.split(key, 4)
        agent: RLAgent = alg.init_agent(init_key, env)
        observation = env.sample_observation(obs_key)
        action = agent.get_action(action_key, observation)
        if include_update:
            batch = _make_dummy_update_batch(agent, env, update_key)
            _run_agent_update(agent, batch, update_key)
        return action

    return eqx.filter_eval_shape(_test, key)


ROLLOUT_NUM_ENVS = 2
ROLLOUT_NUM_STEPS = 4


def _rollout(env: jym.Environment, key: jax.Array, num_steps: int):
    """Reset and step `env` with random actions, vectorized and scanned (same as algorithms would)."""
    venv = jym.VecEnvWrapper(env)
    reset_key, scan_key = jax.random.split(key)
    obs, state = venv.reset(jax.random.split(reset_key, ROLLOUT_NUM_ENVS))

    def step(state, key):
        action_key, step_key = jax.random.split(key)
        action_keys = jax.random.split(action_key, ROLLOUT_NUM_ENVS)
        action = jax.vmap(env.sample_action)(action_keys)
        step_keys = jax.random.split(step_key, ROLLOUT_NUM_ENVS)
        timestep, state = venv.step(step_keys, state, action)
        return state, timestep

    _, timesteps = jax.lax.scan(step, state, jax.random.split(scan_key, num_steps))
    return obs, timesteps


def _assert_per_agent_scalars(
    env: jym.Environment, tree, name: str, *, allow_shared: bool = False
) -> None:
    """`tree` holds one scalar per agent (or a single scalar shared by all agents)."""
    structure = jax.tree.structure(tree)
    allowed = (
        {env.agent_structure, jax.tree.structure(0)}
        if allow_shared
        else {env.agent_structure}
    )
    assert structure in allowed, (
        f"{name} structure {structure} != agent structure {env.agent_structure}"
    )
    for leaf in jax.tree.leaves(tree):
        assert np.shape(leaf) in ((), (1,)), f"{name} is not a scalar: {np.shape(leaf)}"


def _check_env_rollout(env: jym.Environment, env_id: str) -> None:
    """Actually runs the env as a single compiled program and checks if everything
    is present and finite.
    Checks what is returns matches the declared spaces. Other agent checks depend
    on these spaces being correct.
    """

    # Do rollout and put on host so checks can happen w/ numpy to prevent eager compilations
    rollout = eqx.filter_jit(_rollout)(env, SEED, ROLLOUT_NUM_STEPS)
    obs, timesteps = jax.device_get(rollout)

    # strip (step, env) batch dimensions:
    obs = jax.tree.map(lambda x: x[0], obs)
    first, last = (jax.tree.map(lambda x, i=i: x[i, 0], timesteps) for i in (0, -1))

    # declaration can in theory be off, will not break learning; so strict_dtype=False
    reference = jax.eval_shape(env.sample_observation, SEED)
    _assert_equal_pytrees(
        reference, obs, f"reset observation of {env_id}", strict_dtype=False
    )

    # env must be consistent for `lax.scan` to not break:
    _assert_equal_pytrees(obs, last.observation, f"step observation of {env_id}")
    assert jym.ORIGINAL_OBSERVATION_KEY in first.info, (
        f"ORIGINAL_OBSERVATION_KEY not in timestep.info for {env_id}"
    )
    _assert_equal_pytrees(
        obs,
        first.info[jym.ORIGINAL_OBSERVATION_KEY],
        f"info[ORIGINAL_OBSERVATION_KEY] of {env_id}",
    )
    _assert_per_agent_scalars(env, first.reward, f"reward of {env_id}")
    for field in ("terminated", "truncated"):  # The algorithms broadcast a shared done
        _assert_per_agent_scalars(
            env, getattr(first, field), f"{field} of {env_id}", allow_shared=True
        )

    rewards = jax.tree.leaves(timesteps.reward)
    assert all(np.isfinite(r).all() for r in rewards), f"Non-finite reward in {env_id}"
    observations = [*jax.tree.leaves(obs), *jax.tree.leaves(timesteps.observation)]
    assert not any(np.isnan(o).any() for o in observations), (
        f"NaN observation in {env_id}"
    )


def check_env(
    env: jym.Environment | str,
    *,
    flatten_obs: bool = False,
    run_env: bool = True,
) -> None:
    """Checks if the action space samples, the env steps and resets and whether each
    algorithm correctly builds and can train on it.
    """
    if isinstance(env, str):
        env_id = env
        try:
            env = jym.make(env_id)
        except ImportError as e:
            pytest.skip(f"Skipping {env_id} due to ImportError: {e}")
    else:
        env_id = "_"

    if flatten_obs:
        env = jym.wrappers.FlattenObservationWrapper(env)

    if run_env:
        _check_env_rollout(env, env_id)

    if env_id in get_skip_envs(SKIP_AGENT_ENVS):
        return

    reference = jax.eval_shape(env.sample_action, SEED)
    for alg_cls in get_valid_test_algs(env):
        alg: RLAlgorithm = alg_cls()  # type: ignore[operator]
        action = _run_agent_on_env(env, alg, SEED)
        _assert_equal_pytrees(
            reference,
            action,
            f"sampled_action != agent_action for {type(alg).__name__} on {env_id}",
        )
