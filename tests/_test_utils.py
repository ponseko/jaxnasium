from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _consts import AGENT_MIN_CONFIG, SKIP_AGENT_ENVS, SKIP_ENVS

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
            for env_id in registry_envs_for_package(suite):
                skip_envs[env_id] = value
    return skip_envs


def registry_envs_for_package(package: str) -> list[str]:
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


def _assert_equal_pytrees(reference, candidate, name: str = " ") -> None:
    assert jax.tree.structure(reference) == jax.tree.structure(candidate), (
        f"Tree structure mismatch for {name}: {jax.tree.structure(reference)} != {jax.tree.structure(candidate)}"
    )
    for ref_leaf, cand_leaf in zip(
        jax.tree.leaves(reference), jax.tree.leaves(candidate), strict=True
    ):
        ref_shape, ref_dtype = _shape_dtype(ref_leaf)
        cand_shape, cand_dtype = _shape_dtype(cand_leaf)
        assert ref_dtype == cand_dtype, (
            f"Tree dtype mismatch for {name}: {ref_dtype} != {cand_dtype}"
        )
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
        skip_envs = get_skip_envs(SKIP_ENVS)
        if env_id in skip_envs:
            pytest.skip(f"Skipping {env_id}: {skip_envs[env_id]}")
        try:
            env = jym.make(env_id)
        except ImportError as e:
            pytest.skip(f"Skipping {env_id} due to ImportError: {e}")
    else:
        env_id = "_"

    if flatten_obs:
        env = jym.wrappers.FlattenObservationWrapper(env)

    reset_key, action_key, step_key = jax.random.split(SEED, 3)

    if run_env:
        _obs, reset_state = env.reset(reset_key)
        timestep, _state = env.step(
            step_key, reset_state, env.sample_action(action_key)
        )
        assert jym.ORIGINAL_OBSERVATION_KEY in timestep.info, (
            f"ORIGINAL_OBSERVATION_KEY not in timestep.info for {env_id}"
        )

    if env_id in get_skip_envs(SKIP_AGENT_ENVS):
        return

    reference = jax.eval_shape(env.sample_action, action_key)
    for alg_cls in get_valid_test_algs(env):
        alg: RLAlgorithm = alg_cls()  # type: ignore[operator]
        action = _run_agent_on_env(env, alg, SEED)
        _assert_equal_pytrees(
            reference,
            action,
            f"sampled_action != agent_action for {type(alg).__name__} on {env_id}",
        )
