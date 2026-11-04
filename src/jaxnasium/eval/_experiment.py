from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZIP_DEFLATED, ZipFile

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

import jaxnasium as jym

from ._sweep import _jsonify

if TYPE_CHECKING:
    from jaxnasium.algorithms import RLAlgorithm


def _log(*text: str) -> None:
    try:
        sys.stderr.write(" ".join(text) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _run_info() -> dict[str, Any]:
    """Where a run ran: the device, library versions, and the git commit of the working directory."""
    try:
        git = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = git.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "device": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "jaxnasium_version": jym.__version__,
        "git_commit": commit,
    }


def _get_algorithm(algorithm: RLAlgorithm | str, hyperparameters: dict) -> RLAlgorithm:
    if isinstance(algorithm, str):
        import jaxnasium.algorithms as jxalgs

        algorithm = algorithm.upper()
        try:
            cls = getattr(jxalgs, algorithm)
        except AttributeError as e:
            available = [
                name
                for name, obj in vars(jxalgs).items()
                if isinstance(obj, type) and issubclass(obj, jxalgs.RLAlgorithm)
            ]
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. Available: {available}"
            ) from e
        return cls(**hyperparameters)

    return replace(algorithm, **hyperparameters) if hyperparameters else algorithm


def _get_env(env: jym.Environment | str) -> tuple[jym.Environment, str]:
    """The environment, plus the name to record it under. Exlcuding wrappers"""
    if isinstance(env, str):
        return jym.make(env), env
    innermost = env
    while hasattr(innermost, "_env"):
        innermost = innermost._env  # type: ignore
    return env, type(innermost).__name__


@dataclass(frozen=True)
class AlgorithmEvaluationConfig:
    algorithm: RLAlgorithm
    env: jym.Environment
    env_name: str
    eval_env: jym.Environment
    eval_env_name: str
    seed: PRNGKeyArray  # always a typed key array; see `_create_config`
    hyperparameters: dict[str, Any]
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgorithmEvaluation:
    """Train and evaluate a `jaxnasium` algorithm on a given environment, for one
    seed or for many. A possible trial function for a `Sweep`.

    **Arguments**:
        `seed`: Key for training and evaluation. Pass multiple keys to repeat the
        training and evaluation for each (e.g. AlgorithmEvaluation(jax.random.split(key, 10))).
        Optional at init: it can also be supplied when calling (e.g. from a `Sweep`),
        but not both. A seed is required by the time the evaluation actually runs.
        `env`: Environment to train on, or a name to pass to `jym.make`.
        `eval_env`: Environment to evaluate on, or a name to pass to `jym.make`.
            Defaults to `None`, in which case evaluation happens on `env`.
        `algorithm`: `RLAlgorithm` instance or name (e.g. `"PPO"`).
        `batch_size`: the batch size to `jax.lax.map` in case multiple seeds are given.
        `num_evaluations`: Episodes to evaluate the trained agent over.
        `deterministic_policy`: Whether to act greedily (instead of sampling) during evaluation.
        `return_train_metrics`: Also return the per-iteration training metrics.
        `save_path`: Where to write the results, or `None` (the default) to write
            nothing and leave saving to the caller.
            This will save to <save_path>/<algorithm>_<env>_<digest> and writes 4 files:
            - `results.npy`: the evaluation results, shape (num_seeds, num_evaluations)
            - `train_curve.npy`: the training metrics, shape (num_seeds, num_iterations, ...)
            - `input_parameters.json`: the input parameters to the evaluation
            - `full_parameters.json`: the full parameters used for the evaluation, plus
              `_run_info()` (device, library versions, git commit) and wall-clock seconds:
              `compile_duration` (tracing and compiling), `run_duration` (training and
              evaluating all seeds) and `total_duration` (both, plus setup).
        `save_as_zip`: If True, while save_path is not None, will zip the results folder
            and write it to `save_path`.
        `evaluate_interval`: Also does an evaluation call every fraction of training
            iterations. Defaults to `None`: only at the end. Saved as
            `interval_results.npz` with the `iteration` and `timesteps` of each evaluation
            and the `evaluation` returns, shape (num_seeds, num_intervals, num_evaluations).
        `save_agent`: Also save the trained agent of every seed, as
            `agent_<seed index>.jaxon`.

    **Example**:
    ```python
    key = jax.random.key(0)
    evaluation = AlgorithmEvaluation(
        jax.random.split(key, 10), env="CartPole-v1", batch_size=5
    )
    sweep = Sweep(
        evaluation,
        GridSearch({"algorithm": ["PPO", "DQN"]}),
        RandomSearch(
            {"learning_rate": (1e-4, 1e-2, "log")},
            num_samples=64,
            seed=jax.random.PRNGKey(1),
        ),
    )
    # 2 algorithms * 64 learning rates = 128 jobs, each of 10 seeds, 5 at a time.
    ```
    """

    seed: PRNGKeyArray | None = None
    env: jym.Environment | str = "CartPole-v1"
    eval_env: jym.Environment | str | None = None
    algorithm: RLAlgorithm | str = "PPO"
    batch_size: int | None = None
    num_evaluations: int = 50
    deterministic_policy: bool = True
    return_train_metrics: bool = False
    save_path: str | Path | None = None
    save_as_zip: bool = False
    evaluate_interval: float | None = None
    save_agent: bool = False
    _labels: dict[str, Any] = field(default_factory=dict, repr=False)

    def with_labels(self, labels: dict[str, Any]) -> AlgorithmEvaluation:
        """Attach recorded labels for the saved config that won't be passed as call arguments."""
        return replace(self, _labels=dict(labels))

    def _create_config(self, kwargs: dict[str, Any]) -> AlgorithmEvaluationConfig:
        """Create a config from the given kwargs, filling in defaults from this instance."""
        hyperparameters = dict(kwargs)
        if "seed" in hyperparameters and self.seed is not None:
            raise ValueError(
                f"{type(self).__name__} was initialized with a seed, but its run was also given a seed. Choose one."
            )
        seed = hyperparameters.pop("seed", self.seed)
        if seed is None:
            raise ValueError(
                f"No seed was provided to {type(self).__name__}. Pass at init or to the run itself."
            )
        if not jnp.issubdtype(seed.dtype, jax.dtypes.prng_key):
            seed = jax.random.wrap_key_data(seed)
        env, env_name = _get_env(hyperparameters.pop("env", self.env))
        eval_env = hyperparameters.pop("eval_env", self.eval_env)
        if eval_env is None:
            eval_env, eval_env_name = env, env_name
        else:
            eval_env, eval_env_name = _get_env(eval_env)
        algorithm = _get_algorithm(
            hyperparameters.pop("algorithm", self.algorithm), hyperparameters
        )
        return AlgorithmEvaluationConfig(
            algorithm,
            env,
            env_name,
            eval_env,
            eval_env_name,
            seed,
            hyperparameters,
            dict(self._labels),
        )

    def __call__(self, **kwargs: Any) -> Any:
        """Train and evaluate, once per seed.

        Returns a dict {"evaluation": evaluation_returns, "train_metrics": train_metrics} if
        `return_train_metrics` is True, otherwise just the evaluation returns. With
        several seeds, every leaf gains a leading axis over them.
        """
        config = self._create_config(kwargs)
        algorithm, interval_evaluations = config.algorithm, {}
        if self.save_path is None and (self.evaluate_interval or self.save_agent):
            raise ValueError("`evaluate_interval` and `save_agent` need a `save_path`.")
        if self.evaluate_interval is not None:
            from jaxnasium.algorithms.core import Callback

            if algorithm.log_function is None:
                raise ValueError(
                    "`evaluate_interval` needs callbacks, which `log_function=None` "
                    "disables (use `callbacks=()` instead)."
                )
            evaluate = partial(self._evaluate_interval, config, interval_evaluations)
            callback = Callback(evaluate, self.evaluate_interval)
            algorithm = replace(algorithm, callbacks=(*algorithm.callbacks, callback))

        def run(key: PRNGKeyArray) -> Any:
            _log("Running AlgorithmEvaluation")
            _log(f"Algorithm: {type(config.algorithm).__name__}")
            _log(f"Environment: {config.env_name}")
            if config.eval_env_name != config.env_name:
                _log(f"Evaluation environment: {config.eval_env_name}")
            _log(f"Hyperparameters: {config.hyperparameters}")
            train_key, eval_key = jax.random.split(key)
            agent, train_metrics = algorithm.train(train_key, config.env)
            evaluation = agent.evaluate(
                eval_key,
                config.eval_env,
                num_eval_episodes=self.num_evaluations,
                deterministic_policy=self.deterministic_policy,
            )
            result = evaluation
            if self.return_train_metrics:
                result = {"evaluation": evaluation, "train_metrics": train_metrics}
            return (result, agent) if self.save_agent else result

        start_time = time.time()
        if config.seed.ndim == 0:

            def run_single(key: PRNGKeyArray) -> Any:
                return run(key)

            run_single.__name__ = "AlgorithmEvaluation"
            fn = jym.precompile(run_single, config.seed)
        else:

            def run_batch(keys: PRNGKeyArray):
                return jax.lax.map(run, keys, batch_size=self.batch_size)

            actual_batch_size = min(self.batch_size or 0, config.seed.shape[0])
            run_batch.__name__ = f"AlgorithmEvaluation:batched[{actual_batch_size}]"
            fn = jym.precompile(run_batch, config.seed)

        run_start_time = time.time()
        result, agent = fn() if self.save_agent else (fn(), None)

        if self.save_path is not None:
            result = jax.block_until_ready(result)
            durations = {
                "compile_duration": fn.compile_secs,
                "run_duration": time.time() - run_start_time,
                "total_duration": time.time() - start_time,
            }
            agents = ()
            if self.save_agent:
                if config.seed.ndim:
                    num_seeds = config.seed.shape[0]
                    agents = [
                        jax.tree.map(lambda x, i=i: x[i], agent)
                        for i in range(num_seeds)
                    ]
                else:
                    agents = [agent]
                agents = tuple(
                    a.with_hyperparams(callbacks=config.algorithm.callbacks)  # type: ignore
                    for a in agents
                )
            interval_results = None
            if self.evaluate_interval is not None:
                jax.effects_barrier()  # every interval evaluation has been collected
                interval_results = self._interval_results(config, interval_evaluations)
            self._save(config, result, durations, agents, interval_results)
        return result

    def _evaluate_interval(
        self, config, evaluations, key, runner_state, metrics, iteration, num_iterations
    ) -> None:
        """Training callback: evaluates the agent, and collects the evaluation on the host
        in `evaluations[iteration][train key]`. Skips the last iteration, as that is
        evaluated at the end."""

        def evaluate():
            evaluation = runner_state[0].evaluate(
                jax.random.fold_in(key, iteration),
                config.eval_env,
                num_eval_episodes=self.num_evaluations,
                deterministic_policy=self.deterministic_policy,
            )

            def store(key_data, iteration, evaluation):
                per_seed = evaluations.setdefault(int(iteration), {})
                per_seed[np.asarray(key_data).tobytes()] = evaluation

            jax.debug.callback(store, jax.random.key_data(key), iteration, evaluation)

        jax.lax.cond(iteration < num_iterations, evaluate, lambda: None)

    def _interval_results(self, config, evaluations) -> dict[str, Any]:
        """The evaluations collected by `_evaluate_interval`, seeds first."""
        # The train key of every seed, as split in `run`
        train_keys = jax.vmap(lambda k: jax.random.split(k)[0])(config.seed.reshape(-1))
        train_keys = np.asarray(jax.random.key_data(train_keys))
        iterations = np.array(sorted(evaluations))
        evaluation = np.stack(
            [[evaluations[i][k.tobytes()] for i in iterations] for k in train_keys]
        )
        algorithm = config.algorithm
        fraction = iterations / algorithm.num_iterations  # type: ignore
        return {
            "iteration": iterations,
            "timesteps": np.round(fraction * algorithm.total_timesteps).astype(int),  # type: ignore
            "evaluation": evaluation if config.seed.ndim else evaluation[0],
        }

    def _save(
        self,
        config: AlgorithmEvaluationConfig,
        result: Any,
        durations: dict[str, float],
        agents: tuple[Any, ...] = (),
        interval_results: dict[str, Any] | None = None,
    ):
        """Write one configuration's results under `save_path`. see class docstring for the layout."""
        context = self._context(config)
        # hash of the context. Same runs in the same folder overwritten.
        blob = re.sub(r"0x[0-9a-f]+", "0x", json.dumps(context, sort_keys=True))
        digest = hashlib.sha1(blob.encode()).hexdigest()
        run_name = f"{context['algorithm']}_{context['env']}_{digest[:16]}"
        save_root = Path(self.save_path or ".")
        save_root.mkdir(parents=True, exist_ok=True)

        def write_files(run_dir: Path) -> None:
            run_dir.mkdir(parents=True, exist_ok=True)
            evaluation = result["evaluation"] if self.return_train_metrics else result
            np.save(run_dir / "results.npy", np.asarray(evaluation))
            for i, agent in enumerate(agents):
                agent.save(run_dir / f"agent_{i}.jaxon")
            if interval_results is not None:
                np.savez(run_dir / "interval_results.npz", **interval_results)
            if self.return_train_metrics:
                metrics = result["train_metrics"]
                if isinstance(metrics, jax.Array):
                    np.save(run_dir / "train_curve.npy", np.asarray(metrics))
                else:  # a pytree of metrics rather than one reduced array
                    # Helpful for multi agent reward tracking
                    leaves = jax.tree_util.tree_flatten_with_path(metrics)[0]
                    np.savez(
                        run_dir / "train_curve.npz",
                        **{
                            jax.tree_util.keystr(path).lstrip("."): np.asarray(leaf)
                            for path, leaf in leaves
                        },  # type: ignore
                    )

            # The arguments as given, and then everything they resolved to.
            inputs = {k: v for k, v in context.items() if k != "algorithm_parameters"}
            (run_dir / "input_parameters.json").write_text(json.dumps(inputs, indent=2))
            full = {**context, **durations, "run_info": _run_info()}
            (run_dir / "full_parameters.json").write_text(json.dumps(full, indent=2))

        if self.save_as_zip:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / run_name
                write_files(run_dir)
                with ZipFile(save_root / f"{run_name}.zip", "w", ZIP_DEFLATED) as zf:
                    for file in run_dir.iterdir():
                        zf.write(file, arcname=f"{run_name}/{file.name}")
        else:
            write_files(save_root / run_name)

    def context(self, **kwargs: Any) -> dict[str, Any]:
        """
        JSON-ready metadata for a configuration, to pair with its result.
        Used to retrieve the full algorithm and env parameters that were used for this run.
        """
        return self._context(self._create_config(kwargs))

    def _context(self, config: AlgorithmEvaluationConfig) -> dict[str, Any]:
        input_parameters = {**config.hyperparameters, **config.labels}
        return {
            "algorithm": type(config.algorithm).__name__,
            "env": config.env_name,
            "eval_env": config.eval_env_name,
            "seed": _jsonify(jax.random.key_data(config.seed)),
            "num_evaluations": self.num_evaluations,
            "deterministic_policy": self.deterministic_policy,
            **(
                {"evaluate_interval": self.evaluate_interval}
                if self.evaluate_interval
                else {}
            ),
            "input_parameters": _jsonify(input_parameters),
            # Every field of the algorithm, not just the ones that were swept.
            "algorithm_parameters": _jsonify(
                {
                    f.name: getattr(config.algorithm, f.name)
                    for f in fields(config.algorithm)
                }
            ),
        }

    def __repr__(self) -> str:
        algorithm = (
            self.algorithm
            if isinstance(self.algorithm, str)
            else type(self.algorithm).__name__
        )
        env_name = _get_env(self.env)[1]
        if self.eval_env is not None:
            eval_env_name = _get_env(self.eval_env)[1]
            if eval_env_name != env_name:
                return (
                    f"AlgorithmEvaluation({algorithm} on {env_name}, "
                    f"evaluated on {eval_env_name})"
                )
        return f"AlgorithmEvaluation({algorithm} on {env_name})"
