from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field, fields, replace
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
        `algorithm`: `RLAlgorithm` instance or name (e.g. `"PPO"`).
        `batch_size`: the batch size to `jax.lax.map` in case multiple seeds are given.
        `num_evaluations`: Episodes to evaluate the trained agent over.
        `return_train_metrics`: Also return the per-iteration training metrics.
        `save_path`: Where to write the results, or `None` (the default) to write
            nothing and leave saving to the caller.
            This will save to <save_path>/<algorithm>_<env>_<digest> and writes 4 files:
            - `results.npy`: the evaluation results, shape (num_seeds, num_evaluations)
            - `train_curve.npy`: the training metrics, shape (num_seeds, num_iterations, ...)
            - `input_parameters.json`: the input parameters to the evaluation
            - `full_parameters.json`: the full parameters used for the evaluation.
        `save_as_zip`: If True, while save_path is not None, will zip the results folder
            and write it to `save_path`.

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
    algorithm: RLAlgorithm | str = "PPO"
    batch_size: int | None = None
    num_evaluations: int = 50
    return_train_metrics: bool = False
    save_path: str | Path | None = None
    save_as_zip: bool = False
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
        algorithm = _get_algorithm(
            hyperparameters.pop("algorithm", self.algorithm), hyperparameters
        )
        return AlgorithmEvaluationConfig(
            algorithm, env, env_name, seed, hyperparameters, dict(self._labels)
        )

    def __call__(self, **kwargs: Any) -> Any:
        """Train and evaluate, once per seed.

        Returns a dict {"evaluation": evaluation_returns, "train_metrics": train_metrics} if
        `return_train_metrics` is True, otherwise just the evaluation returns. With
        several seeds, every leaf gains a leading axis over them.
        """
        config = self._create_config(kwargs)

        def run(key: PRNGKeyArray) -> Any:
            _log("Running AlgorithmEvaluation")
            _log(f"Algorithm: {type(config.algorithm).__name__}")
            _log(f"Environment: {config.env_name}")
            _log(f"Hyperparameters: {config.hyperparameters}")
            train_key, eval_key = jax.random.split(key)
            agent, train_metrics = config.algorithm.train(train_key, config.env)
            evaluation = agent.evaluate(
                eval_key, config.env, num_eval_episodes=self.num_evaluations
            )
            if not self.return_train_metrics:
                return evaluation
            return {"evaluation": evaluation, "train_metrics": train_metrics}

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

        result = fn()

        if self.save_path is not None:
            result = jax.block_until_ready(result)
            self._save(config, result, time.time() - start_time)
        return result

    def _save(self, config: AlgorithmEvaluationConfig, result: Any, duration: float):
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
            (run_dir / "full_parameters.json").write_text(
                json.dumps({**context, "duration": duration}, indent=2)
            )

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
            "seed": _jsonify(jax.random.key_data(config.seed)),
            "num_evaluations": self.num_evaluations,
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
        return f"AlgorithmEvaluation({algorithm} on {_get_env(self.env)[1]})"
