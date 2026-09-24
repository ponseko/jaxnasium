from __future__ import annotations

import copy
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Protocol, overload

import jax

from ._probe import log_cost_estimate

logger = logging.getLogger(__name__)


def _log(*text: str) -> None:
    try:
        sys.stderr.write(" ".join(text) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


class ParameterSpaceSearch(Protocol):
    """Anything that produces a list of configurations."""

    def configs(self) -> list[dict[str, Any]]: ...


def _jsonify(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonify(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class SweepResult:
    """Simple container for the results of a sweep job.
    Most prominently, this contains the `result` and the `arguments` that produced it.

    **Attributes**:
        `start_time`: Unix time at which the job containing this run started.
        `end_time`: Unix time at which the job containing this run finished.
        `arguments`: The parameter values that produced this result. Branch
            parameters hold their *label*, not the sub-space they expanded into.
        `result`: Whatever the swept function returned, for this configuration.
    """

    start_time: float
    end_time: float
    arguments: dict[str, Any]
    result: Any

    @property
    def duration(self) -> float:
        """Wall-clock seconds spent on this job."""
        return self.end_time - self.start_time

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON, stringifying anything not natively serializable."""
        return json.dumps(
            {field.name: _jsonify(getattr(self, field.name)) for field in fields(self)},
            **kwargs,
        )

    def to_json_no_result(self, **kwargs: Any) -> str:
        """Serialize to JSON, stringifying anything not natively serializable, excluding the result."""
        return json.dumps(
            {
                field.name: _jsonify(getattr(self, field.name))
                for field in fields(self)
                if field.name != "result"
            },
            **kwargs,
        )


@dataclass
class SweepJob:
    """One dispatchable unit of work: a single call to the swept function.

    **Attributes**:
        `args`: The keyword arguments the swept function is called with.
        `labels`: Branch labels recorded in place of what they expanded into.
            Overlay these on `args` to get the human-readable configuration.
    """

    args: dict[str, Any]
    labels: dict[str, Any]

    @property
    def arguments(self) -> dict[str, Any]:
        """The input configuration as recorded"""
        return {**self.args, **self.labels}


def _merge(
    job: SweepJob,
    args: dict[str, Any] | None = None,
    labels: dict[str, Any] | None = None,
) -> SweepJob:
    """Add arguments (and branch labels) to `job`, rejecting duplicates."""
    merged = dict(job.args)
    for name, value in (args or {}).items():
        if name in merged:
            raise ValueError(
                f"Param {name!r} is set by more than one stage on the same path. "
                "Only sibling branches may reuse a parameter name."
            )
        merged[name] = value
    return SweepJob(merged, {**job.labels, **(labels or {})})


def _expand_space(space: Any, jobs: list[SweepJob]) -> list[SweepJob]:
    """Expand every job in `jobs` by a (sub-)space, returning what they grow into.

    A space is a `dict` of fixed keyword arguments, a `list`/`tuple` of spaces applied in order, or a search stage.
    """
    if isinstance(space, dict):
        return [_merge(job, args=space) for job in jobs]
    if isinstance(space, (list, tuple)):
        for item in space:
            jobs = _expand_space(item, jobs)
        return jobs

    # No dict, tuple or list; so this is a search stage (RandomSearch, GridSearch, ...)

    branches = {
        name: spec
        for name, spec in getattr(space, "params", {}).items()
        if isinstance(spec, dict)
    }
    configs = space.configs()

    expanded: list[SweepJob] = []
    for job in jobs:
        for config in configs:
            labels = {n: v for n, v in config.items() if n in branches}
            branched = [
                _merge(
                    job,
                    args={n: v for n, v in config.items() if n not in branches},
                    labels=labels,
                )
            ]
            for name, label in labels.items():
                sub_space = branches[name][label]
                if not isinstance(sub_space, (dict, list, tuple)) and not hasattr(
                    sub_space, "configs"
                ):
                    sub_space = {name: sub_space}
                branched = _expand_space(sub_space, branched)
            expanded.extend(branched)
    return expanded


class Sweep:
    """Build a parameter sweep over function `fn` from one or more stages of parameter
    space search, like `GridSearch`, `OneAtATimeSearch`, `RandomSearch`, or `SobolSearch`.

    For a single stage, `RandomSearch(...).sweep(fn, ...)` is equivalent.

    **Arguments**:
        `fn`: The function to sweep. Must accept all swept params as keywords.
            If it implements `with_labels(labels) -> fn`, Sweep calls that
            before each job so branch labels can be recorded (as
            [`AlgorithmEvaluation`][jaxnasium.eval.AlgorithmEvaluation] does).
        `*stages`: `GridSearch` / `OneAtATimeSearch` / `RandomSearch` / `SobolSearch` stages to nest.
        `print_cost_estimate`: After creation, print the cost estimate of the first job to provide a
        rough estimate of the memory requirements of a job in the sweep. This triggers a compilation, so
        adds some overhead to the creation of the sweep.

    **Attributes**:
        `fn`: Runs one `SweepJob` and returns its `SweepResult`.
        `jobs`: All jobs, to run in a loop or to dispatch by index (e.g. over a slurm job array).

    **Example**:
    ```python
    sweep = Sweep(
        train,
        GridSearch({"env_name": ["CartPole-v1", "Acrobot-v1"]}),
        RandomSearch(
            {"learning_rate": (1e-4, 1e-2, "log")},
            num_samples=64,
            seed=jax.random.PRNGKey(1),  # the same 64 lrs in both envs
        ),
    )
    len(sweep)  # 2 envs * 64 learning rates
    for run in sweep:
        result = run()
        print(result.arguments, result.result)
    # Or dispatch one job: sweep[0]()
    ```

    Branching over algorithms, each with its own hyperparameters:
    ```python
    sweep = Sweep(
        train,
        GridSearch(
            {
                "algorithm": {
                    "PPO": [
                        {"algorithm": PPO()},
                        SobolSearch({"clip_coef": (0.1, 0.3)}, 32, seed=key),
                    ],
                    "SAC": [
                        {"algorithm": SAC()},
                        SobolSearch({"tau": (1e-3, 5e-2)}, 32, seed=key),
                    ],
                }
            }
        ),
    )
    # Results record algorithm="PPO"; the run is called with algorithm=PPO().
    ```

    Or in a slurm job array:
    ```bash
    #SBATCH --array=0-5
    python create_jobs_and_run_id.py --batch_idx $SLURM_ARRAY_TASK_ID
    ```
    """

    fn: Callable[[SweepJob], SweepResult]
    jobs: list[SweepJob]

    def __init__(
        self,
        fn: Callable,
        *stages: ParameterSpaceSearch,
        print_cost_estimate: bool = False,
    ):
        if not stages:
            raise ValueError("At least one sweep stage is required")

        # _expand_space will build all stage combinations and built all the jobs
        self.jobs = _expand_space(list(stages), [SweepJob({}, {})])
        logger.info(f"Created {len(self.jobs)} sweep jobs")

        def run_job(job: SweepJob) -> SweepResult:
            start_time = time.time()
            trial = fn
            # if the trail implements a with_labels, this can be used to pass some extras
            # that will not be passed as call arguments.
            with_labels = getattr(fn, "with_labels", None)
            if job.labels and callable(with_labels):
                trial = with_labels(job.labels)
            result = jax.block_until_ready(trial(**job.args))  # type: ignore
            return SweepResult(
                start_time=start_time,
                end_time=time.time(),
                arguments=job.arguments,
                result=result,
            )

        self.fn = run_job

        if print_cost_estimate:
            _log(
                "Estimating the cost of the first job.",
                " Note that this is only a proxy. Jobs in this sweep with different input arguments may have different costs,"
                " especially when sweeping over different environments, num_envs etc.",
                " Compiling...",
            )
            _log(f"{log_cost_estimate(fn, **self.jobs[0].args)}")
            _log("Done.")

    def __len__(self):
        return len(self.jobs)

    def __iter__(self):
        return (partial(self.fn, job) for job in self.jobs)

    def __repr__(self):
        return f"Sweep({len(self.jobs)} jobs)"

    @overload
    def __getitem__(self, index: int) -> partial[SweepResult]: ...
    @overload
    def __getitem__(self, index: slice) -> Sweep: ...

    def __getitem__(self, index: int | slice) -> partial[SweepResult] | Sweep:
        if isinstance(index, slice):
            # copy of the fn over a subset of the jobs.
            sliced = copy.copy(self)
            sliced.jobs = self.jobs[index]
            return sliced
        return partial(self.fn, self.jobs[index])
