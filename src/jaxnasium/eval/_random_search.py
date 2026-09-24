from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jaxtyping import PRNGKeyArray
from scipy.stats import qmc

from ._sweep import Sweep


def _as_key(seed: PRNGKeyArray | int) -> PRNGKeyArray:
    """A plain int becomes a fresh key; an existing key passes through."""
    if isinstance(seed, int):
        return jax.random.PRNGKey(seed)
    return seed


def _scale_to_spec(spec: list | tuple | dict, u: float) -> Any:
    """Maps a ranomly drawn `u` in [0, 1) onto the specified spec.
    A spec is a list of choices, a dict of branches (drawing a label), or a
    tuple of (low, high, ["linear", "log"])

    Examples:
    >>> _scale_to_spec(["a", "b", "c"], 0.5)
    "b"

    >>> _scale_to_spec((1, 5, "linear"), 0.5)
    3 # int input maps to int output

    >>> _scale_to_spec((1.0, 5.0, "linear"), 0.5)
    3.0 # float input maps to float output

    >>> _scale_to_spec((1, 5, "log"), 0.5)
    2.2360680103302

    """
    is_range = isinstance(spec, tuple) and len(spec) in (2, 3)
    is_choice = isinstance(spec, (list, dict)) and len(spec) > 0
    assert is_range or is_choice, (
        f"Search params must be a non-empty list, a dict of branches, or a "
        f"(low, high) / (low, high, scale) tuple, got {spec!r}"
    )

    if isinstance(spec, (list, dict)):
        # Iterating a branch dict yields its labels, so both draw the same way.
        return list(spec)[int(u * len(spec))]

    if len(spec) == 2:
        low, high = spec
        scale = "linear"
    elif len(spec) == 3:
        low, high, scale = spec
        if scale not in ("linear", "log"):
            raise ValueError(f"Range scale must be 'linear' or 'log', got {scale!r}")
    else:
        raise ValueError(
            f"Range must be (low, high) or (low, high, scale), got {spec!r}"
        )

    if scale == "log":
        if low <= 0 or high <= 0:
            raise ValueError(f"log scale requires positive bounds, got ({low}, {high})")
        log_low, log_high = np.log(low), np.log(high)
        return float(np.exp(log_low + u * (log_high - log_low)))

    if isinstance(low, int) and isinstance(high, int):
        return low + int(u * (high - low))
    return low + u * (high - low)


@dataclass(frozen=True)
class RandomSearch:
    """`num_samples` independent draws from the given parameters.

    Pass as a stage to [`Sweep`][jaxnasium.eval.Sweep], alone or
    chained with other searches. For only performing this random search,
    `RandomSearch(...).sweep(fn, ...)` is a shorthand for `Sweep(fn, self, ...)`.
    See [`Sweep`][jaxnasium.eval.Sweep] for more details.

    **Examples**:

    ```python
    sweep = RandomSearch(
        {"learning_rate": (1e-4, 1e-2, "log"), "gamma": [0.98, 0.99]},
        num_samples=64,
        seed=jax.random.PRNGKey(1),
    ).sweep(train)
    for run in sweep:
        result = run()
        print(result.arguments, result.result)
    ```

    Combined with a grid search over environments (same learning rates in each env):

    ```python
    sweep = Sweep(
        train,
        GridSearch({"env_name": ["CartPole-v1", "Acrobot-v1"]}),
        RandomSearch(
            {"learning_rate": (1e-4, 1e-2, "log")},
            num_samples=64,
            seed=jax.random.PRNGKey(1),
        ),
    )
    for run in sweep:
        result = run()
        print(result.arguments, result.result)
    ```
    """

    params: dict[str, list | tuple | dict]
    num_samples: int
    seed: PRNGKeyArray | int

    def __post_init__(self):
        assert self.params, "RandomSearch params cannot be empty"
        assert self.num_samples > 0, "num_samples must be positive"
        # Eager validation check of each param:
        [_scale_to_spec(value, 0.0) for value in self.params.values()]

    def configs(self) -> list[dict[str, Any]]:
        """Draw `num_samples` configurations"""
        random_u_samples = jax.random.uniform(
            _as_key(self.seed), (self.num_samples, len(self.params))
        )
        random_u_samples = np.asarray(random_u_samples)
        names = list(self.params)
        return [
            {
                name: _scale_to_spec(self.params[name], float(u))
                for name, u in zip(names, row)
            }
            for row in random_u_samples
        ]

    def sweep(
        self,
        fn: Callable,
        *,
        print_cost_estimate: bool = False,
    ) -> Sweep:
        """Create a [`Sweep`][jaxnasium.eval.Sweep] with this search alone."""
        return Sweep(
            fn,
            self,
            print_cost_estimate=print_cost_estimate,
        )


@dataclass(frozen=True)
class SobolSearch:
    """Quasi-random draws that cover the parameter space more evenly than RandomSearch.

    Pass as a stage to [`Sweep`][jaxnasium.eval.Sweep], alone or
    chained with other searches. For only performing this Sobol search,
    `SobolSearch(...).sweep(fn, ...)` is a shorthand for `Sweep(fn, self, ...)`.
    See [`Sweep`][jaxnasium.eval.Sweep] for more details.

    **Examples**:

    ```python
    sweep = SobolSearch(
        {"learning_rate": (1e-4, 1e-2, "log"), "gamma": (0.9, 1.0)},
        num_samples=64,
        seed=jax.random.PRNGKey(1),
    ).sweep(train)
    for run in sweep:
        result = run()
        print(result.arguments, result.result)
    ```

    Combined with a grid search over environments:

    ```python
    sweep = Sweep(
        train,
        GridSearch({"env_name": ["CartPole-v1", "Acrobot-v1"]}),
        SobolSearch(
            {"learning_rate": (1e-4, 1e-2, "log"), "gamma": (0.9, 1.0)},
            num_samples=64,
            seed=jax.random.PRNGKey(1),
        ),
        seed=jax.random.PRNGKey(0),
    )
    for run in sweep:
        result = run()
        print(result.arguments, result.result)
    ```
    """

    params: dict[str, list | tuple | dict]
    num_samples: int
    seed: PRNGKeyArray | int

    def __post_init__(self):
        assert self.params, "SobolSearch params cannot be empty"
        assert self.num_samples > 0, "num_samples must be positive"
        # Eager validation check of each param:
        [_scale_to_spec(value, 0.0) for value in self.params.values()]

    def configs(self) -> list[dict[str, Any]]:
        """`num_samples` configurations spread evenly over the parameter space."""
        # We accept a JAX key, but need to convert it to a regular key
        seed = jax.random.key_data(_as_key(self.seed))  # new random.key() api
        sampler = qmc.Sobol(
            len(self.params), rng=np.random.default_rng(np.asarray(seed))
        )
        random_u_samples = sampler.random(self.num_samples)
        names = list(self.params)
        return [
            {
                name: _scale_to_spec(self.params[name], float(u))
                for name, u in zip(names, row)
            }
            for row in random_u_samples
        ]

    def sweep(self, fn: Callable, *, print_cost_estimate: bool = False) -> Sweep:
        """Create a [`Sweep`][jaxnasium.eval.Sweep] with this search alone."""
        return Sweep(fn, self, print_cost_estimate=print_cost_estimate)
