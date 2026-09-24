import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._sweep import Sweep


@dataclass(frozen=True)
class GridSearch:
    """Sets up a grid search over the given parameter values. All
    combinations of the parameter values are created in its configs.

    Pass as a stage to [`Sweep`][jaxnasium.eval.Sweep], alone or
    chained with other searches. For only performing this grid search,
    `GridSearch(...).sweep(fn, ...)` is a shorthand for `Sweep(fn, self, ...)`.
    See [`Sweep`][jaxnasium.eval.Sweep] for more details.

    **Arguments**:
        `params`: A dict of param names mapping to a non-empty list of values,
            or to a `{label: sub-space}` dict of branches. See
            [`Sweep`][jaxnasium.eval.Sweep] for what a branch expands into.

    **Examples**:

    ```python
    sweep = GridSearch(
        {
            "env_name": ["CartPole-v1", "Acrobot-v1"],
            "gamma": [0.95, 0.99, 0.999],
        }
    ).sweep(train)
    for run in sweep:
        result = run()
        print(result.arguments, result.result)
    ```

    Combined with a random search over learning rates:

    ```python
    sweep = Sweep(
        train,
        GridSearch({"env_name": ["CartPole-v1", "Acrobot-v1"]}),
        RandomSearch(
            {"learning_rate": (1e-4, 1e-2, "log")},
            num_samples=32,
            seed=jax.random.PRNGKey(0),
        ),
    )
    for run in sweep:
        result = run()
        print(result.arguments, result.result)
    ```

    Branching, so each environment brings its own arguments along:

    ```python
    GridSearch(
        {
            "env_name": {
                "CartPole-v1": {
                    "env": jym.make("CartPole-v1"),
                    "total_timesteps": 100_000,
                },
                "Acrobot-v1": {
                    "env": jym.make("Acrobot-v1"),
                    "total_timesteps": 500_000,
                },
            }
        }
    )
    ```
    """

    params: dict[str, list | dict]

    def __post_init__(self):
        assert self.params, "GridSearch params cannot be empty"
        assert all(
            isinstance(values, (list, dict)) and values
            for values in self.params.values()
        ), "GridSearch params must be non-empty lists, or dicts of branches"

    def configs(self) -> list[dict[str, Any]]:
        """
        Returns every combination of the parameter values. For a branch,
        the combination holds its label.
        """
        names = list(self.params)
        return [
            dict(zip(names, combination))
            # Iterating a branch dict yields its labels.
            for combination in itertools.product(*self.params.values())
        ]

    def sweep(self, fn: Callable, *, print_cost_estimate: bool = False) -> Sweep:
        """Create a [`Sweep`][jaxnasium.eval.Sweep] object with this grid search"""
        return Sweep(fn, self, print_cost_estimate=print_cost_estimate)
