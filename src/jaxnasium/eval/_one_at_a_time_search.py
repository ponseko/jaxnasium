from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._sweep import Sweep


@dataclass(frozen=True)
class OneAtATimeSearch:
    """One-factor-at-a-time (OFAT) search over discrete parameter values.

    Sets the first value of each parameter list as the baseline, then varies
    one parameter at a time to each of its other values while holding the rest at
    baseline. Linear in the number of values.

    Pass as a stage to [`Sweep`][jaxnasium.eval.Sweep], alone or chained with
    other searches. For only performing this search,
    `OneAtATimeSearch(...).sweep(fn, ...)` is a shorthand for
    `Sweep(fn, self, ...)`.

    **Arguments**:
        `params`: A dict of param names mapping to a non-empty list of values,
            or to a `{label: sub-space}` dict of branches (see  [`Sweep`][jaxnasium.eval.Sweep]).
            The first entry of each is the baseline value.

    **Examples**:

    ```python
    sweep = OneAtATimeSearch(
        {
            "gamma": [0.99, 0.95, 0.999],
            "lr": [0.1, 0.01],
            # One tag in the results, two keyword arguments in the call.
            "network": {
                "mlp": {"actor_kwargs": MLP_KWARGS, "critic_kwargs": MLP_KWARGS},
                "simba": {
                    "actor_kwargs": SIMBA_KWARGS,
                    "critic_kwargs": SIMBA_KWARGS,
                },
            },
        }
    ).sweep(train)
    for run in sweep:
        result = run()
        print(result.arguments, result.result)
    ```
    """

    params: dict[str, list | dict]

    def __post_init__(self):
        assert self.params, "OneAtATimeSearch params cannot be empty"
        assert all(
            isinstance(values, (list, dict)) and values
            for values in self.params.values()
        ), "OneAtATimeSearch params must be non-empty lists, or dicts of branches"

    def configs(self) -> list[dict[str, Any]]:
        """Returns the baseline config, then one config per non-baseline value of each parameter."""
        # Iterating a branch dict yields its labels, so both specs work the same.
        options = {name: list(values) for name, values in self.params.items()}
        baseline = {name: values[0] for name, values in options.items()}
        configs = [dict(baseline)]
        for name, values in options.items():
            for value in values[1:]:
                configs.append({**baseline, name: value})
        return configs

    def sweep(self, fn: Callable, *, print_cost_estimate: bool = False) -> Sweep:
        """Create a [`Sweep`][jaxnasium.eval.Sweep] with this search alone."""
        return Sweep(fn, self, print_cost_estimate=print_cost_estimate)
