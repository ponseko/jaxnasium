# Parameter Space Searches

Each search produces a list of configurations. They are the stages of a
[`Sweep`][jaxnasium.eval.Sweep]: pass several to nest them, or call `.sweep(fn)` on one to
sweep it alone.

A parameter may also map to a `{label: sub-space}` dict rather than to plain values. The
label is what gets recorded in the results, while the sub-space it expands into supplies the
arguments of the actual call. This is how you sweep, say, algorithm-specific hyperparameters
under a single `algorithm` parameter.

::: jaxnasium.eval.GridSearch
    options:
        members:
            - params
            - configs
            - sweep

::: jaxnasium.eval.RandomSearch
    options:
        members:
            - configs
            - sweep

::: jaxnasium.eval.SobolSearch
    options:
        members:
            - configs
            - sweep

::: jaxnasium.eval.OneAtATimeSearch
    options:
        members:
            - configs
            - sweep
