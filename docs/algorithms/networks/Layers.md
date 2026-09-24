# PyTree Layers

The pieces that make [agent networks](Networks.md) work on composite (PyTree) observation
and action spaces. The input/output modules build one sub-network per space in the tree; the
output layers turn a body's features into a distribution or Q-values for a single space.

## Input

::: jaxnasium.algorithms.core.PyTreeObsSpaceNetwork
    options:
        members:
            - __init__
            - __call__
            - with_params

## Output

::: jaxnasium.algorithms.core.PyTreeActionNetwork
    options:
        members:
            - __init__
            - __call__
            - with_params

::: jaxnasium.algorithms.core.PyTreeQValueNetwork
    options:
        members:
            - __init__
            - __call__
            - with_params

## Output layers

These are the per-space heads selected by the modules above, and can be swapped through the
`discrete_output_layer` / `continuous_output_layer` / `output_layer_type` arguments of the
[agent networks](Networks.md).

::: jaxnasium.algorithms.core.CategoricalLayer
    options:
        members:
            - __call__
            - with_params

::: jaxnasium.algorithms.core.NormalLayer
    options:
        members:
            - __call__
            - with_params

::: jaxnasium.algorithms.core.TanhNormalLayer
    options:
        members:
            - __call__
            - with_params

::: jaxnasium.algorithms.core.QLayer
    options:
        members:
            - __call__
            - with_params
