from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree

DEFAULT_WEIGHT_INIT = partial(jax.nn.initializers.orthogonal, np.sqrt(2))
POLICY_HEAD_WEIGHT_INIT = partial(jax.nn.initializers.orthogonal, 0.01)
VALUE_HEAD_WEIGHT_INIT = partial(jax.nn.initializers.orthogonal, 1.0)
ZERO_INIT = partial(jax.nn.initializers.constant, 0.0)
DEFAULT_BIAS_INIT = 0.0


@eqx.filter_jit
def set_weight_bias(
    key: PRNGKeyArray,
    network: PyTree[eqx.Module],
    weight_init: Callable[..., jax.nn.initializers.Initializer]
    | None = DEFAULT_WEIGHT_INIT,
    bias_init: float | None = DEFAULT_BIAS_INIT,
):
    """Sets all `eqx.nn.Linear` and `eqx.nn.Conv` layers in
    a network to a given weight and bias initialization.
    Defaults to orthogonal weight initialization and zero bias initialization.
    Setting weight_init or bias_init to None will leave the weights or biases unchanged.
    """
    is_layer = lambda x: isinstance(x, (eqx.nn.Linear, eqx.nn.Conv))
    layers, network_structure = jax.tree.flatten(network, is_leaf=is_layer)

    new_layers = layers

    # Update bias
    if bias_init is not None:
        new_layers = [
            eqx.tree_at(
                lambda x: x.bias,
                layer,
                replace_fn=lambda x: jnp.ones_like(x) * bias_init,  # type: ignore
            )
            if is_layer(layer) and layer.bias is not None
            else layer
            for layer in layers
        ]

    keys = jax.random.split(key, len(new_layers))
    if weight_init is not None:
        # Update weight
        new_layers = [
            eqx.tree_at(
                lambda x: x.weight,
                layer,
                replace_fn=lambda x, k=k: weight_init()(k, x.shape, x.dtype),  # type: ignore
            )
            if is_layer(layer)
            else layer
            for layer, k in zip(new_layers, keys)
        ]

    return jax.tree.unflatten(network_structure, new_layers)


rl_initialization = partial(
    set_weight_bias, weight_init=DEFAULT_WEIGHT_INIT, bias_init=DEFAULT_BIAS_INIT
)
