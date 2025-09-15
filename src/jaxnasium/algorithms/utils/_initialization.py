import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray


def rl_initialization(
    key: PRNGKeyArray,
    network: eqx.Module,
    weight_init: jax.nn.initializers.Initializer = jax.nn.initializers.orthogonal(),
    bias_init=0.0,
):
    """Sets all layers in a network to a given weight and bias initialization.
    Defaults to orthogonal weight initialization and zero bias initialization.
    """
    is_layer = lambda x: isinstance(x, eqx.nn.Linear) or isinstance(x, eqx.nn.Conv)
    layers, network_structure = jax.tree.flatten(network, is_leaf=is_layer)

    # Update bias
    new_layers = [
        eqx.tree_at(
            lambda x: x.bias, layer, replace_fn=lambda x: jnp.ones_like(x) * bias_init
        )
        if is_layer(layer) and layer.bias is not None
        else layer
        for layer in layers
    ]

    # Update weight
    new_layers = [
        eqx.tree_at(
            lambda x: x.weight,
            layer,
            replace_fn=lambda x: weight_init(key, x.shape, x.dtype),
        )
        if is_layer(layer)
        else layer
        for layer in layers
    ]

    return jax.tree.unflatten(network_structure, new_layers)
