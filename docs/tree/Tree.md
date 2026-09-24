# PyTree utilities

The `jaxnasium.tree` package provides convenience pytree functions used in the various RL
algorithms which aren't found in the used higher-level libraries (equinox / jax / optax).

```python
import jaxnasium as jym

jym.tree.mean(pytree)
```

## Reductions

::: jaxnasium.tree.mean

::: jaxnasium.tree.sum

::: jaxnasium.tree.batch_mean

::: jaxnasium.tree.batch_sum

## Arithmetic

::: jaxnasium.tree.add

::: jaxnasium.tree.mul

::: jaxnasium.tree.clip

## Structure

::: jaxnasium.tree.get_first

::: jaxnasium.tree.map_one_level

::: jaxnasium.tree.map_distribution

::: jaxnasium.tree.stack

::: jaxnasium.tree.unstack

::: jaxnasium.tree.concatenate

::: jaxnasium.tree.ravel

::: jaxnasium.tree.gather_actions

## Creation

::: jaxnasium.tree.zeros_like

::: jaxnasium.tree.ones_like

::: jaxnasium.tree.split_key_like

::: jaxnasium.tree.split_key_like_structure
