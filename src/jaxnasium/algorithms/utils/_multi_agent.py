import functools
import inspect
from typing import Callable, List, Optional

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTreeDef

from jaxnasium.tree import map_one_level, stack, unstack

from ._transition import Transition


def _result_tuple_to_tuple_result(r):
    """
    Some functions may return tuples. Rather than returning
    a pytree of tuples, we convert it to a tuple of pytrees.
    """
    one_level_leaves, structure = eqx.tree_flatten_one_level(r)
    if isinstance(one_level_leaves[0], tuple):
        tupled = tuple([list(x) for x in zip(*one_level_leaves)])
        r = tuple(jax.tree.unflatten(structure, leaves) for leaves in tupled)
    return r


def _is_prng_key(arg):
    """Check if the argument is a JAX PRNGKeyArray."""
    try:
        jax.random.split(arg)
        return True
    except Exception:
        return False


def split_key_over_agents(key: PRNGKeyArray, agent_structure: PyTreeDef):
    """
    Given a key and a pytree structure, split the key into
    as many keys as there are leaves in the pytree.
    Useful when provided with a flat pytree of agents.

    Similar to `optax.tree_utils.tree_split_key_like`, but operates on PyTreeDefs.

    *Arguments*:
        `key`: A PRNGKeyArray to be split.
        `agent_structure`: A pytree structure of agents.
    """
    num_agents = agent_structure.num_leaves
    keys = list(jax.random.split(key, num_agents))
    return jax.tree.unflatten(agent_structure, keys)


def transform_multi_agent(
    func: Optional[Callable] = None,
    *,
    shared_argnames: Optional[List[str]] = None,
    auto_split_keys: bool = True,
    auto_transpose_transitions: bool = True,
) -> Callable:
    """
    Transformation docorator to handle multi-agent settings.
    Essentially, this function either applies a vmap over the agents (if the agent are homogeneous)
    or applies a `jax.tree.map` over the first level of the PyTree structure of the arguments.

    Essentially, this transformation allows for writing single-agent functions that can
    automatically be upgraded to multi-agent settings.

    **Arguments**:

    - `func`: The function to be transformed. If None, returns a decorator.
    - `shared_argnames`: An optional list of argument names that are shared across agents. If None, the first (non-PRNGKey) argument is
        assumed to be a per-agent argument. All arguments with the same first-level PyTree structure are also considered per-agent arguments.
        The rest are considered shared arguments. PRNG keys that are provided as arguments are automatically split over agents.
    - `auto_split_keys`: When enabled, provided single PRNGKeys are automatically split over the same structure as the first argument.
    - `auto_split_transitions`: When enabled, `Transition` objects are automatically transposed before being passed to the function.
            After the function call, the `Transition` object is reconstructed into the original structure.

    **Example**:
    ```python
    >>> @transform_multi_agent
    ... def get_action(key, agent, observation):
    ...     action_dist = agent.actor(observation)
    ...     return action_dist.sample(seed=key)

    # Usage:
    >>> models = {"agent0": model1, "agent1": model2}
    >>> agent_observations = {"agent0": obs0, "agent1": obs1} # <-- Same first-level structure as `models`
    >>> key = jax.random.PRNGKey(42) # <-- `key` inputs are automatically split over agents
    >>> actions = get_action(agent_states, agent_observations, key)
    >>> # Result: {"agent0": action0, "agent1": action1}
    ```
    """
    assert callable(func) or func is None

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments

            shared: set[str] = set(shared_argnames or ())
            shared.update(k for k in ("self", "cls") if k in params)

            _has_transposed_transition = False

            # Grab the first param key that is not in shared and is not a PRNGKey
            # We use the "first-level" PyTree structure as the "agent structure" and use it as reference
            ref_key = next(
                (k for k in params if k not in shared and not _is_prng_key(params[k])),
                None,
            )
            if ref_key is None:
                raise ValueError("No per-agent args found in the function signature.")
            ref_arg = params[ref_key]
            if isinstance(ref_arg, Transition):
                ref_structure = ref_arg.structure
            else:
                _, ref_structure = eqx.tree_flatten_one_level(ref_arg)

            # Infer the shared arguments from the function signature if not provided
            if shared_argnames is None:
                for k, v in params.items():
                    _, param_structure = eqx.tree_flatten_one_level(v)
                    if ref_structure != param_structure:
                        shared.add(k)

            # Auto splitting:
            for shared_param in shared.copy():
                arg = params[shared_param]

                if auto_split_keys and _is_prng_key(arg):
                    params[shared_param] = split_key_over_agents(arg, ref_structure)
                    shared.remove(shared_param)

                if auto_transpose_transitions and isinstance(arg, Transition):
                    params[shared_param] = arg.view_transposed
                    shared.remove(shared_param)
                    _has_transposed_transition = True

            shared_params = {k: params[k] for k in shared}
            per_agent_params = {k: params[k] for k in params if k not in shared}

            # Prepare a function that takes only per-agent args
            def agent_func(*agent_args):
                per_agent_kwargs = dict(zip(per_agent_params.keys(), agent_args))
                return func(**per_agent_kwargs, **shared_params)

            try:  # Try stack + vmap
                first_arg = next(iter(per_agent_params.values()))
                _, agent_structure = eqx.tree_flatten_one_level(first_arg)
                stacked_args = {k: stack(v) for k, v in per_agent_params.items()}
                result = jax.vmap(agent_func)(*stacked_args.values())
                result = unstack(result, structure=agent_structure)

            except Exception:  # Homogeneous inputs, cannot vmap. Apply map instead.
                result = map_one_level(agent_func, *per_agent_params.values())

            result = _result_tuple_to_tuple_result(result)

            if _has_transposed_transition:
                # If a Transition was transposed, and the function
                # returns an (updated) transpsed Transition,
                # we need to rebuild the original structure
                is_pytree_of_transitions = lambda x: all(
                    isinstance(y, Transition) for y in eqx.tree_flatten_one_level(x)[0]
                )
                if is_pytree_of_transitions(result):
                    result = Transition.from_transposed(result)
                elif isinstance(result, tuple):
                    result_list = []
                    for res in result:
                        if is_pytree_of_transitions(res):
                            result_list.append(Transition.from_transposed(res))
                        else:
                            result_list.append(res)
                    result = tuple(result_list)

            return result

        return wrapper

    return decorator(func) if callable(func) else decorator
