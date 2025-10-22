try:
    from ._tree import (
        tree_batch_sum as tree_batch_sum,
        tree_concatenate as tree_concatenate,
        tree_gather_actions as tree_gather_actions,
        tree_get_first as tree_get_first,
        tree_map_distribution as tree_map_distribution,
        tree_map_one_level as tree_map_one_level,
        tree_mean as tree_mean,
        tree_stack as tree_stack,
        tree_unstack as tree_unstack,
    )

    batch_sum = tree_batch_sum
    get_first = tree_get_first
    gather_actions = tree_gather_actions
    map_one_level = tree_map_one_level
    mean = tree_mean
    stack = tree_stack
    unstack = tree_unstack
    concatenate = tree_concatenate
    map_distribution = tree_map_distribution

    __all__ = [
        "get_first",
        "map_one_level",
        "mean",
        "stack",
        "unstack",
        "concatenate",
        "map_distribution",
    ]
except ImportError:
    print(
        "Jaxnasium.tree module requires `optax` to be installed. Please install via `pip install optax`."
    )
