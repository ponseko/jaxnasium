"""Temporary stub for backwards compatibility."""

import warnings

from jaxnasium.algorithms.experimental._batching import (
    create_batched_grid_search as create_batched_grid_search,
    create_batched_random_search as create_batched_random_search,
)

warnings.warn(
    "importing from jaxnasium.algorithms.utils.experimental is deprecated."
    "You can now import directly from jaxnasium.algorithms.experimental",
    DeprecationWarning,
    stacklevel=2,
)
