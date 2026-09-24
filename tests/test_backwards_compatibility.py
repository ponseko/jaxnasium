"""Smoke tests for jaxnasium.algorithms.utils backwards compatibility."""

import warnings


def test_utils_init_imports():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from jaxnasium.algorithms.utils import (
            Schedule,
            Transition,
            TransitionBuffer,
        )

    assert TransitionBuffer is not None
    assert Transition is not None
    assert Schedule is not None


def test_utils_experimental_imports():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from jaxnasium.algorithms.utils.experimental import (
            create_batched_grid_search,
            create_batched_random_search,
        )

    assert create_batched_grid_search is not None
    assert create_batched_random_search is not None
