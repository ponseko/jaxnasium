"""Shared pytest configuration for the test suite."""

import gc

import equinox as eqx
import jax
import pytest

import jaxnasium

jaxnasium.enable_compilation_cache()

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 2)


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    """Release compiled executables between tests."""
    yield
    eqx.clear_caches()
    jax.clear_caches()
    gc.collect()
