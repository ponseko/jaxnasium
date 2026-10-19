import importlib
import os
from importlib.util import find_spec

import pytest
from _test_utils import check_env, registry_envs_for_package

import jaxnasium as jym

# Only run these in "external" mode
pytestmark = pytest.mark.external

# Every package the registry points to, except those not installed in CI
UNTESTED = {"craftax"}
PACKAGES = sorted({t.split(":")[0] for t in jym.registry._aliases.values()} - UNTESTED)
IMPORT_NAMES = {"playground": "mujoco_playground"}  # Where they differ


# Set in CI to the suite(s) under test: those must fail when missing or broken, not skip
REQUIRED = os.environ.get("JAXNASIUM_REQUIRE_IMPORTS", "").split(",")


def _installed(package: str) -> bool:
    import_name = IMPORT_NAMES.get(package, package)
    if package in REQUIRED:
        importlib.import_module(import_name)
        return True
    return find_spec(import_name) is not None


# One skipped placeholder per missing package, instead of one skip per env
ENV_PARAMS = []
for package in PACKAGES:
    if _installed(package):
        ENV_PARAMS += registry_envs_for_package(package)
    else:
        skip = pytest.mark.skip(reason=f"{package} not installed")
        ENV_PARAMS.append(
            pytest.param(package, id=f"{package}/not-installed", marks=skip)
        )


@pytest.mark.parametrize("env_id", ENV_PARAMS)
def test_registry_env_smoke(env_id: str) -> None:
    check_env(env_id, flatten_obs=True)
