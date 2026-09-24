import pytest
from _test_utils import check_env, registry_envs_for_package

import jaxnasium  # noqa: F401

JAXNASIUM_ENVS = registry_envs_for_package("jaxnasium")


@pytest.mark.parametrize("env_id", JAXNASIUM_ENVS)
def test_jaxnasium_env_smoke(env_id: str) -> None:
    check_env(env_id, flatten_obs=True)
