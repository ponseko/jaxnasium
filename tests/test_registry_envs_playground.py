import pytest
from _test_utils import check_env, registry_envs_for_package

# Only run these in "external" mode
pytestmark = pytest.mark.external

pytest.importorskip("mujoco_playground")


PLAYGROUND_ENVS = registry_envs_for_package("playground")


@pytest.mark.parametrize("env_id", PLAYGROUND_ENVS)
def test_playground_env_smoke(env_id: str) -> None:
    check_env(env_id, flatten_obs=True, run_env=True)
