import pytest
from _test_utils import check_env, registry_envs_for_package

# Only run these in "external" mode
pytestmark = pytest.mark.external

pytest.importorskip("factoriax")


FACTORIAX_ENVS = registry_envs_for_package("factoriax")


@pytest.mark.parametrize("env_id", FACTORIAX_ENVS)
def test_factoriax_env_smoke(env_id: str) -> None:
    check_env(env_id, flatten_obs=True, run_env=True)
