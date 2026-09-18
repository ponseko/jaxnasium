import pytest
from _test_utils import check_env, registry_envs_for_package

# Only run these in "external" mode
pytestmark = pytest.mark.external

pytest.importorskip("pgx")


PGX_ENVS = registry_envs_for_package("pgx")


@pytest.mark.parametrize("env_id", PGX_ENVS)
def test_pgx_env_smoke(env_id: str) -> None:
    check_env(env_id, flatten_obs=True, run_env=True)
