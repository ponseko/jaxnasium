import pytest
from _test_utils import check_env

# Only run these in "external" mode
pytestmark = pytest.mark.external

pytest.importorskip("econojax")
pytest.importorskip("rice_jax")
pytest.importorskip("chargax")


@pytest.mark.skip(reason="EconoJax is pending upgrade")
def test_econojax_env():
    from econojax import EconoJax  # type: ignore

    env = EconoJax(num_population=4)
    check_env(env, flatten_obs=True)


@pytest.mark.skip(reason="Rice JAX is undergoing an update")
def test_rice_jax_env():
    from rice_jax import Rice  # type: ignore
    from rice_jax.util import load_region_yamls  # type: ignore

    region_yamls = load_region_yamls(3)
    env = Rice(region_yamls)

    check_env(env, flatten_obs=True)


def test_chargax_env():
    from chargax import Chargax, ChargingStation  # type: ignore

    station = ChargingStation.init_default_station()
    env = Chargax(station=station)

    check_env(env, flatten_obs=True)
