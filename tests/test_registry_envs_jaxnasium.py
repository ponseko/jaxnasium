import pytest
from _test_utils import check_env, registry_envs_for_package

import jaxnasium as jym
from jaxnasium.wrappers import FlattenObservationWrapper, LogWrapper, Wrapper

JAXNASIUM_ENVS = registry_envs_for_package("jaxnasium")


@pytest.mark.parametrize("env_id", JAXNASIUM_ENVS)
def test_jaxnasium_env_smoke(env_id: str) -> None:
    check_env(env_id, flatten_obs=True)


@pytest.mark.parametrize(
    "wrappers, expected",
    [
        (None, [LogWrapper]),
        ([..., FlattenObservationWrapper], [FlattenObservationWrapper, LogWrapper]),
        ([FlattenObservationWrapper, ...], [LogWrapper, FlattenObservationWrapper]),
        ([FlattenObservationWrapper], [FlattenObservationWrapper]),
    ],
)
def test_make_wrappers(wrappers, expected) -> None:
    """`...` stands for the default wrappers, in its place; outermost wrapper first."""
    env = jym.make("CartPole-v1", wrappers=wrappers)
    applied = []
    while isinstance(env, Wrapper):
        applied.append(type(env))
        env = env._env
    assert applied == expected
