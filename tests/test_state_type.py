import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from typing_extensions import TypeVar

import jaxnasium as jym
from jaxnasium.envs.cartpole import CartPole, EnvState as CartPoleState


class BaseState(jym.EnvState):
    profit: float = 0.0


TState = TypeVar("TState", bound=BaseState, default=BaseState)


class Base(jym.Environment[TState]):
    def reset_env(self, key):
        return jnp.zeros(1), self.state_type(profit=jnp.float32(0.0))

    def step_env(self, key, state, action):
        state = state._replace(profit=state.profit + 1.0)
        return jym.TimeStep(jnp.zeros(1), 1.0, False, False, {}), state

    @property
    def action_space(self):
        return jym.Discrete(2)

    @property
    def observation_space(self):
        return jym.Box(-1, 1, (1,))


class ExtState(BaseState):
    counter: int = 0


TExt = TypeVar("TExt", bound=ExtState, default=ExtState)


class Ext(Base[TExt]):
    def step_env(self, key, state, action):
        timestep, state = super().step_env(key, state, action)
        return timestep, state._replace(counter=state.counter + 1)


class Ext2State(ExtState):
    extra: float = 3.0


class Ext2(Ext[Ext2State]):
    pass


class PlainSubclass(Ext):
    pass


TBound = TypeVar("TBound", bound=ExtState)  # no default


class BoundOnly(Base[TBound]):
    pass


class Unbound(jym.Environment):
    reset_env = Base.reset_env
    step_env = Base.step_env

    @property
    def action_space(self):
        return jym.Discrete(2)

    @property
    def observation_space(self):
        return jym.Box(-1, 1, (1,))


@pytest.mark.parametrize(
    "env_cls, expected",
    [
        (CartPole, CartPoleState),
        (Base, BaseState),  # TypeVar default
        (Ext2, Ext2State),  # rebound through a subclass chain
        (PlainSubclass, ExtState),  # inherited without subscripting
        (BoundOnly, ExtState),  # TypeVar bound
    ],
)
def test_state_type_resolves(env_cls, expected):
    assert env_cls().state_type is expected


def test_state_type_is_kept_through_jitted_steps():
    @eqx.filter_jit
    def rollout(env, key):
        _, state = env.reset(key)
        _, state = env.step(key, state, 0)
        _, state = env.step(key, state, 0)
        return state

    state = rollout(Ext2(), jax.random.PRNGKey(0))
    assert type(state) is Ext2State
    assert int(state.counter) == 2
    assert float(state.profit) == 2.0
    assert float(state.extra) == 3.0


def test_unbound_env_raises():
    with pytest.raises(TypeError, match="does not bind a concrete state type"):
        _ = Unbound().state_type


def test_wrapper_raises():
    with pytest.raises(TypeError, match="of a wrapper"):
        _ = jym.LogWrapper(Ext2()).state_type
