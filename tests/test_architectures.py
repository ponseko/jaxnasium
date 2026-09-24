import jax
import jax.numpy as jnp
import pytest

from jaxnasium.algorithms.architectures import CNN, MLP, BroNet, SimBa

SEED = jax.random.PRNGKey(0)


def test_mlp_forward_and_out_features():
    net = MLP(in_features=6, key=SEED, hidden_sizes=(32, 16))
    assert net.out_features == 16

    x = jnp.ones((6,))
    y = net(x)
    assert y.shape == (16,)
    assert jnp.all(jnp.isfinite(y))


def test_mlp_single_layer():
    net = MLP(in_features=4, key=SEED, hidden_sizes=(8,))
    assert net.out_features == 8
    assert net(jnp.ones((4,))).shape == (8,)


def test_mlp_with_params_factory():
    factory = MLP.with_params(hidden_sizes=(10, 5))
    net = factory(in_features=3, key=SEED)
    assert isinstance(net, MLP)
    assert net.out_features == 5
    assert net(jnp.ones((3,))).shape == (5,)


@pytest.mark.parametrize("channels_axis", ["first", "last"])
def test_cnn_forward_flattens_to_vector(channels_axis):
    input_shape = (3, 8, 8) if channels_axis == "first" else (8, 8, 3)
    net = CNN(input_shape, key=SEED, channels_axis=channels_axis)

    x = jnp.ones(input_shape)
    y = net(x)
    assert y.ndim == 1
    assert y.shape[0] == net.out_features
    assert jnp.all(jnp.isfinite(y))


def test_bronet_forward_and_out_features():
    net = BroNet(in_features=6, key=SEED, depth=2, width_size=32)
    assert net.out_features == 32
    assert net.depth == 2

    y = net(jnp.ones((6,)))
    assert y.shape == (32,)
    assert jnp.all(jnp.isfinite(y))


def test_bronet_with_params_factory():
    factory = BroNet.with_params(depth=3, width_size=16)
    net = factory(in_features=5, key=SEED)
    assert isinstance(net, BroNet)
    assert len(net.layers) == 2 + 3
    assert net(jnp.ones((5,))).shape == (16,)


def test_simba_forward_and_out_features():
    net = SimBa(in_features=6, key=SEED, depth=2, width_size=32)
    assert net.out_features == 32
    assert net.depth == 2
    assert len(net.blocks) == 2

    y = net(jnp.ones((6,)))
    assert y.shape == (32,)
    assert jnp.all(jnp.isfinite(y))
