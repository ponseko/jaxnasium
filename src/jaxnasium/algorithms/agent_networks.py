import logging
from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium.algorithms.architectures import CNN, SimBa
from jaxnasium.algorithms.core import (
    POLICY_HEAD_WEIGHT_INIT,
    VALUE_HEAD_WEIGHT_INIT,
    CategoricalLayer,
    NormalLayer,
    PyTreeActionNetwork,
    PyTreeObsSpaceNetwork,
    PyTreeQValueNetwork,
    QLayer,
    set_weight_bias,
)
from jaxnasium.algorithms.types import Network, OutSizedNetwork

logger = logging.getLogger(__name__)

_DEFAULT_ARCHITECTURE_2D = CNN.with_params(
    out_channels=(32, 64, 64),
    kernel_sizes=(3, 3, 3),
    strides=(1, 1, 1),
    padding=(1, 1, 1),
)


"""
The base Reinforcement Learning Network classes (actor, V-network, Q-network).
Each of these consist of three components:
    - An observation processor (PyTreeObsSpaceNetwork)
        This accepts a PyTree of observation spaces and builds a network per observation space.
        1d observation spaces are processed via the configured ``architecture_1d`` network.
        2d observation spaces are processed via the configured ``architecture_2d`` network.
        The output of each observation processor is concatenated into a single 1d vector.
    - A body (default SimBa) = A shared body network that takes the output of the observation processor and processes it jointly.
    - An output processor (PyTreeOutputNetwork)
        This accepts a PyTree of output spaces and builds a network per output space.
        Automtically builds a discrete or continuous output network based on the output space.
        Returns the output of each output network in the same PyTree structure as the action space.
"""


class ActorNetwork(eqx.Module):
    obs_processor: PyTreeObsSpaceNetwork
    body: OutSizedNetwork
    output_layers: PyTreeActionNetwork

    def __init__(
        self,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        *,
        key: PRNGKeyArray,
        body: Callable[..., OutSizedNetwork] = SimBa,
        obs_architecture_1d: Callable[..., Network] = eqx.nn.Identity,
        obs_architecture_2d: Callable[..., Network] = _DEFAULT_ARCHITECTURE_2D,
        discrete_output_layer: Callable[..., Network] = CategoricalLayer,
        continuous_output_layer: Callable[..., Network] = NormalLayer,
        assume_independent_output: bool = True,
    ):
        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)

        if isinstance(obs_space, jym.AgentObservation):
            obs_space = obs_space.observation

        self.obs_processor = PyTreeObsSpaceNetwork(
            obs_space,
            key=obs_key,
            architecture_1d=obs_architecture_1d,
            architecture_2d=obs_architecture_2d,
        )

        self.body = body(self.obs_processor.out_features, key=body_key)

        self.output_layers = PyTreeActionNetwork(
            self.body.out_features,
            output_space,
            key=output_key,
            discrete_output_layer=discrete_output_layer,
            continuous_output_layer=continuous_output_layer,
            assume_independent=assume_independent_output,
        )

        body_key, head_key = jax.random.split(wb_key)
        self.obs_processor, self.body, self.output_layers = set_weight_bias(
            body_key, (self.obs_processor, self.body, self.output_layers)
        )
        self.output_layers = set_weight_bias(
            head_key, self.output_layers, weight_init=POLICY_HEAD_WEIGHT_INIT
        )

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        x = self.obs_processor(x, key=key)
        x = self.body(x, key=key)
        return self.output_layers(x, action_mask=action_mask, key=key)


class ValueNetwork(eqx.Module):
    obs_processor: PyTreeObsSpaceNetwork
    body: OutSizedNetwork
    output_layers: Network

    def __init__(
        self,
        obs_space: PyTree[jym.Space],
        *,
        key: PRNGKeyArray,
        body: Callable[..., OutSizedNetwork] = SimBa,
        obs_architecture_1d: Callable[..., Network] = eqx.nn.Identity,
        obs_architecture_2d: Callable[..., Network] = _DEFAULT_ARCHITECTURE_2D,
        output_layer_type: Callable[..., Network] = eqx.Partial(
            eqx.nn.Linear, out_features="scalar"
        ),
    ):
        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)

        if isinstance(obs_space, jym.AgentObservation):
            obs_space = obs_space.critic_input

        self.obs_processor = PyTreeObsSpaceNetwork(
            obs_space,
            key=obs_key,
            architecture_1d=obs_architecture_1d,
            architecture_2d=obs_architecture_2d,
        )

        self.body = body(self.obs_processor.out_features, key=body_key)

        self.output_layers = output_layer_type(
            in_features=self.body.out_features, key=output_key
        )

        body_key, head_key = jax.random.split(wb_key)
        (self.obs_processor, self.body, self.output_layers) = set_weight_bias(
            key=body_key, network=(self.obs_processor, self.body, self.output_layers)
        )
        self.output_layers = set_weight_bias(
            key=head_key, network=self.output_layers, weight_init=VALUE_HEAD_WEIGHT_INIT
        )

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        if isinstance(x, jym.AgentObservation):
            x = x.critic_input

        x = self.obs_processor(x, key=key)
        x = self.body(x, key=key)
        return self.output_layers(x, key=key)


class QValueNetwork(eqx.Module):
    obs_processor: PyTreeObsSpaceNetwork
    body: OutSizedNetwork
    output_layers: PyTreeQValueNetwork

    include_action_in_input: bool = eqx.field(static=True)

    def __init__(
        self,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        *,
        key: PRNGKeyArray,
        body: Callable[..., OutSizedNetwork] = SimBa,
        obs_architecture_1d: Callable[..., Network] = eqx.nn.Identity,
        obs_architecture_2d: Callable[..., Network] = _DEFAULT_ARCHITECTURE_2D,
        output_layer_type: Callable[..., Network] = QLayer.with_params(
            layer_type=eqx.nn.Linear
        ),
    ):
        if isinstance(obs_space, jym.AgentObservation):
            obs_space = obs_space.critic_input

        is_continuous = [isinstance(s, jym.Box) for s in jax.tree.leaves(output_space)]
        if any(is_continuous):
            self.include_action_in_input = True
            obs_space = {"_OBSERVATION": obs_space, "_ACTION": output_space}
        else:
            self.include_action_in_input = False

        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)

        self.obs_processor = PyTreeObsSpaceNetwork(
            obs_space,
            key=obs_key,
            architecture_1d=obs_architecture_1d,
            architecture_2d=obs_architecture_2d,
        )

        self.body = body(self.obs_processor.out_features, key=body_key)

        self.output_layers = PyTreeQValueNetwork(
            self.body.out_features,
            output_space,
            key=output_key,
            output_layer=output_layer_type,
        )

        body_key, head_key = jax.random.split(wb_key)
        (self.obs_processor, self.body, self.output_layers) = set_weight_bias(
            key=body_key, network=(self.obs_processor, self.body, self.output_layers)
        )
        self.output_layers = set_weight_bias(
            key=head_key, network=self.output_layers, weight_init=VALUE_HEAD_WEIGHT_INIT
        )

    def __call__(
        self, x, action=None, *, key: PRNGKeyArray | None = None
    ) -> Array | PyTree[Array]:
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.critic_input

        if self.include_action_in_input:
            assert action is not None, "Action not provided in continuous Q network."
            x = {"_OBSERVATION": x, "_ACTION": action}

        x = self.obs_processor(x, key=key)
        x = self.body(x, key=key)
        return self.output_layers(x, action_mask=action_mask, key=key)


AdvantageNetwork = QValueNetwork
