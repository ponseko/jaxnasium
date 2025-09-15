import logging

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium.algorithms.utils import DistraxContainer, rl_initialization

from ._architectures import MLP
from ._input_output import AgentObservationNet, AgentOutputNet

logger = logging.getLogger(__name__)


class ActorNetwork(eqx.Module):
    obs_processor: AgentObservationNet
    mlp: MLP
    output_layers: AgentOutputNet

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        **network_kwargs,
    ):
        key_in, key_mlp, key_out = jax.random.split(key, 3)

        self.obs_processor = AgentObservationNet(key_in, obs_space, **network_kwargs)
        self.mlp = MLP(key_mlp, self.obs_processor.out_features, **network_kwargs)
        self.output_layers = AgentOutputNet(
            key_out, self.mlp.out_features, output_space, **network_kwargs
        )

        # Set all biases to 0 instead of eqx default
        self.obs_processor = rl_initialization(key_in, self.obs_processor)
        self.mlp = rl_initialization(key_mlp, self.mlp)
        self.output_layers = rl_initialization(key_out, self.output_layers)

    def __call__(self, x):
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        x = self.obs_processor(x)
        x = self.mlp(x)
        action_dists = self.output_layers(x, action_mask)
        if isinstance(action_dists, distrax.Distribution):
            return action_dists  # Single distribution

        # Else return a grouped container of distributions
        return DistraxContainer(action_dists)


class ValueNetwork(eqx.Module):
    obs_processor: AgentObservationNet
    mlp: MLP
    output_layers: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        **network_kwargs,
    ):
        key_in, key_mlp, key_out = jax.random.split(key, 3)

        self.obs_processor = AgentObservationNet(key_in, obs_space, **network_kwargs)
        self.mlp = MLP(key_mlp, self.obs_processor.out_features, **network_kwargs)
        self.output_layers = eqx.nn.Linear(self.mlp.out_features, 1, key=key_out)

        # Set all biases to 0 instead of eqx default
        self.obs_processor = rl_initialization(key_in, self.obs_processor)
        self.mlp = rl_initialization(key_mlp, self.mlp)
        self.output_layers = rl_initialization(key_out, self.output_layers)

    def __call__(self, x):
        if isinstance(x, jym.AgentObservation):
            x = x.observation

        x = self.obs_processor(x)
        x = self.mlp(x)
        out = self.output_layers(x)
        return jnp.squeeze(out, axis=-1)


class QValueNetwork(eqx.Module):
    obs_processor: AgentObservationNet
    mlp: MLP
    output_layers: AgentOutputNet

    include_action_in_input: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        **network_kwargs,
    ):
        is_continuous = [isinstance(s, jym.Box) for s in jax.tree.leaves(output_space)]
        if any(is_continuous):
            self.include_action_in_input = True
            if not all(is_continuous):
                logging.warning(
                    "Mixed action spaces with continuous QNetwork may have adverse training effects"
                )
            obs_space = {"_OBSERVATION": obs_space, "_ACTION": output_space}

        key_in, key_mlp, key_out = jax.random.split(key, 3)

        self.obs_processor = AgentObservationNet(key_in, obs_space, **network_kwargs)
        self.mlp = MLP(key_mlp, self.obs_processor.out_features, **network_kwargs)
        self.output_layers = AgentOutputNet(
            key_out,
            self.mlp.out_features,
            output_space,
            discrete_output_dist=None,
            continuous_output_dist=None,
            **network_kwargs,
        )

        # Set all biases to 0 instead of eqx default
        self.obs_processor = rl_initialization(key_in, self.obs_processor)
        self.mlp = rl_initialization(key_mlp, self.mlp)
        self.output_layers = rl_initialization(key_out, self.output_layers)

    def __call__(self, x, action=None):
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        if self.include_action_in_input:
            assert action is not None, "Action not provided in continuous Q network."
            x = {"_OBSERVATION": x, "_ACTION": action}

        x = self.obs_processor(x)
        x = self.mlp(x)
        q_values = self.output_layers(x, action_mask)
        return q_values


AdvantageCriticNetwork = QValueNetwork
