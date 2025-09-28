import logging

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium.algorithms.utils import DistraxContainer, rl_initialization

from ._architectures import MLP
from ._input_output import AutoAgentObservationNet, AutoAgentOutputNet

logger = logging.getLogger(__name__)


"""
The base Reinforcement Learning Network classes (actor, V-network, Q-network).
Each of these consist of three components:
    - An observation processor (AutoAgentObservationNet)
        This accepts a PyTree of observation spaces and builds a network per observation space.
        In case of a single 1d observation space, this will simply be the Identity network.
        In case of a 2d observation space, this will be a CNN.
        The output of each observation processor is concatenated into a single 1d vector.
    - A MLP =
        This is a simple MLP network that takes the output of the observation processor and passes it through a MLP.
    - An output processor (AutoAgentOutputNet)
        This accepts a PyTree of output spaces and builds a network per output space.
        Automtically builds a discrete or continuous output network based on the output space.
        Returns the output of each output network in the same PyTree structure as the action space.
"""


class ActorNetwork(eqx.Module):
    obs_processor: AutoAgentObservationNet
    mlp: MLP
    output_layers: AutoAgentOutputNet

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        **network_kwargs,
    ):
        key_in, key_mlp, key_out = jax.random.split(key, 3)

        # Automatically builds a network per observation space.
        # In case of a single 1d observation space, this will simply be the Identity network.
        # 2d observation spaces will be processed with a CNN.
        self.obs_processor = AutoAgentObservationNet(
            key_in, obs_space, **network_kwargs
        )
        self.mlp = MLP(key_mlp, self.obs_processor.out_features, **network_kwargs)
        self.output_layers = AutoAgentOutputNet(
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
    obs_processor: AutoAgentObservationNet
    mlp: MLP
    output_layers: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        **network_kwargs,
    ):
        key_in, key_mlp, key_out = jax.random.split(key, 3)

        # Automatically builds a network per observation space.
        # In case of a single 1d observation space, this will simply be the Identity network.
        # 2d observation spaces will be processed with a CNN.
        self.obs_processor = AutoAgentObservationNet(
            key_in, obs_space, **network_kwargs
        )
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
    obs_processor: AutoAgentObservationNet
    mlp: MLP
    output_layers: AutoAgentOutputNet

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

        self.obs_processor = AutoAgentObservationNet(
            key_in, obs_space, **network_kwargs
        )
        self.mlp = MLP(key_mlp, self.obs_processor.out_features, **network_kwargs)
        self.output_layers = AutoAgentOutputNet(
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
