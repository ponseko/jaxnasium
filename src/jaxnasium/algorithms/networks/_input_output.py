import logging
from typing import Callable, List, Literal, Tuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import PRNGKeyArray, PyTree, PyTreeDef

import jaxnasium as jym
import jaxnasium.tree
from jaxnasium.algorithms.utils import TanhNormalFactory

from ._architectures import CNN, Identity

logger = logging.getLogger(__name__)


class AutoAgentObservationNet(eqx.Module):
    """Input network for **single and multi** observation space environments.

    Builds a separate input network for each observation space. Automatically
    builds a 1d architecture for 1d observation spaces (default=None/Identity) and
    a 2d architecture for 2d observation spaces (default=NatureCNN-like).

    During a forward call this network simply returns a jax.tree.map over all input spaces
    and concatenates the outputs of all input networks as a single 1d vector.
    """

    networks: PyTree[eqx.Module]

    num_observation_spaces: int = eqx.field(static=True)
    input_structure: PyTreeDef = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, obs_space: PyTree[jym.Space], **kwargs):
        def create_obs_processor(key: PRNGKeyArray, obs_space: jym.Space, **kwargs):
            if obs_space.shape == () or len(obs_space.shape) == 1:
                return self._create_1d_obs_processor(key, obs_space, **kwargs)
            elif len(obs_space.shape) == 3 or len(obs_space.shape) == 2:
                return self._create_2d_obs_processor(key, obs_space, **kwargs)
            raise ValueError(f"Unsupported observation space shape: {obs_space.shape}")

        self.num_observation_spaces = len(jax.tree.leaves(obs_space))
        self.input_structure = jax.tree.structure(obs_space)

        keys = optax.tree.split_key_like(key, obs_space)
        self.networks = jax.tree.map(
            lambda o, k: create_obs_processor(k, o, **kwargs),
            obs_space,
            keys,
        )

        # Infer the output feature size
        dummy_obs = jax.tree.map(lambda o: jnp.zeros(o.shape), obs_space)
        f = lambda obs: jnp.atleast_1d(self(obs))
        self.out_features = jax.eval_shape(f, dummy_obs).shape[0]

    def __call__(self, x):
        # Convert non-float inputs to float32
        x = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), x)

        outputs = jax.tree.map(
            lambda layer, x: layer(x),
            self.networks,
            x,
            is_leaf=lambda x: isinstance(x, eqx.Module),
        )
        return jaxnasium.tree.concatenate(outputs)

    def _create_1d_obs_processor(
        self,
        key: PRNGKeyArray,
        obs_space: jym.Space,
        architecture_1d: Literal["identity"] = "identity",
        **kwargs,
    ):
        if architecture_1d.lower() in ["identity"]:
            return Identity()

        # elif architecture == "broNet":

        raise ValueError(f"Unsupported 1d architecture: {architecture_1d}")

    def _create_2d_obs_processor(
        self,
        key: PRNGKeyArray,
        obs_space: jym.Space,
        architecture_2d: Literal["cnn"] = "cnn",
        cnn_hidden_sizes: Tuple[int, ...] = (32, 64, 64),
        cnn_kernel_sizes: Tuple[int, ...] = (3, 3, 2),
        cnn_strides: Tuple[int, ...] = (1, 1, 1),
        cnn_padding: Tuple[int, ...] = (0, 0, 0),
        **kwargs,
    ):
        if architecture_2d.lower() in ["cnn", "naturecnn"]:
            return CNN(
                key,
                obs_space,
                cnn_hidden_sizes,
                cnn_kernel_sizes,
                cnn_strides,
                cnn_padding,
            )
        # elif architecture == "hadamax":
        #     pass

        raise ValueError(f"Unsupported 2d architecture: {architecture_2d}")


class AutoAgentOutputNet(eqx.Module):
    """Output network for **single and multi** action space environments.

    Will build an individual output head for each action space and inner individual
    output heads in multidimensional spaces. During a forward call this network
    simply returns a jax.tree.map over all output spaces.
    Inner multidimensional spaces are vmapped, and are expected to be 1d and homogeneous.
    """

    networks: PyTree[eqx.Module]
    num_action_spaces: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_features: int,
        output_space: PyTree[jym.Space],
        *,
        discrete_output_dist: Literal["categorical",] | None = "categorical",
        continuous_output_dist: Literal["normal", "tanhnormal"] | None = "normal",
        **kwargs,
    ):
        def create_output_network(key: PRNGKeyArray, output_space: jym.Space):
            is_discrete = hasattr(output_space, "n") or hasattr(output_space, "nvec")
            if is_discrete:
                return DiscreteOutputNetwork(
                    key, in_features, output_space, distribution=discrete_output_dist
                )

            is_continu = hasattr(output_space, "low") and hasattr(output_space, "high")
            if is_continu:
                return ContinuousOutputNetwork(
                    key,
                    in_features,
                    output_space,  # type: ignore
                    distribution=continuous_output_dist,
                )

            else:
                raise ValueError(f"Unsupported action space: {output_space}")

        self.num_action_spaces = len(jax.tree.leaves(output_space))
        keys = optax.tree.split_key_like(key, output_space)
        self.networks = jax.tree.map(
            lambda a, k: create_output_network(k, a), output_space, keys
        )

    def __call__(self, x, action_mask):
        if action_mask is None:  # Dummy action mask if not provided
            action_mask = jax.tree.map(
                lambda _: None,
                self.networks,
                is_leaf=lambda x: isinstance(x, eqx.Module),
            )

        return jax.tree.map(
            lambda layer, mask: layer(x, mask),
            self.networks,
            action_mask,
            is_leaf=lambda x: isinstance(x, eqx.Module),
        )


class DiscreteOutputNetwork(eqx.Module):
    layers: List[eqx.nn.Linear]
    distribution: Callable[..., distrax.Distribution] | None = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_features: int,
        output_space: jym.Space,
        distribution: Literal["categorical"] | None,
    ):
        if distribution is None:
            self.distribution = None
        elif distribution == "categorical":
            self.distribution = distrax.Categorical
        else:
            raise ValueError(f"Unsupported discrete distribution: {distribution}")

        # Get n (Discrete) or nvec (MultiDiscrete)
        num_outputs = getattr(output_space, "n", getattr(output_space, "nvec", None))
        if num_outputs is None:
            raise ValueError(f"Unsupported action space: {output_space}")

        # We create a list of outputs: [n] for Discrete, [n,n,...] for MultiDiscrete
        num_outputs = np.atleast_1d(num_outputs).tolist()
        assert len(set(num_outputs)) == 1, (
            "Only homogeneous MultiDiscrete spaces supported to be supported for vmap."
            f" (all nvec elements must be the same, got {num_outputs}) "
            "For heterogeneous spaces, use a composite of spaces instead."
        )

        # Then we create an output head per element (1 for Discrete)
        # These heads can be stacked + vmapped in the forward call
        keys = optax.tree.split_key_like(key, num_outputs)
        self.layers = jax.tree.map(
            lambda o, k: eqx.nn.Linear(in_features, o, key=k), num_outputs, keys
        )

    def __call__(self, x, action_mask):
        if len(self.layers) == 1:  # single dimensional output
            logits = self.layers[0](x)
        else:
            stacked_layers = jaxnasium.tree.stack(self.layers)
            logits = jax.vmap(lambda layer: layer(x))(stacked_layers)

        if action_mask is not None:
            logits = self._apply_action_mask(logits, action_mask)

        if self.distribution is not None:
            return self.distribution(logits=logits)
        return logits

    def _apply_action_mask(self, logits, action_mask):
        """Apply the action mask to the output of the network.

        NOTE: This requires a (multi-)discrete action space.
        NOTE: Currently, action mask is assumed to be a PyTree of the same structure as the action space.
            Therefore, masking is not supported when the mask is dependent on another action.
        """

        BIG_NEGATIVE = -1e9
        masked_logits = jax.tree.map(
            lambda a, mask: ((jnp.ones_like(a) * BIG_NEGATIVE) * (1 - mask)) + a,
            logits,
            action_mask,
        )
        return masked_logits


class ContinuousOutputNetwork(eqx.Module):
    layers: List[eqx.nn.Linear]
    distribution: Callable[..., distrax.Distribution] | None = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_features: int,
        output_space: jym.Box,
        distribution: Literal["normal", "tanhnormal"] | None,
    ):
        assert hasattr(output_space, "low") and hasattr(output_space, "high"), (
            "Continuous action space is assumed to be a `Box`-like and "
            "must have 'low' and 'high' and `shape` attributes."
        )

        low = np.array(output_space.low, dtype=float)
        high = np.array(output_space.high, dtype=float)
        if distribution is None:
            self.distribution = None
        elif distribution.lower() == "normal":
            self.distribution = distrax.Normal
        elif distribution.lower() == "tanhnormal":
            self.distribution = TanhNormalFactory(low=low, high=high)
        else:
            raise ValueError(f"Unsupported continuous distribution: {distribution}")

        num_outputs = np.ones(output_space.shape, dtype=int)
        if distribution is not None:
            num_outputs = num_outputs * 2  # mean, std
            # NOTE: this assumes Normal / TanhNormal distribution with 2 outputs

        # We create a list of output heads per dimension
        # These heads can be stacked + vmapped in the forward call
        num_outputs = num_outputs.tolist()

        assert len(set(num_outputs)) == 1, (
            "Only homogeneous Box spaces supported to be supported for vmap."
            f" (all shape elements must be the same, got {num_outputs}) "
            "For heterogeneous spaces, use a composite of spaces instead."
        )

        keys = optax.tree.split_key_like(key, num_outputs)
        self.layers = jax.tree.map(
            lambda o, k: eqx.nn.Linear(in_features, o, key=k), num_outputs, keys
        )

    def __call__(self, x, action_mask):
        if action_mask is not None:
            logging.debug("Action mask provided for continuous action space, ignoring.")

        if len(self.layers) == 1:  # single dimensional output
            logits = self.layers[0](x)
        else:
            stacked_layers = jaxnasium.tree.stack(self.layers)
            logits = jax.vmap(lambda layer: layer(x))(stacked_layers)

        if self.distribution is None:
            return logits.squeeze()

        mean = logits[..., 0]
        log_std = logits[..., 1]
        log_std = jnp.clip(log_std, -20, 2)
        std = jax.nn.softplus(log_std)

        return self.distribution(mean, std)
