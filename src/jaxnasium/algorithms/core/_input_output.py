import logging
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Literal, Self

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

import jaxnasium as jym

from ..architectures import CNN, Identity
from ..types import (
    ContinuousSpaceLike,
    DiscreteSpaceLike,
    Network,
    SpaceLike,
)
from ._distributions import TanhNormal, make_independent

logger = logging.getLogger(__name__)


def _is_space_discrete(space: SpaceLike) -> bool:
    return hasattr(space, "n") or hasattr(space, "nvec")


def _is_space_continuous(space: SpaceLike) -> bool:
    return hasattr(space, "low") and hasattr(space, "high")


def _is_callable_module(x) -> bool:
    """Check if x is a callable eqx.Module."""
    return isinstance(x, eqx.Module) and callable(x)


def _is_distribution(x: Any) -> bool:
    return isinstance(x, distrax.Distribution)


def _assert_homogeneous_output_space(num_outputs: list[int]):
    assert len(set(num_outputs)) == 1, (
        "Only homogeneous multi-dimensional output spaces supported to be supported for vmap."
        f" (all nvec elements must be the same, got {num_outputs}) "
        "For heterogeneous spaces, use a composite of spaces instead. "
        "(E.g. {'action1': (Discrete(n), 'action2': Discrete(m), ...}) "
        "This can be done through `jym.MultiDiscreteToListDiscreteWrapper` "
        "or by flattening the action space using `jym.FlattenActionWrapper`."
    )


def _apply_action_mask(logits: Array, action_mask: Array) -> Array:
    """Mask out invalid actions by pushing their logits/values to -inf.

    NOTE: This requires a (multi-)discrete output space.
    NOTE: The mask is assumed to be a PyTree of the same structure as the
        output space. Masking dependent on another action is not supported.
    """
    try:
        BIG_NEGATIVE = -1e9
        return jax.tree.map(
            lambda a, mask: a + (BIG_NEGATIVE * (1 - mask)),
            logits,
            action_mask,
        )
    except Exception as e:
        logger.error(f"Failed to apply action mask: {e}")
        raise ValueError(
            "Failed to apply action mask with the above error. "
            "logits and action_mask must have the same pytree structure and shapes. "
            f"logits: {jax.tree.structure(logits)} with shapes {jax.tree.map(lambda x: x.shape, logits)} "
            f"action_mask: {jax.tree.structure(action_mask)} with shapes {jax.tree.map(lambda x: x.shape, action_mask)}."
            "Further note that action masking is not supported for actions which are conditionally dependent. "
            "In this case, flatten the action space (e.g. using a `jym.FlattenActionWrapper`) or use a custom model."
        )


def _discrete_num_outputs(output_space: DiscreteSpaceLike) -> list[int]:
    """Number of outputs per dimension: `[n]` (Discrete) or `[n, n, ...]` (MultiDiscrete)."""
    num_outputs = getattr(output_space, "n", getattr(output_space, "nvec", None))
    if num_outputs is None:
        raise ValueError(f"Unsupported discrete output space: {output_space}")
    num_outputs = np.atleast_1d(num_outputs).tolist()
    _assert_homogeneous_output_space(num_outputs)
    return num_outputs


def _num_action_dims(output_shape: tuple[int, ...]) -> int:
    return 1 if output_shape == () else int(np.prod(output_shape))


def _create_per_dimension_layers(
    in_features: int,
    num_outputs: list[int],
    key: PRNGKeyArray,
    layer_type: Callable[..., Network],
) -> list[Network]:
    """One (homogeneous) output layer per output dimension."""
    keys = optax.tree.split_key_like(key, num_outputs)
    return jax.tree.map(
        lambda o, k: layer_type(in_features, o, key=k), num_outputs, keys
    )


def _forward_per_dimension(
    layers: list[Network], x, *, key: PRNGKeyArray | None = None
) -> Array:
    """Forward `x` through one layer per *output dimension*, as a single stacked call."""
    if len(layers) == 1:  # single-dimensional output space
        return layers[0](x, key=key)

    stacked = jym.tree.stack(layers)
    if key is None:
        return jax.vmap(lambda layer: layer(x, key=None))(stacked)
    keys = jax.random.split(key, len(layers))
    return jax.vmap(lambda layer, k: layer(x, key=k))(stacked, jnp.stack(keys))


class FlattenLayer(eqx.Module):
    """Flattens the input array to a 1D vector."""

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        return x.reshape(-1)


class PyTreeFlattenLayer(eqx.Module):
    """Flattens every leaf of a pytree into a single 1D vector."""

    def __call__(self, x: Any, *args, **kwargs) -> Array:
        leaves = jax.tree.leaves(x)
        flat = [jnp.ravel(jnp.asarray(leaf, dtype=jnp.float32)) for leaf in leaves]
        return jnp.concatenate(flat)


def _infer_out_features(self, obs):
    return jnp.atleast_1d(self(obs))


class PyTreeObsSpaceNetwork(eqx.Module):
    """Builds a separate observation network for each observation space.

    Automatically builds a given 1d architecture for 1d observation spaces
    and a given 2d architecture for 2d observation spaces.

    During a forward call this network simply returns a jax.tree.map over all observation spaces
    and concatenates the outputs of all observation networks as a single 1d vector.

    **Arguments:**
    - `obs_space`: The observation space(s) to build networks for.
    - `key`: A PRNG key for reproducibility.
    - `architecture_1d`: The architecture to use for 1d observation spaces.
    - `architecture_2d`: The architecture to use for 2d observation spaces.
    """

    networks: PyTree[Network]

    num_observation_spaces: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        obs_space: PyTree[SpaceLike],
        *,
        key: PRNGKeyArray,
        architecture_1d: Callable[..., Network] = Identity,
        architecture_2d: Callable[..., Network] = CNN.with_params(
            out_channels=(32, 64, 64),
            kernel_sizes=(3, 3, 2),
            strides=(1, 1, 1),
            padding=(0, 0, 0),
        ),
    ):
        def create_obs_processor(key: PRNGKeyArray, obs_space: SpaceLike):
            if obs_space.shape == () or len(obs_space.shape) == 1:
                return self._create_1d_obs_processor(key, obs_space, architecture_1d)
            elif len(obs_space.shape) == 3:
                return self._create_2d_obs_processor(key, obs_space, architecture_2d)
            elif len(obs_space.shape) == 2:
                raise ValueError(
                    f"2D observation space shape without a channel axis ({obs_space.shape}) detected. "
                    "Either add a channel axis or use a FlattenObservationWrapper."
                )
            raise ValueError(f"Unsupported observation space shape: {obs_space.shape}")

        # For continuous action space where a Q network is used, the action is included
        # in the observation space. We process it seperately by flattening only.
        action_input = None
        original_obs_space = obs_space
        if isinstance(obs_space, dict):
            original_obs_space = obs_space.copy()
            action_input = obs_space.pop("_ACTION", None)

        self.num_observation_spaces = len(jax.tree.leaves(obs_space))

        keys = optax.tree.split_key_like(key, obs_space)
        self.networks = jax.tree.map(
            lambda o, k: create_obs_processor(k, o),
            obs_space,
            keys,
        )

        # Add back the action input if present
        if action_input is not None:
            action_input_layer = PyTreeFlattenLayer()
            self.networks["_ACTION"] = action_input_layer

        # Infer the output feature size
        dummy_obs = jax.tree.map(lambda o: jnp.zeros(o.shape), original_obs_space)
        self.out_features = jax.eval_shape(_infer_out_features, self, dummy_obs).shape[
            0
        ]

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        x = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), x)

        outputs = jax.tree.map(
            lambda layer, x: layer(x, key=key),
            self.networks,
            x,
            is_leaf=_is_callable_module,
        )
        return jym.tree.ravel(outputs)

    def _create_1d_obs_processor(
        self,
        key: PRNGKeyArray,
        obs_space: SpaceLike,
        architecture: Callable[..., Network],
    ):
        try:
            if obs_space.shape == ():
                in_features = 1
            elif len(obs_space.shape) == 1:
                in_features = obs_space.shape[0]
            else:
                raise ValueError(
                    f"Unsupported observation space shape: {obs_space.shape}"
                )
            return architecture(in_features, key=key)
        except AttributeError:
            raise ValueError(f"Unsupported observation space {obs_space}")

    def _create_2d_obs_processor(
        self,
        key: PRNGKeyArray,
        obs_space: SpaceLike,
        architecture: Callable[..., Network],
    ):
        try:
            if len(obs_space.shape) == 3:
                # Honour an explicit `channels_axis` else we infer it.
                bound = getattr(architecture, "keywords", None) or {}
                if "channels_axis" in bound:
                    return architecture(obs_space.shape, key=key)
                channels_axis = self._infer_channels_axis(obs_space)
                return architecture(
                    obs_space.shape,
                    key=key,
                    channels_axis=channels_axis,
                )
            raise ValueError(f"Unsupported observation space shape: {obs_space.shape}")
        except AttributeError:
            raise ValueError(f"Unsupported observation space {obs_space}")

    def _infer_channels_axis(self, obs_space: SpaceLike):
        """
        Attempts to infer the channels axis from the observation space shape.
        By checking for the smallest dimension and assuming it to be the channels dimension.
        """
        if len(obs_space.shape) != 3:
            raise ValueError(
                "`infer_channels_axis` requires a (C, H, W) or (H, W, C) observation space."
            )
        c0, c1, c2 = obs_space.shape
        is_chw = c0 < min(c1, c2)
        is_hwc = c2 < min(c0, c1)
        if is_chw and not is_hwc:
            return "first"
        if is_hwc and not is_chw:
            return "last"
        raise ValueError(
            f"Cannot infer channel axis from shape {obs_space.shape}. "
            "Pass `channels_axis` explicitly  ('first' or 'last')."
        )

    @classmethod
    def with_params(
        cls,
        *,
        architecture_1d: Callable[..., Network] = Identity,
        architecture_2d: Callable[..., Network] = CNN.with_params(
            out_channels=(32, 64, 64),
            kernel_sizes=(3, 3, 2),
            strides=(1, 1, 1),
            padding=(0, 0, 0),
        ),
    ) -> Callable[..., Self]:
        return eqx.Partial(
            cls,
            architecture_1d=architecture_1d,
            architecture_2d=architecture_2d,
        )


## Output:


class CategoricalLayer(eqx.Module):
    """A layer type returning a `distrax.Categorical` distribution"""

    layers: list[Network]
    dtype: Any = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        output_space: DiscreteSpaceLike,
        *,
        key: PRNGKeyArray,
        layer_type: Callable[..., Network] = eqx.nn.Linear,
    ):
        self.layers = _create_per_dimension_layers(
            in_features, _discrete_num_outputs(output_space), key, layer_type
        )
        self.dtype = output_space.dtype

    def __call__(self, x, action_mask=None, *, key: PRNGKeyArray | None = None):
        logits = _forward_per_dimension(self.layers, x, key=key)
        if action_mask is not None:
            logits = _apply_action_mask(logits, action_mask)
        return distrax.Categorical(logits=logits, dtype=self.dtype)

    @classmethod
    def with_params(
        cls, *, layer_type: Callable[..., Network] = eqx.nn.Linear
    ) -> Callable[..., Self]:
        return eqx.Partial(cls, layer_type=layer_type)


class _ConstantLogStd(eqx.Module):
    """A free `log_std` parameter, independent of state (CleanRL/SB3 PPO's choice)."""

    log_std: Array

    def __call__(self, x=None, *, key: PRNGKeyArray | None = None) -> Array:
        return self.log_std


class _GaussianOutputLayer(eqx.Module):
    """Shared class for Gaussian-family output layers over a `Box` space.

    **Arguments:**
    - `in_features`: The number of input features to the layer.
    - `output_space`: The output space to build the layer for.
    - `key`: A PRNG key.
    - `state_dependent_std`: Whether to use a state-dependent `log_std` head or a free parameter.
    - `log_std_min`: The minimum value for the `log_std` head.
    - `log_std_max`: The maximum value for the `log_std` head.
    - `log_std_init`: The initial value for the free `log_std` parameter (only used when state_dependent_std is disabled).
    - `layer_type`: The layer type to use for the `mean` and `log_std` heads.
    """

    mean: Network
    log_std: Network

    output_space: ContinuousSpaceLike = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        output_space: ContinuousSpaceLike,
        *,
        key: PRNGKeyArray,
        state_dependent_std: bool = True,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        log_std_init: float = 0.0,
        layer_type: Callable[..., Network] = eqx.nn.Linear,
    ):
        self.output_space = output_space
        self.output_shape = output_space.shape
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        num_dims = _num_action_dims(self.output_shape)
        mean_key, log_std_key = jax.random.split(key)
        self.mean = layer_type(in_features, num_dims, key=mean_key)

        if state_dependent_std:
            self.log_std = layer_type(in_features, num_dims, key=log_std_key)
        else:
            self.log_std = _ConstantLogStd(jnp.full(self.output_shape, log_std_init))

    @abstractmethod
    def _distribution(self, mean: Array, std: Array) -> distrax.Distribution: ...

    def __call__(self, x, action_mask=None, *, key: PRNGKeyArray | None = None):
        if action_mask is not None:
            logger.debug("Action mask provided for continuous space, ignoring.")
        mean = self.mean(x, key=key).reshape(self.output_shape)
        log_std = self.log_std(x, key=key).reshape(self.output_shape)
        if not isinstance(self.log_std, _ConstantLogStd):  # NOTE: we could squash this?
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                jnp.tanh(log_std) + 1.0
            )  # tanh squash to [log_std_min, log_std_max]
        return self._distribution(mean, jnp.exp(log_std))

    @classmethod
    def with_params(
        cls,
        *,
        state_dependent_std: bool = True,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        log_std_init: float = 0.0,
        layer_type: Callable[..., Network] = eqx.nn.Linear,
    ) -> Callable[..., Self]:
        return eqx.Partial(
            cls,
            state_dependent_std=state_dependent_std,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            log_std_init=log_std_init,
            layer_type=layer_type,
        )


class NormalLayer(_GaussianOutputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("state_dependent_std", False)
        super().__init__(*args, **kwargs)

    def _distribution(self, mean: Array, std: Array) -> distrax.Distribution:
        return distrax.Normal(mean, std)


class TanhNormalLayer(_GaussianOutputLayer):
    def _distribution(self, mean: Array, std: Array) -> distrax.Distribution:
        low, high = self.output_space.low, self.output_space.high
        return TanhNormal(mean, std, shift=(high + low) / 2.0, scale=(high - low) / 2.0)


class QLayer(eqx.Module):
    """Latent -> raw values (no distribution).

    Mode is inferred from the output space:

    - discrete: one value per action, i.e. `Q(s, .)`.
    - continuous: a single scalar `Q(s, a)`, which requires the action as input.
    """

    layers: list[Network]
    mode: Literal["discrete", "continuous"] = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        output_space: SpaceLike,
        *,
        key: PRNGKeyArray,
        layer_type: Callable[..., Network] = eqx.nn.Linear,
    ):
        if _is_space_discrete(output_space):
            self.mode = "discrete"
            self.layers = _create_per_dimension_layers(
                in_features,
                _discrete_num_outputs(output_space),  # type: ignore[arg-type]
                key,
                layer_type,
            )
        elif _is_space_continuous(output_space):
            self.mode = "continuous"
            self.layers = [layer_type(in_features, 1, key=key)]  # scalar Q(s, a)
        else:
            raise ValueError(f"Unsupported output space: {output_space}")

    def __call__(self, x, action_mask=None, *, key: PRNGKeyArray | None = None):
        out = _forward_per_dimension(self.layers, x, key=key)
        if self.mode == "continuous":
            if action_mask is not None:
                logger.debug("Action mask provided for continuous space, ignoring.")
            return out.squeeze()
        if action_mask is not None:
            out = _apply_action_mask(out, action_mask)
        return out

    @classmethod
    def with_params(
        cls, *, layer_type: Callable[..., Network] = eqx.nn.Linear
    ) -> Callable[..., Self]:
        return eqx.Partial(cls, layer_type=layer_type)


class _PyTreeOutputNetwork(eqx.Module):
    """Base Output network for a (single or PyTree of) output space(s)."""

    heads: PyTree[Network]

    num_output_spaces: int = eqx.field(static=True)


class PyTreeActionNetwork(_PyTreeOutputNetwork):
    """Output network producing an **action distribution** per output space.

    Builds one head per output-space leaf, selecting a discrete or continuous
    head automatically from the space and returning a
    `distrax.Distribution` (policy head).

    A single head may itself be multi-dimensional (homogeneous),
    in which case its sub-layers are stacked and vmapped in the forward call.

    During a forward call this maps over all output spaces. When every head
    returns a distribution and `assume_independent` is set, per-space
    distributions are wrapped in `Independent` and, for multiple spaces,
    combined into a `distrax.Joint`. Otherwise the raw PyTree of outputs is
    returned unchanged.
    """

    assume_independent: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        output_space: PyTree[SpaceLike],
        *,
        key: PRNGKeyArray,
        discrete_output_layer: Callable[..., Network] = CategoricalLayer,
        continuous_output_layer: Callable[..., Network] = NormalLayer,
        assume_independent: bool = True,
    ):
        def create_head(key: PRNGKeyArray, space: SpaceLike):
            if _is_space_discrete(space):
                return discrete_output_layer(in_features, space, key=key)
            elif _is_space_continuous(space):
                return continuous_output_layer(in_features, space, key=key)
            raise ValueError(f"Unsupported output space: {space}")

        self.num_output_spaces = len(jax.tree.leaves(output_space))

        keys = optax.tree.split_key_like(key, output_space)
        self.heads = jax.tree.map(lambda o, k: create_head(k, o), output_space, keys)
        self.assume_independent = assume_independent

    def __call__(self, x, action_mask=None, *, key: PRNGKeyArray | None = None):
        if action_mask is None:  # Dummy action mask if not provided
            action_mask = jax.tree.map(
                lambda _: None, self.heads, is_leaf=_is_callable_module
            )

        outputs = jax.tree.map(
            lambda head, mask: head(x, action_mask=mask, key=key),
            self.heads,
            action_mask,
            is_leaf=_is_callable_module,
        )

        # Policy heads return distributions; Q/value heads return raw arrays.
        dist_list = jax.tree.leaves(outputs, is_leaf=_is_distribution)
        if (
            self.assume_independent
            and dist_list
            and all(_is_distribution(o) for o in dist_list)
        ):
            outputs = jax.tree.map(make_independent, outputs, is_leaf=_is_distribution)
            if len(dist_list) > 1:
                return distrax.Joint(outputs)

        return outputs

    @classmethod
    def with_params(
        cls,
        *,
        discrete_output_layer: Callable[..., Network] = CategoricalLayer,
        continuous_output_layer: Callable[..., Network] = NormalLayer,
        assume_independent: bool = True,
    ) -> Callable[..., Self]:
        return eqx.Partial(
            cls,
            discrete_output_layer=discrete_output_layer,
            continuous_output_layer=continuous_output_layer,
            assume_independent=assume_independent,
        )


class PyTreeQValueNetwork(_PyTreeOutputNetwork):
    """Output network producing **raw Q-values** per output space (no distribution).

    Builds one head per output-space leaf, selecting a discrete or continuous
    head automatically from the space and returning raw values (Q-value head).

    A single head may itself be multi-dimensional (homogeneous),
    in which case its sub-layers are stacked and vmapped in the forward call.

    During a forward call this maps over all output spaces. The raw PyTree of outputs is
    returned unchanged.
    """

    def __init__(
        self,
        in_features: int,
        output_space: PyTree[SpaceLike],
        *,
        key: PRNGKeyArray,
        output_layer: Callable[..., Network] = QLayer,
    ):
        self.num_output_spaces = len(jax.tree.leaves(output_space))

        keys = optax.tree.split_key_like(key, output_space)
        self.heads = jax.tree.map(
            lambda o, k: output_layer(in_features, o, key=k), output_space, keys
        )

    def __call__(self, x, action_mask=None, *, key: PRNGKeyArray | None = None):
        if action_mask is None:  # Dummy action mask if not provided
            action_mask = jax.tree.map(
                lambda _: None, self.heads, is_leaf=_is_callable_module
            )

        outputs = jax.tree.map(
            lambda head, mask: head(x, action_mask=mask, key=key),
            self.heads,
            action_mask,
            is_leaf=_is_callable_module,
        )
        return outputs

    @classmethod
    def with_params(
        cls, *, output_layer: Callable[..., Network] = QLayer
    ) -> Callable[..., Self]:
        return eqx.Partial(cls, output_layer=output_layer)
