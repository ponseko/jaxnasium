import logging
from typing import Any

import equinox as eqx
import jax
from jaxtyping import PyTree

from jaxnasium._environment import Observation
from jaxnasium._spaces import Box, Discrete, MultiDiscrete, Space
from jaxnasium._types import AgentObservation

logger = logging.getLogger(__name__)


def partition_obs_and_masks(
    observation_tree: PyTree[Observation], multi_agent: bool
) -> tuple[PyTree, PyTree]:
    """
    Seperates a PyTree of observations of type `AgentObservation` into two trees:
    one with the observations and one with the masks.
    If the observation is not of type `AgentObservation`, the second tree will only
    contain `None` values.
    This is used such that wrappers can act on the observations only when action masks
    are present.

    Useage:
    ```python
    (observation, ...), env_state = self._env.step/reset(...)
    obs, masks = self.partition_obs_and_masks(observation)
    obs = ... # do something with obs ...
    observation = eqx.combine(obs, masks)
    ```

    **Arguments:**

    - `observation_tree`: (PyTree of) observations to be partitioned.
    - `multi_agent`: Whether the environment is multi-agent or not.

    """
    observations = [observation_tree]
    if multi_agent:
        observations, _ = eqx.tree_flatten_one_level(observation_tree)
    if all(not isinstance(o, AgentObservation) for o in observations):
        filter_spec = True
    elif all(isinstance(o, AgentObservation) for o in observations):

        def _filter_spec(o: AgentObservation) -> AgentObservation:
            return AgentObservation(
                observation=True,
                action_mask=False,
                critic_observation=True if o.critic_observation is not None else None,
            )

        specs = [_filter_spec(o) for o in observations]
        if multi_agent:
            _, treedef = eqx.tree_flatten_one_level(observation_tree)
            filter_spec = jax.tree.unflatten(treedef, specs)
        else:
            filter_spec = specs[0]
    else:
        raise ValueError(
            "Observations for all agents must be either AgentObservation or not."
        )
    return eqx.partition(observation_tree, filter_spec=filter_spec)


def gymnasium_to_jaxnasium_space(space: Any) -> Space | PyTree[Space]:
    """Also works for Gymnax spaces"""

    def convert_single_space(space: Any) -> Space:
        space_class_name = space.__class__.__name__
        if space_class_name == "Discrete":
            return Discrete(space.n)
        elif space_class_name == "Box":
            return Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=space.dtype,
            )
        elif space_class_name == "MultiDiscrete":
            return MultiDiscrete(
                nvec=space.nvec,
                dtype=space.dtype,
            )
        else:
            raise NotImplementedError(
                f"Conversion for space type {space_class_name} is not implemented."
            )

    # Convert pytrees of spaces
    return jax.tree.map(convert_single_space, space)
