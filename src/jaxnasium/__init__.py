from importlib.metadata import version

__version__ = version("jaxnasium")
from jaxnasium import _registry, envs as envs, tree as tree

from ._environment import Environment as Environment, TimeStep as TimeStep
from ._spaces import (
    Box as Box,
    Discrete as Discrete,
    MultiDiscrete as MultiDiscrete,
    Space as Space,
)
from ._types import AgentObservation as AgentObservation
from ._wrappers import (
    DiscreteActionWrapper as DiscreteActionWrapper,
    FlattenActionSpaceWrapper as FlattenActionSpaceWrapper,
    FlattenObservationWrapper as FlattenObservationWrapper,
    LogWrapper as LogWrapper,
    NormalizeVecObsWrapper as NormalizeVecObsWrapper,
    NormalizeVecRewardWrapper as NormalizeVecRewardWrapper,
    ScaleRewardWrapper as ScaleRewardWrapper,
    TransformRewardWrapper as TransformRewardWrapper,
    VecEnvWrapper as VecEnvWrapper,
    Wrapper as Wrapper,
    is_wrapped as is_wrapped,
    remove_wrapper as remove_wrapper,
)
from ._wrappers_ext import (
    BraxWrapper as BraxWrapper,
    GymnaxWrapper as GymnaxWrapper,
    JaxMARLWrapper as JaxMARLWrapper,
    JumanjiWrapper as JumanjiWrapper,
    NavixWrapper as NavixWrapper,
    PgxWrapper as PgxWrapper,
    xMinigridWrapper as xMinigridWrapper,
)

registry = _registry.registry
make = _registry.make
