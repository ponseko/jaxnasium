from importlib.metadata import version

__version__ = version("jaxnasium")
from jaxnasium import _registry, envs as envs, eval as eval, tree as tree

from ._compilation import (
    enable_compilation_cache as enable_compilation_cache,
    precompile as precompile,
)
from ._environment import (
    ORIGINAL_OBSERVATION_KEY as ORIGINAL_OBSERVATION_KEY,
    Environment as Environment,
    EnvState as EnvState,
    TimeStep as TimeStep,
)
from ._spaces import (
    Box as Box,
    Discrete as Discrete,
    MultiDiscrete as MultiDiscrete,
    Space as Space,
)
from ._types import AgentObservation as AgentObservation
from .wrappers import (
    BraxWrapper as BraxWrapper,
    DiscreteActionWrapper as DiscreteActionWrapper,
    FlattenActionSpaceWrapper as FlattenActionSpaceWrapper,
    FlattenObservationWrapper as FlattenObservationWrapper,
    GymnasiumWrapper as GymnasiumWrapper,
    GymnaxWrapper as GymnaxWrapper,
    JaxMARLWrapper as JaxMARLWrapper,
    JumanjiWrapper as JumanjiWrapper,
    LogWrapper as LogWrapper,
    MetaParamsWrapper as MetaParamsWrapper,
    NavixWrapper as NavixWrapper,
    NormalizeVecObsWrapper as NormalizeVecObsWrapper,
    NormalizeVecRewardWrapper as NormalizeVecRewardWrapper,
    OctaxWrapper as OctaxWrapper,
    PgxWrapper as PgxWrapper,
    ScaleRewardWrapper as ScaleRewardWrapper,
    StackActionSpaceWrapper as StackActionSpaceWrapper,
    TransformRewardWrapper as TransformRewardWrapper,
    VecEnvWrapper as VecEnvWrapper,
    Wrapper as Wrapper,
    insert_wrapper as insert_wrapper,
    is_wrapped as is_wrapped,
    remove_wrapper as remove_wrapper,
    unwrap_to as unwrap_to,
    xMinigridWrapper as xMinigridWrapper,
)

registry = _registry.registry
make = _registry.make
