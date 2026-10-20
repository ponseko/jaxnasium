# Wrappers

Wrappers modify the behaviour of an environment and can be stacked freely.

```python
import jaxnasium as jym

env = jym.make("Breakout-MinAtar")
env = jym.FlattenObservationWrapper(env)
env = jym.LogWrapper(env)
```

## Base class

::: jaxnasium.Wrapper

## Environment wrappers

::: jaxnasium.VecEnvWrapper

::: jaxnasium.LogWrapper

::: jaxnasium.NormalizeVecObsWrapper

::: jaxnasium.NormalizeVecRewardWrapper

::: jaxnasium.TransformObservationWrapper

::: jaxnasium.FlattenObservationWrapper

::: jaxnasium.FrameStackWrapper

::: jaxnasium.TransformRewardWrapper

::: jaxnasium.ScaleRewardWrapper

::: jaxnasium.DiscreteActionWrapper

::: jaxnasium.FlattenActionSpaceWrapper

::: jaxnasium.StackActionSpaceWrapper

::: jaxnasium.MetaParamsWrapper

## Third-party library wrappers

These are applied automatically by [`jaxnasium.make`](Available-Environments.md) when
an environment comes from an external suite. You only need them directly if you
construct the third-party environment yourself.

::: jaxnasium.wrappers.BraxWrapper
    options:
        members: false

::: jaxnasium.wrappers.GymnaxWrapper
    options:
        members: false

::: jaxnasium.wrappers.JaxMARLWrapper
    options:
        members: false

::: jaxnasium.wrappers.JumanjiWrapper
    options:
        members: false

::: jaxnasium.wrappers.NavixWrapper
    options:
        members: false

::: jaxnasium.wrappers.PgxWrapper
    options:
        members: false

::: jaxnasium.wrappers.xMinigridWrapper
    options:
        members: false

::: jaxnasium.wrappers.OctaxWrapper
    options:
        members: false

::: jaxnasium.wrappers.PlaygroundWrapper
    options:
        members: false

## Utility functions

::: jaxnasium.is_wrapped

::: jaxnasium.unwrap_to

::: jaxnasium.remove_wrapper

::: jaxnasium.insert_wrapper
