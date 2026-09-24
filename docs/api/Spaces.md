# Spaces

Spaces describe the shape and range of observations and actions. Jaxnsium environments may
return a single `Space`, or a PyTree of spaces (Additionally, see [Multi-Agent](../algorithms/Multi-Agent.md)).

::: jaxnasium.Space

::: jaxnasium.Box
    options:
        members:
            - sample

::: jaxnasium.Discrete
    options:
        members:
            - sample

::: jaxnasium.MultiDiscrete
    options:
        members:
            - sample
