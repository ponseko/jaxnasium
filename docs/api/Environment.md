# Environment

::: jaxnasium.Environment
    options:
        members:
            - step
            - step_env
            - reset
            - reset_env
            - observation_space
            - sample_observation
            - action_space
            - sample_action
            - auto_reset
            - multi_agent
            - agent_structure
            - state_type

!!! note
    When subclassing this class with a type parameter (`Environment[MyState]`), the state
    type is available at runtime via `state_type`; this may be helpfull for environments that
    should have an extensible state.
    The state type is read from class definitions only: instantiating a subscripted class, e.g. `MyEnv[ExtendedState]()`, silently creates a `MyEnv` with its default state type.
    Declare a subclass instead.

## Timestep

::: jaxnasium.TimeStep
    options:
        members: false

## Environment state

::: jaxnasium.EnvState
    options:
        members: false

## Observation container (optional)

::: jaxnasium.AgentObservation
    options:
        members: false
