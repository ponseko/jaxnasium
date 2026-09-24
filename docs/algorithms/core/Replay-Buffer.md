# Replay Buffer

A circular buffer of [`Transition`][jaxnasium.algorithms.core.Transition]s, used by the off-policy algorithms (DQN, SAC).

::: jaxnasium.algorithms.core.TransitionBuffer
    options:
        members:
            - insert
            - sample

## Prioritized replay

::: jaxnasium.algorithms.core.PrioritizedTransitionBuffer
    options:
        inherited_members: false
        members:
            - insert
            - sample
            - update_priorities
            - update_beta
