from typing import NamedTuple, Optional

from jaxtyping import Array, Bool, Float, Num, PyTree


class AgentObservation(NamedTuple):
    """A container for the observation of a **single** agent, with optional action masking.

    Typically, this container is optional. However, Algorithms in
    `jaxnasium.algorithms` expect observations to be wrapped in this type when
    action masking is enabled.

    **Arguments:**

    - `observation`: The observation of the agent.
    - `action_mask`: The action mask of the agent. A boolean array of the same shape as the action space.
    """

    observation: Num[Array, "..."] | PyTree[Bool[Array, "..."]]
    action_mask: Optional[Bool[Array, "..."] | PyTree[Bool[Array, "..."]]] = None


class TimeStep(NamedTuple):
    """A container for the output of an environment's step function.
    (`timestep, state = env.step(...)`).

    This class follows the [Gymnasium](https://gymnasium.farama.org/) standard API,
    with the signature: `(obs, reward, terminated, truncated, info)` tuple.

    **Arguments:**

    - `observation`: The environment state representation provided to the agent.
      Can be an Array or a PyTree of arrays.
      When using action masking, the observation should be of type `AgentObservation`.
    - `reward`: The reward signal from the previous action, indicating performance.
      Can be a scalar Array or a PyTree of reward Arrays (in the case of multi agent-environments).
    - `terminated`: Boolean flag indicating whether the episode has ended due to reaching a terminal state (e.g., goal reached, game over).
    - `truncated`: Boolean flag indicating whether the episode ended due to external factors (e.g., reaching max steps, timeout).
    - `info`: Dictionary containing any additional information about the environment step.
    """

    observation: (
        Num[Array, "..."] | PyTree[Num[Array, "..."]] | PyTree[AgentObservation]
    )
    reward: Float[Array, "..."] | PyTree[Float[Array, "..."]]
    terminated: Bool[Array, "..."] | PyTree[Bool[Array, "..."]]
    truncated: Bool[Array, "..."] | PyTree[Bool[Array, "..."]]
    info: dict
