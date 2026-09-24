from typing import NamedTuple, Self

import equinox as eqx
from jaxtyping import Array, Bool, Float, Num, PyTree


class AgentObservation(eqx.Module):
    """A container for the observation of a **single** agent, with optional action masking.

    Typically, this container is optional. However, Algorithms in
    `jaxnasium.algorithms` expect observations to be wrapped in this type when
    action masking is enabled.

    **Arguments:**

    - `observation`: The observation of the agent.
    - `action_mask`: The action mask of the agent. A boolean array of the same shape as the action space.
    - `critic_observation`: Optional observation of the critic for the agent. when not provided, the regular observation is used.
     This may be used for CTDE (like) approaches.
    """

    observation: Num[Array, "..."] | PyTree[Bool[Array, "..."]]
    action_mask: Bool[Array, "..."] | PyTree[Bool[Array, "..."]] | None = None
    critic_observation: Num[Array, "..."] | PyTree[Num[Array, "..."]] | None = None

    def replace(self, **updates) -> Self:
        keys, values = zip(*updates.items())
        return eqx.tree_at(
            lambda c: [c.__dict__[key] for key in keys],
            self,
            values,
            is_leaf=lambda x: x is None,
        )

    @property
    def critic_input(self) -> Num[Array, "..."] | PyTree[Num[Array, "..."]]:
        if self.critic_observation is not None:
            return self.critic_observation
        return self.observation


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
