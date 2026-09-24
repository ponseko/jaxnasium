import logging
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree, PyTreeDef

from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY

if TYPE_CHECKING:
    from ._normalization import Normalizer


logger = logging.getLogger(__name__)


class Transition(eqx.Module):
    """
    Container for (possibly batches of) transitions
    Comes with functionality of creating minibatches and some utilities for multi-agent reinforcement learning.
    """

    observation: Array
    action: Array
    reward: Float[Array, " "]
    terminated: Bool[Array, " "]
    truncated: Bool[Array, " "]
    log_prob: Float[Array, "..."] | None = None
    info: dict | None = None
    value: Float[Array, " "] | None = None
    next_value: Float[Array, " "] | None = None
    next_observation: Array | None = None
    return_: Float[Array, " "] | None = None
    advantage: Float[Array, "..."] | None = None
    target: Float[Array, " "] | None = None
    bootstrap_n: Int[Array, " "] | None = None
    extra: Any = None

    # PER stores the values here so they are easily tracked and updated after reshuffling.
    PER_weight: Float[Array, " "] | None = None
    PER_index: Int[Array, " "] | None = None
    PER_priority: Float[Array, " "] | None = None

    def __post_init__(self):
        # Remove next obs from info to save memory. If next_observation is required;
        # it should be set explicitly on `next_observation`.
        if isinstance(self.info, dict) and ORIGINAL_OBSERVATION_KEY in self.info:
            info = {**self.info}
            info.pop(ORIGINAL_OBSERVATION_KEY, None)
            object.__setattr__(self, "info", info)

    def replace(self, **updates):
        keys, values = zip(*updates.items())
        return eqx.tree_at(
            lambda c: [c.__dict__[key] for key in keys],
            self,
            values,
            is_leaf=lambda x: x is None,
        )

    @property
    def structure(self) -> PyTreeDef:  # pyright: ignore[reportInvalidTypeForm]
        """
        Returns the top-level structure of the transition objects (using reward as a reference).
        This is either PyTreeDef(*) for single agents
        or PyTreeDef((*, x num_agents)) for multi-agent environments.
        usefull for unflattening Transition.flat.properties back to the original structure.
        """
        return jax.tree.structure(self.reward)

    def normalize(self, normalizer: "Normalizer"):
        """Normalizes the observation and rewards in the transition based on the given normalizer."""
        if self.next_observation is not None:
            return self.replace(
                observation=normalizer.normalize_obs(self.observation),
                next_observation=normalizer.normalize_obs(self.next_observation),
                reward=normalizer.normalize_reward(self.reward),
            )
        return self.replace(
            observation=normalizer.normalize_obs(self.observation),
            reward=normalizer.normalize_reward(self.reward),
        )

    @property
    def view_transposed(self) -> PyTree["Transition"]:
        """
        For single-agent settings, this will do nothing and return the original transition.

        For multi-agent settings:
        The original transition is a Transition of PyTrees
            e.g. Transition(observation={a1: ..., a2: ...}, action={a1: ..., a2: ...}, ...)
        The transposed transition is a PyTree of Transitions
            e.g. {a1: Transition(observation=..., action=..., ...), a2: Transition(observation=..., action=..., ...), ...}
        This is useful for multi-agent environments where we want to have a single Transition object per agent.
        """
        if self.structure == jax.tree.structure(0):  # single agent
            return self

        field_names = list(self.__dataclass_fields__.keys())

        fields = {}
        per_agent_keys = []
        for f in field_names:
            attr = getattr(self, f)
            attr_structure = jax.tree.structure(
                attr, is_leaf=lambda x, _attr=attr: x is not _attr
            )
            if attr_structure == self.structure:  # Compare with reference structure
                fields[f] = jax.tree.leaves(
                    attr, is_leaf=lambda x, _attr=attr: x is not _attr
                )
                per_agent_keys.append(f)
                continue
            fields[f] = attr

        per_agent_transitions = [
            Transition(
                **{
                    field_name: fields[field_name][i]
                    for field_name in field_names
                    if field_name in per_agent_keys
                },
                **{
                    field_name: fields[field_name]
                    for field_name in field_names
                    if field_name not in per_agent_keys
                },
            )
            for i in range(len(fields[field_names[0]]))
        ]

        return jax.tree.unflatten(self.structure, per_agent_transitions)

    @classmethod
    def from_transposed(cls, transposed: PyTree["Transition"]) -> "Transition":
        if isinstance(transposed, Transition):  # no effect in single agent
            return transposed

        per_agent_leaves, original_structure = jax.tree.flatten(
            transposed, is_leaf=lambda x: isinstance(x, Transition)
        )

        def _merge(*xs):
            # if all ids are the same -- the item was copied to each agent (includes None's)
            if all(id(x) == id(xs[0]) for x in xs):
                return xs[0]
            try:  # Else unflatten the leaves back to the original structure
                return jax.tree.unflatten(original_structure, xs)
            except Exception:  # fallback
                return xs[0]

        # tree_map over all agent-Transitions will rebuild a single Transition
        return jax.tree.map(
            _merge,
            *per_agent_leaves,
            is_leaf=lambda x: (
                x is not per_agent_leaves and not isinstance(x, Transition)
            ),
        )

    def scan(self, fn, init, *, reverse=False, unroll=1, **scan_kwargs) -> Any:
        """Scan an arbitrary function over the time axis of this transition batch.

        The user writes ``fn`` as if operating on **single-agent** (plain array)
        data.  In multi-agent settings the transition is automatically
        transposed into per-agent Transitions, scanned individually, and the
        results are merged back.

        **Arguments:**

        - ``fn``: ``(carry, step: Transition) -> (carry, output)`` – written
          for a single agent.
        - ``init``: Initial carry value.  If the transition is multi-agent and
          ``init``'s pytree structure already matches the agent structure, each
          agent gets its own init leaf. If any of ``init``'s
           first-level children match the agent structure, these are forwarded to
            the respective agent. Otherwise ``init`` is broadcast
          (replicated) to every agent.
        - ``reverse``, ``unroll``, ``**scan_kwargs``: forwarded to ``jax.lax.scan``.

        **Returns:**
            ``(final_carry, outputs)`` where in multi-agent mode every output
            has been merged back into the original per-agent pytree structure.
        """

        structure = self.structure

        # ---- single-agent: plain scan ----------------------------------------
        if structure.num_leaves <= 1:
            return jax.lax.scan(
                fn, init, self, reverse=reverse, unroll=unroll, **scan_kwargs
            )

        scan_fn = lambda i, x: jax.lax.scan(
            fn, i, x, reverse=reverse, unroll=unroll, **scan_kwargs
        )

        from ._multi_agent import map_multi_agent

        out = map_multi_agent(scan_fn, init, self, agent_structure=structure)

        return out

    def make_minibatches(
        self,
        key: PRNGKeyArray,
        n_minibatches: int,
        n_epochs: int = 1,
        n_batch_axis: int = 1,
    ) -> "Transition":
        """
        Creates shuffled minibatches from the transition.
        Returns a copy of the transition with each leaf reshaped to (num_minibatches, ...),

        This function first flattens the transition over the leading n_batch_axis.
        This is useful if your data hasn't been flattened yet and may be structured as
        (rollout_length, num_envs, ...), where num_envs is the number of parallel environments.

        If n_epochs > 1, it will create n_epochs copies of the minibatches. and stack these
        such that there is a single leading axis to scan over for training.

        If the batch size is not divisible by the number of minibatches, the remainder is
        truncated so that each minibatch has the same size.

        **Arguments:**
        - `key`: JAX PRNG key for randomization.
        - `num_minibatches`: Number of minibatches to create.
        - `n_epochs`: Number of copies the minibatches should be stacked.
        - `n_batch_axis`: Number of leading batch axes to flatten over. Default is 1 (already flattened).
        """

        def create_minibatch(rng, _):
            rng, key = jax.random.split(rng)

            # Random permutation of the batch indices
            batch_idx = jax.random.permutation(key, batch_size)

            # Drop remainder so each minibatch has equal size
            batch_idx = batch_idx[:effective_batch_size]

            # take from the batch in a new order (the order of the randomized batch_idx)
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, batch_idx, axis=0), batch
            )

            # split in minibatches
            minibatches = jax.tree.map(
                lambda x: x.reshape((n_minibatches, minibatch_size) + x.shape[1:]),
                shuffled_batch,
            )
            return rng, minibatches

        # reshape (flatten over all batch axes)
        batch = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[n_batch_axis:]), self)
        batch_size = jax.tree.leaves(batch)[0].shape[0]
        minibatch_size = batch_size // n_minibatches
        effective_batch_size = minibatch_size * n_minibatches
        if batch_size != effective_batch_size:
            logger.warning(
                f"Batch size {batch_size} is not divisible by number of minibatches {n_minibatches}. "
                f"Dropping {batch_size - effective_batch_size} samples to make equal sized minibatches."
            )

        # Create n_epochs of minibatches
        _, minibatches = jax.lax.scan(create_minibatch, key, None, n_epochs)

        # (n_epochs, n_minibatches, ...) --> (n_epochs * n_minibatches, ...)
        minibatches = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]), minibatches
        )

        return minibatches


def n_step_to_cumulative_single_step(
    transition: Transition, n_step: int, gamma: float
) -> Transition:
    """Converts an n-step transition into a single step transition with the cumulative reward.
    terminated, truncated, and next_observation are set to the boundary index of
    the first terminated or truncated step.

    n_step_axis is assumed to be the leading axis of the transition.

    Operates on single transitions, use `jax.vmap` to apply to a batch of transitions.
    """
    if n_step <= 1:
        return transition

    proxy = jax.tree.leaves(transition)[0]
    assert proxy.shape[0] == n_step, (
        f"Leading axis length {proxy.shape[0]} does not match n_step {n_step}"
    )

    def _collapse(t: Transition) -> Transition:
        """Collapse a (single-agent) n-step transition into one step."""
        done = jnp.logical_or(t.terminated, t.truncated)

        # 1's up to and including the first done step, 0's afterwards
        trace_still_active = jnp.cumprod(
            jnp.concatenate([jnp.ones(1), jnp.logical_not(done)[:-1]])
        )
        discounts = gamma ** jnp.arange(n_step)
        cum_reward = jnp.sum(t.reward * discounts * trace_still_active)

        boundary_idx = jnp.where(jnp.any(done), jnp.argmax(done), n_step - 1)

        return t.replace(
            observation=jax.tree.map(lambda x: x[0], t.observation),
            action=jax.tree.map(lambda x: x[0], t.action),
            reward=cum_reward,
            terminated=t.terminated[boundary_idx],
            truncated=t.truncated[boundary_idx],
            next_observation=jax.tree.map(
                lambda x: x[boundary_idx], t.next_observation
            ),
            bootstrap_n=boundary_idx,
        )

    # In the multi-agent case, we tranpose to per-agent and process each agent, then merge back.
    # No-op in the single-agent case.
    per_agent = transition.view_transposed
    collapsed = jax.tree.map(
        _collapse, per_agent, is_leaf=lambda x: isinstance(x, Transition)
    )
    return Transition.from_transposed(collapsed)
