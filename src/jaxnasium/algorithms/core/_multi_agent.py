import os
from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PyTreeDef

import jaxnasium as jym

from ._transition import Transition

AGENT_BATCH_SIZE_ENV_VAR = "JAXNASIUM_MULTI_AGENT_BATCH_SIZE"


def _agent_batch_size() -> int | None:
    """Homogeneous agents are typically vmapped (via `jax.lax.map`), which may be memory-intensive.
    This environment variable sets the batch_size of `lax.map` to limit peak memory usage.
    0 -> vmap all agents (default)
    any positive integer -> vmap in chunks of that size
    False -> run sequentially (no vmap)
    """
    value = os.environ.get(AGENT_BATCH_SIZE_ENV_VAR)
    if value is None:
        return 0
    if value.lower() == "false":
        return None
    return int(value) if value else 0


def _is_prng_key(arg):
    """Check if the argument is a JAX PRNGKeyArray."""
    try:
        jax.random.split(arg)
        return True
    except Exception:
        return False


def _result_tuple_to_tuple_result(r, outer_def=None):
    if outer_def is None:
        _, outer_def = eqx.tree_flatten_one_level(r)
    children = outer_def.flatten_up_to(r)
    if type(children[0]) is not tuple:
        return r  # f returned a single value -> r already has outer structure

    transposed = zip(*children)  # tuple-of-(list of children)
    return tuple(
        _result_tuple_to_tuple_result(
            jax.tree.unflatten(outer_def, list(group)), outer_def
        )
        for group in transposed
    )


def _is_pytree_of_agents(x):
    """Check if *x* is a pytree whose first-level leaves are all RLAgents."""
    from .._algorithm import RLAgent

    try:
        leaves, _ = eqx.tree_flatten_one_level(x)
        return all(isinstance(leaf, RLAgent) for leaf in leaves)
    except Exception:
        return False


def _is_pytree_of_transitions(x):
    """Check if *x* is a pytree whose first-level leaves are all Transitions."""
    try:
        leaves, _ = eqx.tree_flatten_one_level(x)
        return all(isinstance(leaf, Transition) for leaf in leaves)
    except Exception:
        return False


def _matches_agent_structure(x, ref_structure) -> bool:
    try:
        _, structure = eqx.tree_flatten_one_level(x)
    except Exception:
        return False
    return structure == ref_structure


def to_per_agent(x, ref_structure):
    if _matches_agent_structure(x, ref_structure):
        return x
    return jax.tree.unflatten(ref_structure, [x] * ref_structure.num_leaves)


def _agent_signature(subtree):
    """Checks if the arguments can be vmapped over"""
    try:
        leaves, treedef = jax.tree.flatten(subtree)
    except Exception:
        return None

    signature = []
    for leaf in leaves:
        if not eqx.is_array_like(leaf):
            return None
        signature.append((jnp.shape(leaf), jnp.result_type(leaf)))
    return treedef, tuple(signature)


def _is_homogeneous(per_agent_arguments, agent_structure) -> bool:
    """Whether every genuinely per-agent argument is identical in shape across agents."""
    if agent_structure.num_leaves < 2:
        return False
    for tree, is_per_agent in per_agent_arguments:
        if not is_per_agent:
            continue  # a shared value is closed over, never stacked
        signatures = [
            _agent_signature(sub) for sub in agent_structure.flatten_up_to(tree)
        ]
        if signatures[0] is None or any(s != signatures[0] for s in signatures[1:]):
            return False
    return True


def _vmap_over_agents(f, positional, keyword, agent_structure) -> Any:
    """Run `f` once over an agent axis instead of once per agent."""
    mapped: list[Any] = []
    slots: list[tuple[bool, Any]] = []  # (is_mapped, index-or-shared-value)

    def _make_arguments(tree, is_per_agent):
        if is_per_agent:
            slots.append((True, len(mapped)))
            mapped.append(jym.tree.stack(tree))
        else:  # All agents share the same argument
            slots.append((False, agent_structure.flatten_up_to(tree)[0]))

    for tree, is_per_agent in positional:
        _make_arguments(tree, is_per_agent)
    kw_keys = list(keyword)
    for key in kw_keys:
        _make_arguments(*keyword[key])

    num_positional = len(positional)

    def call(mapped_arguments):
        resolved = [
            mapped_arguments[payload] if is_mapped else payload
            for is_mapped, payload in slots
        ]
        args = resolved[:num_positional]
        kwargs = dict(zip(kw_keys, resolved[num_positional:]))
        return f(*args, **kwargs)

    stacked_result = jax.lax.map(call, tuple(mapped), batch_size=_agent_batch_size())

    return jax.tree.unflatten(
        agent_structure,
        [
            jax.tree.map(lambda x, _i=i: x[_i], stacked_result)
            for i in range(agent_structure.num_leaves)
        ],
    )


def map_multi_agent(
    f,
    tree,
    *rest,
    agent_structure: PyTreeDef | None = None,  # pyright: ignore[reportInvalidTypeForm]
    vmap: bool = True,
    **kwargs,
) -> Any:
    """Apply `f` per agent, over the first level of each per-agent argument.

    **Arguments:**

    - `f`: the single-agent function.
    - `tree`, `*rest`, `**kwargs`: arguments to `f`. Arguments whose first level matches the agent structure are split per agent.
        Other arguments are broadcast to every agent. Some elements are special-case:
        PRNG keys are split, `Transition`s are transposed, and `MultiAgentWrapper`s are unwrapped.
    - `agent_structure`: first-level structure to map over. Inferred from the first  non-key argument when omitted.
    - `vmap`: whether `f` may vmapped across agents. This is possible when agent are homogeneous. The batch_size of this
        map may be set with the environment variable `JAXNASIUM_MULTI_AGENT_BATCH_SIZE`. When unset, all agents are vmapped together.
        Set to False to disable and run sequentially instead.

    """

    arguments = [tree] + list(rest)

    def _unwrap(el):
        """Expose the agent axis of the containers that hide one behind their own.

        Needed before the agent structure is known, so it must not depend on it.
        """
        if isinstance(el, Transition):
            return el.view_transposed
        elif isinstance(el, MultiAgentWrapper):
            return el.agents
        return el

    # infer agent structure if not provided
    if agent_structure is None:
        first_arg = next((arg for arg in arguments if not _is_prng_key(arg)), None)
        _, agent_structure = eqx.tree_flatten_one_level(_unwrap(first_arg))

    def _process_item(el):
        """Per-agent view of `el`, plus whether it genuinely varies per agent."""
        if _is_prng_key(el):
            return jym.tree.split_key_like_structure(el, agent_structure), True
        el = _unwrap(el)
        if _matches_agent_structure(el, agent_structure):
            return el, True
        return to_per_agent(el, agent_structure), False

    processed_arguments = [_process_item(item) for item in arguments]
    processed_kwargs = {k: _process_item(v) for k, v in kwargs.items()}

    if vmap and _is_homogeneous(
        processed_arguments + list(processed_kwargs.values()), agent_structure
    ):
        result = _vmap_over_agents(
            f, processed_arguments, processed_kwargs, agent_structure
        )

    elif kwargs:
        processed_arguments = [a[0] for a in processed_arguments]
        processed_kwargs = {k: v[0] for k, v in processed_kwargs.items()}
        kw_keys = list(processed_kwargs)
        bundled_kwargs = jym.tree.map_one_level(
            lambda *vals: dict(zip(kw_keys, vals)),
            *[processed_kwargs[k] for k in kw_keys],
        )

        def _new_call(*args_w_kw_dict):
            *args, kw_dict = args_w_kw_dict
            return f(*args, **kw_dict)

        result = jym.tree.map_one_level(_new_call, *processed_arguments, bundled_kwargs)

    else:
        processed_arguments = [a[0] for a in processed_arguments]
        result = jym.tree.map_one_level(f, *processed_arguments)

    # Now the ouput can contain:
    # a PyTree of Agents originally a MultiAgentWrapper → re-wrap in MultiAgentWrapper
    # a PyTree of Transitions originally a Transition → merge back into a single Transition
    # a PyTree of tuples that we want to convert to a tuple of PyTrees (e.g. dist.sample_and_log_prob returns a tuple, which we want to preserve)

    out = _result_tuple_to_tuple_result(result)

    source_wrapper = next(
        (x for x in arguments if isinstance(x, MultiAgentWrapper)), None
    )

    def _process_output(o):
        if source_wrapper is not None and _is_pytree_of_agents(o):
            return eqx.tree_at(lambda w: w.agents, source_wrapper, o)

        if _is_pytree_of_transitions(o):
            return Transition.from_transposed(o)

        if type(o) is tuple:
            return tuple(_process_output(x) for x in o)

        return o

    return _process_output(out)


class MultiAgentWrapper(eqx.Module):
    agents: PyTree
    trainer: Any = None
    """The team's trainer: the `RLAlgorithm` the agents were built from, unsplit.
     `None` for a wrapper over plain modules rather than agents (e.g. `agent.critic` in multi-agent mode), which has no trainer.
    """

    @property
    def _structure(self) -> PyTreeDef:  # pyright: ignore[reportInvalidTypeForm]
        """First-level pytree structure of the wrapped agents."""
        return eqx.tree_flatten_one_level(self.agents)[1]

    def _matches_structure(self, arg) -> bool:
        """Check whether *arg*'s first-level structure matches the agent structure."""
        return _matches_agent_structure(arg, self._structure)

    def with_hyperparams(self, **hyperparams) -> "MultiAgentWrapper":
        """Override the hyperparameters of every agent's trainer *and* the wrapper trainer."""
        agents = map_multi_agent(
            lambda agent, **kw: agent.with_hyperparams(**kw),
            self.agents,
            **hyperparams,
            agent_structure=self._structure,
            vmap=False,  # don't vmap this
        )
        return type(self)(agents, trainer=replace(self.trainer, **hyperparams))

    def __call__(self, *args, **kwargs):
        return self.__getattr__("__call__")(*args, **kwargs)

    def __getattr__(self, name: str):
        agents = self.agents
        ref_structure = self._structure
        first_agent = eqx.tree_flatten_one_level(agents)[0][0]
        first_attr = getattr(first_agent, name)

        def _is_bound_method(x):
            return hasattr(x, "__self__") and hasattr(x, "__func__")

        if _is_bound_method(first_attr):
            # Methods on an agent are per-agent by default;
            # methods decorated with `@collective` are opted-out, and are called on the wrapper itself.
            if getattr(first_attr.__func__, "__per_agent__", True) is False:
                return first_attr.__func__.__get__(self)

            def multi_agent_dispatcher(*args, **kwargs):
                return map_multi_agent(
                    lambda a, *args, **kw: getattr(a, name)(*args, **kw),
                    self,
                    *args,
                    **kwargs,
                    agent_structure=ref_structure,
                )

            return multi_agent_dispatcher

        # Plain attribute access shouldn't vmap
        elif isinstance(first_attr, eqx.Module):
            return MultiAgentWrapper(
                map_multi_agent(
                    lambda a: getattr(a, name),
                    agents,
                    agent_structure=ref_structure,
                    vmap=False,
                )
            )

        return map_multi_agent(
            lambda a: getattr(a, name),
            agents,
            agent_structure=ref_structure,
            vmap=False,
        )
