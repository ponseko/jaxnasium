import functools
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

_LOGWRAPPER_REQ_KEYS = ("returned_episode", "returned_episode_returns", "timestep")


class Callback(NamedTuple):
    """Simple (optional) callback container that calls
    `fn(key, agent, metrics, iteration, num_iterations)` every `interval`.
    `interval` is a fixed number of iterations, or a fraction of
    `num_iterations` when below 1."""

    fn: Callable
    interval: int | float = 0.05

    def __call__(self, key, agent, metrics, iteration, num_iterations) -> None:
        interval = self.interval
        if interval < 1:
            assert num_iterations is not None, (
                "a fractional interval needs num_iterations"
            )
            interval = max(1, int(num_iterations * interval))
        call = iteration % interval == 0
        if num_iterations is not None:
            call = call | (iteration == num_iterations)
        jax.lax.cond(
            call,
            lambda: self.fn(key, agent, metrics, iteration, num_iterations),
            lambda: None,
        )


def mean_episode_returns(metrics: dict[str, Any]) -> PyTree[Float[Array, ""]]:
    """mean episode return across all vectorized environments in a single batch of metrics
     produces a single scalar (per agent) per scan iteration.
    Requires LogWrapper on the environment.
    """
    missing = [key for key in _LOGWRAPPER_REQ_KEYS if key not in metrics]
    if missing:
        raise ValueError(
            f"Missing keys {missing} in the training metrics. "
            "Is the environment wrapped with LogWrapper?"
        )

    finished = metrics["returned_episode"]
    num_finished = jnp.sum(finished)

    def _mean(returns):
        total = jnp.sum(jnp.where(finished, returns, 0.0))
        return jnp.where(num_finished > 0, total / num_finished, jnp.nan)

    # map due to possible Multi-Agent reward structure
    return jax.tree.map(_mean, metrics["returned_episode_returns"])


def print_returns(key, agent, metrics, iteration, num_iterations) -> None:
    """Callback printing the return of every episode finished in this iteration.
    Requires a `LogWrapper` on the environment."""
    missing = [key for key in _LOGWRAPPER_REQ_KEYS if key not in metrics]
    if missing:
        raise ValueError(
            f"Missing keys {missing} in the training metrics. "
            "Is the environment wrapped with LogWrapper?"
        )

    def print_on_host(data):
        returned_episode = np.asarray(data["returned_episode"])
        # tree.map for multi-agent cases
        returned_episode_returns = jax.tree.map(
            np.asarray, data["returned_episode_returns"]
        )
        timestep = np.asarray(data["timestep"])
        if timestep.ndim < returned_episode.ndim:
            timestep = np.broadcast_to(timestep[..., None], returned_episode.shape)

        num_envs = timestep.shape[-1]
        return_values = jax.tree.map(
            lambda x: x[returned_episode], returned_episode_returns
        )
        timesteps = timestep[returned_episode] * num_envs
        for t in range(len(timesteps)):
            return_values_t = jax.tree.map(
                lambda x, _t=t: x[_t].item() if hasattr(x[_t], "item") else x[_t],
                return_values,
            )
            return_values_t = jax.tree.map(lambda x: round(x, 3), return_values_t)
            print(f"global step={timesteps[t]}, episodic return={return_values_t}")

    jax.debug.callback(print_on_host, {k: metrics[k] for k in _LOGWRAPPER_REQ_KEYS})


def progress_bar(key, agent, metrics, iteration, num_iterations) -> None:
    """Callback showing the training progress in a `tqdm` progress bar."""
    try:
        import tqdm.auto
    except ImportError:
        raise ImportError("tqdm is not installed. Install it with `pip install tqdm`.")
    bar = []  # the bar of the training run in progress

    def update_on_host(iteration):
        iteration = int(iteration)
        if not bar or iteration < bar[0].n:  # a new run
            bar[:] = [
                tqdm.auto.tqdm(
                    total=num_iterations, desc="Training", unit=" iterations"
                )
            ]
        bar[0].update(iteration - bar[0].n)
        if iteration == num_iterations:
            bar[0].refresh()  # tqdm skips draws that come too fast

    jax.debug.callback(update_on_host, iteration)


def scan_callback(
    func: Callable | None = None,
    callback_fn: Callable | Literal["tqdm", "simple"] | None = None,
    callback_interval: float = 20,
    n: int | None = None,
    reduce_ys_fn: Callable | Literal["mean"] | None = None,
) -> Callable:
    """Wrap a scan body so its per-iteration metrics can be logged and shrunk.

    **Arguments**:
        `func`: function to wrap
        `callback_fn`: function to call on the ys of the scan
        `callback_interval`: how often to call the callback in iterations. Can be a fraction of `n`.
        `n`: total number of iterations, required if `callback_interval` 0.0 < 1.0.
        `reduce_ys_fn`: optional function to reduce the metrics *after* the callback is called,
            but before they are returned to the scan. May be used to aggregate metrics to reduce
            memory consumption. Set this to `mean` (default) to average `LogWrapper` returned
            returned returns across the current iteration of data.
    """
    assert callable(func) or func is None

    assert callback_interval > 0, "callback_interval must be greater than 0"

    callbacks = _legacy_callback(callback_fn, callback_interval)

    def _scan_callback(func):
        @functools.wraps(func)
        def wrapper(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x

            carry, this_iter_metrics = func(carry, x)

            for callback in callbacks:
                callback(None, None, this_iter_metrics, iter_num + 1, n)

            if reduce_ys_fn is not None:
                reduce_fn = (
                    mean_episode_returns if reduce_ys_fn == "mean" else reduce_ys_fn
                )
                this_iter_metrics = reduce_fn(this_iter_metrics)

            return carry, this_iter_metrics

        return wrapper

    return _scan_callback(func) if callable(func) else _scan_callback


def pretty_print_network(network: eqx.Module):
    def _count_parameters(module: eqx.Module):
        """Count the total number of trainable parameters in a module."""
        total_params = 0
        for leaf in jax.tree.leaves(module):
            if hasattr(leaf, "shape"):
                total_params += leaf.size
        return total_params

    def _print_recursive(network: eqx.Module, indent: str = "", is_last: bool = True):
        module_name = network.__class__.__name__
        param_count = _count_parameters(network)

        indent_display = ""
        if indent:
            tree_char = "└── " if is_last else "├── "
            indent_display = indent[:-4] + tree_char

        full_path = indent_display + module_name
        print(f"{full_path:<50} {param_count:<15,}")

        # Get all child modules
        child_modules = jax.tree.leaves(
            network, is_leaf=lambda x: x is not network and isinstance(x, eqx.Module)
        )
        child_modules = [
            child for child in child_modules if isinstance(child, eqx.Module)
        ]

        # Recursively process child modules
        for i, child in enumerate(child_modules):
            new_indent = indent + ("    " if is_last else "│   ")
            is_last_child = i == len(child_modules) - 1
            _print_recursive(child, new_indent, is_last_child)

    print("=" * 100)
    print(f"{'Module name':<50} {'Param count':<15}")
    _print_recursive(network)
    print("\n")
    print(f"Total parameters: {_count_parameters(network):,}")
    print("=" * 100)


def _legacy_callback(
    callback_fn: Callable | Literal["tqdm", "simple", "unset"] | None,
    callback_interval: float | Literal["unset"],
) -> tuple[Callback, ...]:
    """A `log_function` and `log_interval` as a `Callback` (none if not set)."""
    if callback_fn is None or callback_fn == "unset":
        return ()
    if callback_fn == "simple":
        fn = print_returns
    elif callback_fn == "tqdm":
        fn = progress_bar
    else:  # a host function of the metrics and iteration

        def fn(key, agent, metrics, iteration, num_iterations):
            jax.debug.callback(callback_fn, metrics, iteration)

    return (Callback(fn, callback_interval),)  # type: ignore
