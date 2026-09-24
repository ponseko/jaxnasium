import functools
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

_LOGWRAPPER_REQ_KEYS = ("returned_episode", "returned_episode_returns", "timestep")


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
    if callback_interval < 1:
        assert n is not None, "n must be provided if callback_interval is less than 1"
        callback_interval = int(n * callback_interval)

    if callback_fn == "tqdm":
        try:
            import tqdm.auto
        except ImportError:
            raise ImportError(
                "tqdm is not installed. Please install it with `pip install tqdm`."
            )

        progress_bar = []

        def update_tqdm_bar(_, iteration):
            if iteration == 0:
                progress_bar.append(
                    tqdm.auto.tqdm(
                        total=n, desc="Training Progress", unit=" iterations "
                    )
                )

            progress_bar[0].update(callback_interval)

    def simple_reward_logger(data, iteration):
        missing = [key for key in _LOGWRAPPER_REQ_KEYS if key not in data]
        if missing:
            raise ValueError(
                f"Missing keys {missing} in the training metrics. "
                "Is the environment wrapped with LogWrapper?"
            )

        returned_episode = np.asarray(data["returned_episode"])
        returned_episode_returns = np.asarray(data["returned_episode_returns"])
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

    def maybe_log(iteration: int, data):
        if callback_fn is not None and callback_interval > 0:
            if callback_fn == "tqdm":
                log_fn = update_tqdm_bar
            elif callback_fn == "simple":
                log_fn = simple_reward_logger
            else:
                log_fn = callback_fn

            _ = jax.lax.cond(
                iteration % callback_interval == 0,
                lambda: jax.debug.callback(
                    lambda d, i: log_fn(d, i) if callable(log_fn) else None,
                    data,
                    iteration,
                ),
                lambda: None,
            )

    def _scan_callback(func):
        @functools.wraps(func)
        def wrapper(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x

            carry, this_iter_metrics = func(carry, x)
            maybe_log(iter_num, this_iter_metrics)

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
