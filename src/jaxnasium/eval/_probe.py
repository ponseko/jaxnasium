import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


# NOTE: unused internally at the moment
def _is_traceable(fn: Callable[..., Any], kwargs: dict[str, Any], name: str) -> bool:
    """Whether `fn` can be traced with `kwargs[name]` replaced by a JAX tracer.

    Used essentially to check if we can vmap over the parameter.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eqx.filter_make_jaxpr(lambda traced: fn(**{**kwargs, name: traced}))(
                jnp.asarray(kwargs[name])
            )
        return True
    except (ValueError, TypeError) as e:
        logger.debug(f"Param {name!r} is static: {type(e).__name__}: {e}")
        return False
    finally:
        jax.clear_caches()
        eqx.clear_caches()


def _get_last(spec: list | tuple) -> Any:
    """Grabs the last value. Tuples may be given as
    (first, last, [space]), hence we check for that."""
    if isinstance(spec, tuple) and len(spec) == 3:
        return spec[1]
    return spec[-1]


def split_static_dynamic_params(
    fn: Callable[..., Any],
    params: dict[str, list | tuple],
) -> tuple[dict[str, list | tuple], dict[str, list | tuple]]:
    """Split sweep params into static and dynamic groups by tracing `fn`.

    *Static* params change the computation graph when varied, so each value needs its own job.
    *Dynamic* params only feed data into a fixed graph and can be batched with `vmap`.

    Note that some params might be dynamic under some other parameters (perhaps because they
    have no effect), but should be static under other parameters. This function attempts to
    identify these cases and return parameters as static whenever *any* combination would
    mark the paramater as static. However, as probing every possible combination could be
    very expensive in large spaces, we only probe a subset.

    parameters marked as static are never traced.

    **Arguments**:
        `fn`: Function to probe. Traced, never executed for real.
        `params`: param names to a sequence of values / range specs.

    **Returns**:
        The static params and the dynamic params, as two dicts.
    """
    declared_static: set[str] = set()
    declare: Callable[..., set[str]] | None = getattr(fn, "static_params", None)
    if callable(declare):
        declared_static = set(declare(params))
        logger.debug(f"Declared static, not probed: {sorted(declared_static)}")

    static_params: dict[str, list | tuple] = {}
    dynamic_candidates: dict[str, list | tuple] = {}
    for name, spec in params.items():
        if name in declared_static:
            static_params[name] = spec
            continue
        try:
            jnp.asarray(spec[0] if isinstance(spec, tuple) else spec)
        except TypeError:
            static_params[name] = spec
        else:
            dynamic_candidates[name] = spec

    if not dynamic_candidates:
        return static_params, {}

    # Probe fn with each parameter at its first value and each parameter as its last value.
    first_value_baseline = {name: spec[0] for name, spec in params.items()}
    last_value_baseline = {name: _get_last(spec) for name, spec in params.items()}
    contexts = [
        first_value_baseline,
        last_value_baseline,
    ]
    # NOTE: disabled probing more params
    # for name, spec in static_params.items():
    #     for i, value in enumerate(list(spec)):
    #         if i == 0 or i == len(spec) - 1:
    #             continue  # already added
    # contexts.append({**first_value_baseline, name: value})
    # contexts.append({**last_value_baseline, name: value})

    dynamic_params: dict[str, list | tuple] = {}
    for name, spec in dynamic_candidates.items():
        if not all(_is_traceable(fn, ctx, name) for ctx in contexts):
            static_params[name] = spec
        else:
            dynamic_params[name] = spec

    return static_params, dynamic_params


def _format_bytes(n: float | None) -> str:
    """Human-readable byte count."""
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def _format_flops(n: float | None) -> str:
    """Human-readable FLOP count."""
    if n is None:
        return "n/a"
    n = float(n)
    for unit, scale in (
        ("PFLOPs", 1e15),
        ("TFLOPs", 1e12),
        ("GFLOPs", 1e9),
        ("MFLOPs", 1e6),
        ("KFLOPs", 1e3),
        ("FLOPs", 1.0),
    ):
        if abs(n) >= scale or unit == "FLOPs":
            return f"{n / scale:.2f} {unit}"
    return f"{n:.2e} FLOPs"


@dataclass
class CostEstimate:
    """Container class for the result of log_cost_estimate.
    Ballpark XLA cost / memory estimate for a compiled call.
    Comes with a __repr__ for pretty printing.
    """

    fn_name: str
    device_memory_bytes: float | None = None
    temp_bytes: float | None = None
    argument_bytes: float | None = None
    output_bytes: float | None = None
    flops: float | None = None
    bytes_accessed: float | None = None
    num_gpus: int = 0
    device_byte_sizes: list[int | None] | None = None
    error: str | None = None

    def __repr__(self) -> str:
        limits = self.device_byte_sizes or [None]
        lines = [
            "=================================================",
            f"CostEstimate({self.fn_name!r}:",
        ]
        if self.error is not None:
            lines.append(f"  failed — {self.error}")
        else:
            lines.extend(
                [
                    f"  device memory required ≈ {_format_bytes(self.device_memory_bytes)}",
                    f"  FLOPs ≈ {_format_flops(self.flops)}",
                    f"  bytes accessed ≈ {_format_bytes(self.bytes_accessed)}",
                ]
            )
        lines.extend(
            [
                f"  GPUs detected = {self.num_gpus}",
                f"  device bytes_limits = {'[' + ', '.join(_format_bytes(n) for n in limits) + ']'}",
                " ================================================",
                "  Note: These values are a proxy only and provide limited guarantees on the real usage.",
                "    (compiler / backend / version / positioning of the stars may alter the real usage.)",
                " ================================================",
            ]
        )
        return "\n".join(lines)


def log_cost_estimate(fn: Callable[..., Any], **kwargs: Any) -> CostEstimate:
    """Compile ``fn(**kwargs)`` and return a ballpark memory / FLOP estimate.

    Typical usage: ``print(log_cost_estimate(fn, **kwargs))``.

    This is a compiler proxy only — it cannot guarantee real peak usage, which
    can vary with backend, JAX version, rematerialization, and inputs that
    change the computation graph.

    https://docs.jax.dev/en/latest/aot.html#debug-information-and-analyses-when-available
    """
    fn_name = getattr(fn, "__name__", repr(fn))

    try:
        n_gpus = jax.device_count("gpu")
    except RuntimeError:
        n_gpus = 0

    # Memory sizes of each (gpu/tpu) device; does not include cpu memory.
    device_byte_sizes: list[int | None] = []
    for device in jax.devices():
        try:
            stats = device.memory_stats()
            device_byte_sizes.append(int(stats["bytes_limit"]))
        except Exception:
            logger.debug(f"Could not get memory stats for device {device}")
            device_byte_sizes.append(None)

    # Compile and get mem + flop stats
    try:
        compiled = jax.jit(lambda: fn(**kwargs)).lower().compile()
        cost = compiled.cost_analysis() or {}
        memory = compiled.memory_analysis()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.warning(f"Could not estimate cost for {fn_name}: {msg}")
        return CostEstimate(
            fn_name=fn_name,
            num_gpus=n_gpus,
            device_byte_sizes=device_byte_sizes,
            error=msg,
        )

    temp = argument = output = device_memory = None
    if memory is not None:
        temp = float(memory.temp_size_in_bytes)
        argument = float(memory.argument_size_in_bytes)
        output = float(memory.output_size_in_bytes)
        device_memory = temp + argument + output - float(memory.alias_size_in_bytes)

    flops = bytes_accessed = None
    if isinstance(cost, dict):
        if "flops" in cost:
            flops = float(cost["flops"])
        if "bytes accessed" in cost:
            bytes_accessed = float(cost["bytes accessed"])

    return CostEstimate(
        fn_name=fn_name,
        device_memory_bytes=device_memory,
        temp_bytes=temp,
        argument_bytes=argument,
        output_bytes=output,
        flops=flops,
        bytes_accessed=bytes_accessed,
        num_gpus=n_gpus,
        device_byte_sizes=device_byte_sizes,
    )
