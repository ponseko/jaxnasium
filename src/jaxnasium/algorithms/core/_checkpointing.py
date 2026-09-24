import functools
import importlib
from typing import Any

_QUALNAME = "jaxnasium[qualname]"
_NAMEDTUPLE = "jaxnasium[namedtuple]"
_PARTIAL = "jaxnasium[partial]"

"""Agent checkpointing, built on `jaxon`.
Alternatively, checkpointing can be done like any other eqx.Module as described here:
https://docs.kidger.site/equinox/examples/serialisation/
"""


def _resolve(reference: str) -> Any:
    module_name, qualname = reference.split(":")
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _marshal(x: Any):
    if isinstance(x, functools.partial):
        # members are marshaled recursively, so `func` goes through _QUALNAME below
        return _PARTIAL, {
            "func": x.func,
            "args": list(x.args),
            "kwargs": dict(x.keywords),
        }

    if isinstance(x, tuple) and hasattr(x, "_fields"):  # NamedTuple, e.g. optax states
        cls = type(x)
        return _NAMEDTUPLE, {
            "cls": f"{cls.__module__}:{cls.__qualname__}",
            "fields": dict(zip(x._fields, x)),  # type: ignore
        }

    module_name = getattr(x, "__module__", None)
    qualname = getattr(x, "__qualname__", None)
    if module_name is None or qualname is None or "<locals>" in qualname:
        return None
    reference = f"{module_name}:{qualname}"
    try:
        if _resolve(reference) is x:
            return _QUALNAME, reference
    except Exception:
        return None
    return None


def _unmarshal(type_info: str, marshaled: Any):
    if type_info == _QUALNAME:
        return _resolve(marshaled)
    if type_info == _NAMEDTUPLE:
        return _resolve(marshaled["cls"])(**marshaled["fields"])
    if type_info == _PARTIAL:
        return functools.partial(
            marshaled["func"], *marshaled["args"], **marshaled["kwargs"]
        )
    return None


def save_agent(file_path: str, agent: Any) -> None:
    """Writes `agent` -- parameters, optimizer state and trainer -- to `file_path` using `jaxon`."""
    try:
        import jaxon
    except ImportError as e:
        raise ImportError(
            "Agent checkpointing requires `jaxon`. Install it with `pip install jaxon`."
        ) from e
    jaxon.save(file_path, agent, custom_marshalers=[_marshal])


def load_agent(file_path: str) -> Any:
    """Reads a checkpoint written by `save_agent` using `jaxon`."""
    try:
        import jaxon
    except ImportError as e:
        raise ImportError(
            "Agent checkpointing requires `jaxon`. Install it with `pip install jaxon`."
        ) from e
    return jaxon.load(file_path, custom_unmarshalers=[_unmarshal])
