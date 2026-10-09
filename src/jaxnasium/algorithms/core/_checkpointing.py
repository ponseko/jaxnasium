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


def _reference(x: Any) -> str | None:
    """Importable `module:qualname` of `x`.
    If `x` cannot be looked up by name (e.g. for elements defined inside functions), this returns None."""
    module_name = getattr(x, "__module__", None)
    qualname = getattr(x, "__qualname__", None)
    if module_name is None or qualname is None or "<locals>" in qualname:
        return None
    reference = f"{module_name}:{qualname}"
    try:
        if _resolve(reference) is x:
            return reference
    except Exception:
        return None
    return None


def _marshal(x: Any):
    if isinstance(x, functools.partial):
        if _reference(type(x)) is None:
            return None
        # jax.tree_util.Partial is also a functool.partial; so we save the cls here too
        return _PARTIAL, {
            "cls": type(x),
            "func": x.func,
            "args": list(x.args),
            "kwargs": dict(x.keywords),
            "dict": dict(vars(x)),
        }

    if isinstance(x, tuple) and hasattr(x, "_fields"):  # NamedTuple, e.g. optax states
        cls_reference = _reference(type(x))
        if cls_reference is None:
            return None
        return _NAMEDTUPLE, {
            "cls": cls_reference,
            "fields": dict(zip(x._fields, x)),  # type: ignore
        }

    reference = _reference(x)
    if reference is None:
        return None
    return _QUALNAME, reference


def _unmarshal(type_info: str, marshaled: Any):
    if type_info == _QUALNAME:
        return _resolve(marshaled)
    if type_info == _NAMEDTUPLE:
        cls = _resolve(marshaled["cls"])
        if not (
            isinstance(cls, type) and issubclass(cls, tuple) and hasattr(cls, "_fields")
        ):
            raise TypeError(f"{marshaled['cls']!r} is not a NamedTuple class")
        return cls(**marshaled["fields"])
    if type_info == _PARTIAL:
        cls = marshaled.get("cls", functools.partial)
        if not (isinstance(cls, type) and issubclass(cls, functools.partial)):
            raise TypeError(f"{cls!r} is not a functools.partial class")
        out = cls(marshaled["func"], *marshaled["args"], **marshaled["kwargs"])
        out.__dict__.update(marshaled.get("dict", {}))
        return out
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
