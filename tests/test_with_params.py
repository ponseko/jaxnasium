import inspect
import pkgutil
from importlib import import_module

import pytest

import jaxnasium.algorithms

# `key` is the exception; here we follow equinox's convention.
NOT_PRE_BOUND = {"key"}
NOT_FORWARDED_AT_DEFAULT = {"channels_axis"}

"""
Tests to ensure that architectures that expose .with_params(), have this
function match the constructor signature.
"""


def _constructor_keywords(cls) -> set[str]:
    """Keyword-only `__init__` params, looking past `*args, **kwargs` forwarders."""
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        params = inspect.signature(init).parameters
        keywords = {
            name
            for name, p in params.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        forwards = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        if keywords or not forwards:
            return keywords - NOT_PRE_BOUND
    return set()


def _classes_with_with_params():
    """Every class in `jaxnasium` exposing its own `with_params`."""
    found = {}
    packages = [jaxnasium.algorithms]
    seen_modules = set()
    while packages:
        pkg = packages.pop()
        for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if info.name in seen_modules:
                continue
            seen_modules.add(info.name)
            try:
                module = import_module(info.name)
            except Exception:  # optional extras, deprecated shims
                continue
            for name, obj in vars(module).items():
                if not isinstance(obj, type) or name.startswith("_"):
                    continue
                if "with_params" not in vars(obj):
                    continue  # inherited, not its own -- checked on the defining class
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return found


CLASSES = _classes_with_with_params()


def test_discovery_found_the_known_classes():
    """Guard the test itself: if discovery silently finds nothing, it proves nothing."""
    assert len(CLASSES) >= 5, (
        f"expected to discover several classes, got {list(CLASSES)}"
    )
    names = {n.rsplit(".", 1)[-1] for n in CLASSES}
    assert {"MLP", "CNN", "BroNet"} <= names, f"missing known classes, found {names}"


@pytest.mark.parametrize("qualname", sorted(CLASSES))
def test_with_params_matches_constructor_keywords(qualname):
    cls = CLASSES[qualname]
    expected = _constructor_keywords(cls)
    actual = {
        name
        for name, p in inspect.signature(cls.with_params).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    } - NOT_PRE_BOUND

    missing = expected - actual
    extra = actual - expected
    assert not missing, (
        f"{qualname}.with_params() is missing {sorted(missing)}, so passing them would "
        f"be silently dropped. Add them to `with_params` as well as `__init__`."
    )
    assert not extra, (
        f"{qualname}.with_params() accepts {sorted(extra)}, which `__init__` does not "
        f"take -- passing them raises from inside the constructor."
    )


@pytest.mark.parametrize("qualname", sorted(CLASSES))
def test_with_params_actually_forwards_what_it_accepts(qualname):
    """A parameter present in both signatures must still reach the constructor."""
    cls = CLASSES[qualname]
    signature = inspect.signature(cls.with_params)
    bindable = {
        name: p.default
        for name, p in signature.parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
        and p.default is not inspect.Parameter.empty
        and name not in NOT_PRE_BOUND
    }
    if not bindable:
        pytest.skip(f"{qualname}.with_params has no defaulted keywords to bind")

    bound = cls.with_params(**bindable)
    assert bound.keywords is not None, (
        f"{qualname}.with_params did not return a partial"
    )
    not_forwarded = set(bindable) - set(bound.keywords) - NOT_FORWARDED_AT_DEFAULT
    assert not not_forwarded, (
        f"{qualname}.with_params() accepts {sorted(not_forwarded)} but does not forward "
        f"them to the constructor."
    )
