import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import equinox as eqx
import jax

logger = logging.getLogger(__name__)

DEFAULT_CACHE_SUBDIR = "jaxnasium/jit-cache"


def _default_cache_dir() -> Path:
    root = (
        os.environ.get("CACHE_HOME_DIR")
        or os.environ.get("XDG_CACHE_HOME")
        or (Path.home() / ".cache")
    )
    return Path(root) / DEFAULT_CACHE_SUBDIR


def enable_compilation_cache(
    path: str | os.PathLike | None = None,
    *,
    min_compile_time_secs: float = 0.01,
    min_entry_size_bytes: int = 0,
) -> Path:
    """Convenience function to enable JAX's persistent compilation cache.

    Default path is `$CACHE_HOME_DIR/jaxnasium/jit-cache` or
    `$XDG_CACHE_HOME/jaxnasium/jit-cache` or `~/.cache/jaxnasium/jit-cache`.

    https://docs.jax.dev/en/latest/persistent_compilation_cache.html
    """
    cache_dir = Path(path) if path is not None else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs", min_compile_time_secs
    )
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size_bytes)

    logger.info("Persistent compilation cache enabled at %s", cache_dir)
    return cache_dir


def _format_secs(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    minutes, seconds = divmod(secs, 60)
    return f"{int(minutes)}m{seconds:04.1f}s"


def _log(text: str) -> None:
    try:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()
    except Exception:
        pass


class _Step:
    """Ticks the elapsed time on one line, then replaces it with a checkmark."""

    def __init__(self, doing: str, done: str):
        self.doing = doing
        self.done = done
        self.start = time.perf_counter()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _write(self, text: str) -> None:
        try:
            sys.stderr.write(text)
            sys.stderr.flush()
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            self._write(f"\r\033[2K  {self.doing} ... {_format_secs(self.elapsed)}")

    def __enter__(self) -> Self:
        try:
            animate = sys.stderr.isatty()
        except Exception:
            animate = False
        if animate:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._write("\r\033[2K")
        _log(f"  ✓ {self.done} in {_format_secs(self.elapsed)}")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start


@dataclass(frozen=True)
class CompiledFunction:
    """A compiled function stored with the arguments it was compiled for.
    Convenient for precompiling functions, then calling it with no arguments.
    """

    compiled_fn: Any
    lowered: Any
    name: str
    compile_secs: float
    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not args and not kwargs:
            args, kwargs = self._args, self._kwargs
        return self.compiled_fn(*args, **kwargs)


def precompile(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> CompiledFunction:
    """Compile `fn(*args, **kwargs)` ahead of time, with some progress to stderr
    to keep you entertained while compiling; especially since compiling the full
    training loop can take a while.

    https://docs.jax.dev/en/latest/aot.html

    Additionally, the returned function stores the arguments it was compiled for,
    so you can call it directly with no arguments.

    ```python
    train = jym.precompile(ppo.train, key, env)
    # Compiling `train` ...
    #   ✓ traced and lowered in 6.3s
    #   ✓ compiled with XLA in 12.8s
    # ✓ `train` ready in 19.1s
    agent, metrics = train()
    ```
    """

    # Attempt to get a pretty name of the function.
    name, unwrapped = None, fn
    for _ in range(8):
        name = getattr(unwrapped, "__name__", None)
        if isinstance(name, str):
            break
        inner = getattr(unwrapped, "__wrapped__", None) or getattr(
            unwrapped, "func", None
        )
        if inner is None:
            break
        unwrapped = inner
    if not isinstance(name, str):
        name = type(fn).__name__

    start = time.perf_counter()
    _log(f"Compiling `{name}` ...")

    with _Step("tracing and lowering", "traced and lowered"):
        lowered = eqx.filter_jit(fn).lower(*args, **kwargs)  # type: ignore

    with _Step("compiling with XLA", "compiled with XLA"):
        compiled = lowered.compile()

    elapsed = time.perf_counter() - start
    _log(f"✓  `{name}` ready in {_format_secs(elapsed)}")
    logger.info("Precompiled %s in %s", name, _format_secs(elapsed))

    return CompiledFunction(compiled, lowered, name, elapsed, args, kwargs)
