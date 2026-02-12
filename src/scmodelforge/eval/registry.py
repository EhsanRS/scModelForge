"""Benchmark registry — map string names to benchmark classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scmodelforge.eval.base import BaseBenchmark

_REGISTRY: dict[str, type[BaseBenchmark]] = {}

_state = {"plugins_loaded": False}


def _ensure_plugins() -> None:
    """Discover third-party benchmark plugins on first access."""
    if _state["plugins_loaded"]:
        return
    _state["plugins_loaded"] = True
    from scmodelforge._plugins import BENCHMARK_GROUP, discover_plugins

    discover_plugins(BENCHMARK_GROUP, _REGISTRY)


def register_benchmark(name: str):
    """Class decorator that registers a benchmark under *name*.

    Raises
    ------
    ValueError
        If *name* is already registered.
    """

    def decorator(cls: type[BaseBenchmark]) -> type[BaseBenchmark]:
        if name in _REGISTRY:
            msg = f"Benchmark '{name}' is already registered (class={_REGISTRY[name].__name__})."
            raise ValueError(msg)
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_benchmark(name: str, **kwargs: Any) -> BaseBenchmark:
    """Instantiate a registered benchmark by name.

    Parameters
    ----------
    name
        Registry key (e.g. ``"linear_probe"``).
    **kwargs
        Forwarded to the benchmark constructor.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    _ensure_plugins()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        msg = f"Unknown benchmark '{name}'. Available: {available}"
        raise ValueError(msg)
    return _REGISTRY[name](**kwargs)


def list_benchmarks() -> list[str]:
    """Return sorted list of registered benchmark names."""
    _ensure_plugins()
    return sorted(_REGISTRY)
