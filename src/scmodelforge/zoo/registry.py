"""External model registry — map string names to adapter classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scmodelforge.zoo.base import BaseModelAdapter

_REGISTRY: dict[str, type[BaseModelAdapter]] = {}

_state = {"plugins_loaded": False}


def _ensure_plugins() -> None:
    """Discover third-party zoo plugins on first access."""
    if _state["plugins_loaded"]:
        return
    _state["plugins_loaded"] = True
    from scmodelforge._plugins import ZOO_GROUP, discover_plugins

    discover_plugins(ZOO_GROUP, _REGISTRY)


def register_external_model(name: str):
    """Class decorator that registers an external model adapter under *name*.

    Raises
    ------
    ValueError
        If *name* is already registered.
    """

    def decorator(cls: type[BaseModelAdapter]) -> type[BaseModelAdapter]:
        if name in _REGISTRY:
            msg = f"External model '{name}' is already registered (class={_REGISTRY[name].__name__})."
            raise ValueError(msg)
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_external_model(name: str, **kwargs: Any) -> BaseModelAdapter:
    """Instantiate a registered external model adapter by name.

    Parameters
    ----------
    name
        Registry key (e.g. ``"geneformer"``).
    **kwargs
        Forwarded to the adapter constructor.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    _ensure_plugins()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        msg = f"Unknown external model '{name}'. Available: {available}"
        raise ValueError(msg)
    return _REGISTRY[name](**kwargs)


def list_external_models() -> list[str]:
    """Return sorted list of registered external model names."""
    _ensure_plugins()
    return sorted(_REGISTRY)
