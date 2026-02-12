"""Tokenizer registry — map string names to tokenizer classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scmodelforge.tokenizers.base import BaseTokenizer

_REGISTRY: dict[str, type[BaseTokenizer]] = {}

_state = {"plugins_loaded": False}


def _ensure_plugins() -> None:
    """Discover third-party tokenizer plugins on first access."""
    if _state["plugins_loaded"]:
        return
    _state["plugins_loaded"] = True
    from scmodelforge._plugins import TOKENIZER_GROUP, discover_plugins

    discover_plugins(TOKENIZER_GROUP, _REGISTRY)


def register_tokenizer(name: str):
    """Class decorator that registers a tokenizer under *name*.

    Raises
    ------
    ValueError
        If *name* is already registered.
    """

    def decorator(cls: type[BaseTokenizer]) -> type[BaseTokenizer]:
        if name in _REGISTRY:
            raise ValueError(f"Tokenizer '{name}' is already registered (class={_REGISTRY[name].__name__}).")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_tokenizer(name: str, **kwargs: Any) -> BaseTokenizer:
    """Instantiate a registered tokenizer by name.

    Parameters
    ----------
    name
        Registry key (e.g. ``"rank_value"``).
    **kwargs
        Forwarded to the tokenizer constructor.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    _ensure_plugins()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown tokenizer '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_tokenizers() -> list[str]:
    """Return sorted list of registered tokenizer names."""
    _ensure_plugins()
    return sorted(_REGISTRY)
