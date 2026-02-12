"""Model registry — map string names to model classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from scmodelforge.config.schema import ModelConfig

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str):
    """Class decorator that registers a model under *name*.

    Raises
    ------
    ValueError
        If *name* is already registered.
    """

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered (class={_MODEL_REGISTRY[name].__name__}).")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, config: ModelConfig) -> nn.Module:
    """Instantiate a registered model by name using a :class:`ModelConfig`.

    Parameters
    ----------
    name
        Registry key (e.g. ``"transformer_encoder"``).
    config
        Model configuration — forwarded to the class's ``from_config()`` method.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    cls = _MODEL_REGISTRY[name]
    return cls.from_config(config)  # type: ignore[attr-defined]


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_MODEL_REGISTRY)
