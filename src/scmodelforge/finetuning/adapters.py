"""LoRA adapter utilities wrapping HuggingFace ``peft``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from scmodelforge.config.schema import LoRAConfig

DEFAULT_TARGET_MODULES = ["out_proj", "linear1", "linear2"]


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Wrap a backbone model with LoRA adapters.

    Parameters
    ----------
    model
        The backbone ``nn.Module`` to wrap.
    config
        LoRA configuration specifying rank, alpha, dropout, etc.

    Returns
    -------
    nn.Module
        A ``PeftModel`` wrapping the original model. Non-LoRA parameters
        are frozen; only adapter parameters are trainable.
    """
    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
    except ImportError:
        msg = (
            "peft is required for LoRA support. "
            "Install with: pip install peft"
        )
        raise ImportError(msg) from None

    target = config.target_modules or DEFAULT_TARGET_MODULES
    peft_config = PeftLoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=target,
        bias=config.bias,
    )
    return get_peft_model(model, peft_config)


def has_lora(model: nn.Module) -> bool:
    """Check if a model is wrapped with peft LoRA.

    Parameters
    ----------
    model
        The model to check.

    Returns
    -------
    bool
        ``True`` if the model is a ``PeftModel``.
    """
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(model, PeftModel)


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights.

    Parameters
    ----------
    model
        A ``PeftModel`` with LoRA adapters.
    path
        Directory path to save the adapter weights.

    Raises
    ------
    ValueError
        If the model does not have LoRA adapters.
    """
    if not has_lora(model):
        msg = "Model does not have LoRA adapters."
        raise ValueError(msg)
    model.save_pretrained(path)


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA adapter weights onto a base model.

    Parameters
    ----------
    model
        The base model (without LoRA adapters).
    path
        Directory path containing saved adapter weights.

    Returns
    -------
    nn.Module
        A ``PeftModel`` with loaded adapter weights.
    """
    try:
        from peft import PeftModel
    except ImportError:
        msg = (
            "peft is required for LoRA support. "
            "Install with: pip install peft"
        )
        raise ImportError(msg) from None
    return PeftModel.from_pretrained(model, path)


def count_lora_parameters(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts.

    Parameters
    ----------
    model
        A model, optionally wrapped with LoRA.

    Returns
    -------
    tuple[int, int]
        ``(trainable_params, total_params)``.
    """
    if has_lora(model):
        return model.get_nb_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
