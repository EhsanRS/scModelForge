"""Model utility functions: weight initialization and parameter counting."""

from __future__ import annotations

import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier uniform for linear layers and normal for embeddings.

    Applied via ``model.apply(init_weights)``.

    Parameters
    ----------
    module
        A single ``nn.Module`` (called recursively by ``apply``).
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight.data[module.padding_idx])
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Parameters
    ----------
    model
        The model to count parameters for.
    trainable_only
        If ``True`` (default), count only parameters that require gradients.

    Returns
    -------
    int
        Total number of (trainable) parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
