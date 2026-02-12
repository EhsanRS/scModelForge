"""Pooling functions for extracting cell-level embeddings from token sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def cls_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return the hidden state of the first (CLS) token.

    Parameters
    ----------
    hidden_states
        Shape ``(B, S, H)``.
    attention_mask
        Unused — accepted for API symmetry with :func:`mean_pool`.

    Returns
    -------
    torch.Tensor
        Shape ``(B, H)``.
    """
    return hidden_states[:, 0]


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-pool over non-padding tokens.

    Parameters
    ----------
    hidden_states
        Shape ``(B, S, H)``.
    attention_mask
        Shape ``(B, S)`` with 1 for real tokens and 0 for padding.
        If ``None``, all positions are treated as real tokens.

    Returns
    -------
    torch.Tensor
        Shape ``(B, H)``.
    """
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
    summed = (hidden_states * mask).sum(dim=1)  # (B, H)
    lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1) — avoid div-by-zero
    return summed / lengths
