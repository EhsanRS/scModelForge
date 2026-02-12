"""Attention utilities for transformer models."""

from __future__ import annotations

import torch


def generate_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Generate a causal (autoregressive) attention mask.

    Returns a ``(S, S)`` float mask where allowed positions are ``0.0``
    and blocked (future) positions are ``-inf``, compatible with
    ``nn.TransformerEncoder``'s ``mask`` parameter.

    Parameters
    ----------
    seq_len
        Sequence length.
    device
        Target device for the mask tensor.

    Returns
    -------
    torch.Tensor
        Causal mask of shape ``(S, S)``.
    """
    return torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
