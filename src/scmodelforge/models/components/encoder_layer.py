"""Custom encoder layer with pluggable attention.

Drop-in replacement for ``nn.TransformerEncoderLayer`` that accepts any
attention module and preserves the exact submodule names that peft LoRA
targets: ``self_attn``, ``linear1``, ``linear2``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScModelForgeEncoderLayer(nn.Module):
    """Transformer encoder layer with pluggable attention.

    Mirrors ``nn.TransformerEncoderLayer`` submodule names exactly:

    - ``self_attn``: the attention module (any custom type)
    - ``linear1``: first FFN linear layer
    - ``linear2``: second FFN linear layer
    - ``norm1``: pre-attention LayerNorm
    - ``norm2``: pre-FFN LayerNorm
    - ``dropout1``, ``dropout2``: dropout layers

    Uses pre-norm (``norm_first=True``) architecture matching our
    existing models.

    Parameters
    ----------
    self_attn
        Attention module (e.g. ``FlashSelfAttention``, ``GeneGeneAttention``).
    d_model
        Model dimension.
    dim_feedforward
        FFN intermediate dimension.
    dropout
        Dropout probability.
    activation
        Activation function name for the FFN.
    layer_norm_eps
        Epsilon for LayerNorm.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._activation = self._get_activation(activation)

    @staticmethod
    def _get_activation(name: str):
        """Return an activation function by name."""
        if name == "gelu":
            return F.gelu
        if name == "relu":
            return F.relu
        msg = f"Unknown activation: {name!r}"
        raise ValueError(msg)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Pre-norm transformer encoder layer forward pass.

        Parameters
        ----------
        src
            Input tensor ``(B, S, D)``.
        src_mask
            Attention mask ``(S, S)`` or ``(B*nhead, S, S)``.
        src_key_padding_mask
            Padding mask ``(B, S)`` where ``True`` = ignore.
        **kwargs
            Passed through to ``self_attn.forward()`` (e.g. ``gene_indices``).

        Returns
        -------
        torch.Tensor
            Output tensor ``(B, S, D)``.
        """
        # Pre-norm → attention → residual
        x = self.norm1(src)
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            **kwargs,
        )
        src = src + self.dropout1(attn_out)

        # Pre-norm → FFN → residual
        x = self.norm2(src)
        ff_out = self.linear2(self._activation(self.linear1(x)))
        return src + self.dropout2(ff_out)
