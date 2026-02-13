"""Attention utilities and encoder builders for transformer models."""

from __future__ import annotations

import torch
import torch.nn as nn

from scmodelforge.models.components.custom_attention import build_attention
from scmodelforge.models.components.encoder import ScModelForgeEncoder
from scmodelforge.models.components.encoder_layer import ScModelForgeEncoderLayer


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


def build_encoder_layer(
    attention_type: str,
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    dropout: float = 0.1,
    activation: str = "gelu",
    layer_norm_eps: float = 1e-12,
    max_genes: int = 30000,
    gene_bias_init_std: float = 0.02,
) -> nn.TransformerEncoderLayer | ScModelForgeEncoderLayer:
    """Build an encoder layer with the given attention type.

    Returns ``nn.TransformerEncoderLayer`` for ``"standard"`` to preserve
    exact backward compatibility.  Returns :class:`ScModelForgeEncoderLayer`
    for all custom attention types.

    Parameters
    ----------
    attention_type
        One of ``"standard"``, ``"flash"``, ``"gene_bias"``, ``"linear"``.
    d_model
        Model dimension.
    nhead
        Number of attention heads.
    dim_feedforward
        FFN intermediate dimension.
    dropout
        Dropout probability.
    activation
        FFN activation function name.
    layer_norm_eps
        LayerNorm epsilon.
    max_genes
        Max gene vocab size (for gene_bias attention).
    gene_bias_init_std
        Gene bias initialisation std (for gene_bias attention).

    Returns
    -------
    nn.TransformerEncoderLayer | ScModelForgeEncoderLayer
    """
    if attention_type == "standard":
        return nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )

    attn = build_attention(
        attention_type=attention_type,
        d_model=d_model,
        nhead=nhead,
        dropout=dropout,
        max_genes=max_genes,
        gene_bias_init_std=gene_bias_init_std,
    )
    return ScModelForgeEncoderLayer(
        self_attn=attn,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
    )


def build_encoder(
    attention_type: str,
    encoder_layer: nn.TransformerEncoderLayer | ScModelForgeEncoderLayer,
    num_layers: int,
    d_model: int | None = None,
    layer_norm_eps: float = 1e-12,
) -> nn.TransformerEncoder | ScModelForgeEncoder:
    """Build an encoder stack.

    Returns ``nn.TransformerEncoder`` for ``"standard"`` and ``"flash"``
    (no kwargs needed). Returns :class:`ScModelForgeEncoder` for
    ``"gene_bias"`` and ``"linear"`` (need ``**kwargs`` propagation).

    Parameters
    ----------
    attention_type
        Attention type string.
    encoder_layer
        A single encoder layer (will be deep-copied internally).
    num_layers
        Number of encoder layers.
    d_model
        Model dimension (only needed for ``ScModelForgeEncoder`` norm).
    layer_norm_eps
        LayerNorm epsilon.

    Returns
    -------
    nn.TransformerEncoder | ScModelForgeEncoder
    """
    if attention_type in ("standard", "flash"):
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # gene_bias, linear — need **kwargs propagation
    import copy

    layers = nn.ModuleList([encoder_layer] + [copy.deepcopy(encoder_layer) for _ in range(num_layers - 1)])
    return ScModelForgeEncoder(layers=layers)
