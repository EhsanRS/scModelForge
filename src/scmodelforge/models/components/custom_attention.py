"""Custom attention mechanism implementations.

Provides pluggable attention modules that share the same submodule naming
convention as ``nn.MultiheadAttention`` (``in_proj_weight``, ``in_proj_bias``,
``out_proj``), so LoRA and other adapters work unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_VALID_ATTENTION_TYPES = frozenset({"standard", "flash", "gene_bias", "linear"})


class FlashSelfAttention(nn.Module):
    """Multi-head attention using ``F.scaled_dot_product_attention``.

    Uses PyTorch's SDPA backend selection (flash, math, or mem-efficient)
    automatically.  Mirrors ``nn.MultiheadAttention`` submodule names so
    that peft LoRA finds ``out_proj``.

    Parameters
    ----------
    d_model
        Total model dimension.
    nhead
        Number of attention heads.
    dropout
        Attention dropout probability.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        # Attribute expected by nn.TransformerEncoder to determine batch dim
        self.batch_first = True

        # Combined QKV projection (matches nn.MultiheadAttention naming)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass compatible with ``nn.MultiheadAttention`` interface.

        Parameters
        ----------
        query, key, value
            Input tensors of shape ``(B, S, D)``.
        key_padding_mask
            Boolean mask ``(B, S)`` where ``True`` means ignore.
        attn_mask
            Additive float mask ``(S, S)`` or ``(B*nhead, S, S)``.

        Returns
        -------
        tuple[torch.Tensor, None]
            Output tensor ``(B, S, D)`` and ``None`` (no attention weights).
        """
        B, S, _ = query.shape

        # QKV projection
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape: (B, S, D) -> (B, nhead, S, head_dim)
        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Build SDPA attention mask
        sdpa_mask = None
        if attn_mask is not None or key_padding_mask is not None:
            sdpa_mask = self._build_sdpa_mask(attn_mask, key_padding_mask, B, S, query.dtype, query.device)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=dropout_p)

        # Reshape back: (B, nhead, S, head_dim) -> (B, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(attn_out), None

    def _build_sdpa_mask(
        self,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        B: int,
        S: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Combine attn_mask and key_padding_mask into a single SDPA mask."""
        # Start with zeros
        mask = torch.zeros(B, 1, S, S, dtype=dtype, device=device)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                mask = mask + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                # (B*nhead, S, S) -> (B, nhead, S, S)
                mask = mask + attn_mask.view(B, self.nhead, S, S)

        if key_padding_mask is not None:
            # nn.TransformerEncoder may convert bool mask to float (0/-inf)
            if key_padding_mask.dtype == torch.bool:
                pad_mask = torch.zeros(B, 1, 1, S, dtype=dtype, device=device)
                pad_mask = pad_mask.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            else:
                # Already float: 0.0=attend, -inf=ignore
                pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(dtype=dtype)
            mask = mask + pad_mask

        return mask


class GeneGeneAttention(nn.Module):
    """Multi-head attention with additive gene-gene bias matrix.

    Adds a learnable bias ``gene_bias[i, j]`` to the attention score
    between genes *i* and *j* before softmax.  The bias is indexed by
    ``gene_indices`` passed via ``forward(**kwargs)``.

    Parameters
    ----------
    d_model
        Total model dimension.
    nhead
        Number of attention heads.
    dropout
        Attention dropout probability.
    max_genes
        Maximum gene vocabulary size for the bias matrix.
    gene_bias_init_std
        Std for the bias initialisation.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        max_genes: int = 30000,
        gene_bias_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        self.batch_first = True

        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable gene-gene bias shared across heads
        self.gene_bias = nn.Parameter(torch.empty(max_genes, max_genes))
        nn.init.normal_(self.gene_bias, std=gene_bias_init_std)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass with gene-gene attention bias.

        Parameters
        ----------
        query, key, value
            Input tensors ``(B, S, D)``.
        key_padding_mask
            Boolean mask ``(B, S)`` where ``True`` = ignore.
        attn_mask
            Additive float mask.
        **kwargs
            Must contain ``gene_indices`` of shape ``(B, S)`` — token IDs
            used to index the gene-gene bias matrix.

        Returns
        -------
        tuple[torch.Tensor, None]
        """
        gene_indices = kwargs.get("gene_indices")
        B, S, _ = query.shape

        # QKV projection
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, nhead, S, S)

        # Add gene-gene bias
        if gene_indices is not None:
            # Index: gene_bias[gene_indices[b, i], gene_indices[b, j]]
            row_idx = gene_indices.unsqueeze(2).expand(-1, -1, S)  # (B, S, S)
            col_idx = gene_indices.unsqueeze(1).expand(-1, S, -1)  # (B, S, S)
            bias = self.gene_bias[row_idx, col_idx]  # (B, S, S)
            scores = scores + bias.unsqueeze(1)  # broadcast across heads

        # Apply masks
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                scores = scores + attn_mask.view(B, self.nhead, S, S)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_weights = F.softmax(scores, dim=-1)
        dropout_p = self.dropout if self.training else 0.0
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(attn_out), None


class LinearAttention(nn.Module):
    """Linear attention using kernel feature maps for O(n) complexity.

    Uses the ELU+1 feature map: ``phi(x) = elu(x) + 1``.
    Computes: ``out = (phi(Q) @ (phi(K)^T @ V)) / (phi(Q) @ phi(K)^T @ 1)``.

    Parameters
    ----------
    d_model
        Total model dimension.
    nhead
        Number of attention heads.
    dropout
        Dropout probability (applied to output, not attention weights).
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        self.batch_first = True

        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
        """ELU+1 feature map for linear attention."""
        return F.elu(x) + 1.0

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass with linear attention.

        Parameters
        ----------
        query, key, value
            Input tensors ``(B, S, D)``.
        key_padding_mask
            Boolean mask ``(B, S)`` where ``True`` = ignore.
        attn_mask
            Not used (linear attention doesn't support arbitrary masks).

        Returns
        -------
        tuple[torch.Tensor, None]
        """
        B, S, _ = query.shape

        # QKV projection
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Apply feature map
        q = self._elu_feature_map(q)
        k = self._elu_feature_map(k)

        # Zero out keys at padded positions
        if key_padding_mask is not None:
            # key_padding_mask: (B, S), True=ignore
            mask = (~key_padding_mask).float().unsqueeze(1).unsqueeze(-1)  # (B, 1, S, 1)
            k = k * mask
            v = v * mask

        # Linear attention: O(n) via associativity
        # KV = phi(K)^T @ V : (B, nhead, head_dim, head_dim)
        kv = torch.matmul(k.transpose(-2, -1), v)
        # numerator = phi(Q) @ KV : (B, nhead, S, head_dim)
        numerator = torch.matmul(q, kv)

        # Denominator for normalisation
        # Z = phi(Q) @ phi(K)^T @ 1 : (B, nhead, S, 1)
        k_sum = k.sum(dim=-2, keepdim=True)  # (B, nhead, 1, head_dim)
        denominator = torch.matmul(q, k_sum.transpose(-2, -1))  # (B, nhead, S, 1)
        denominator = denominator.clamp(min=1e-6)

        attn_out = numerator / denominator

        # Reshape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        if self.dropout > 0.0 and self.training:
            attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)

        return self.out_proj(attn_out), None


def build_attention(
    attention_type: str,
    d_model: int,
    nhead: int,
    dropout: float = 0.0,
    max_genes: int = 30000,
    gene_bias_init_std: float = 0.02,
) -> nn.Module:
    """Factory function returning the appropriate attention module.

    Parameters
    ----------
    attention_type
        One of ``"standard"``, ``"flash"``, ``"gene_bias"``, ``"linear"``.
    d_model
        Model dimension.
    nhead
        Number of attention heads.
    dropout
        Attention dropout.
    max_genes
        Max gene vocab size (for gene_bias).
    gene_bias_init_std
        Std for gene bias init (for gene_bias).

    Returns
    -------
    nn.Module
        An attention module with ``nn.MultiheadAttention``-compatible interface.

    Raises
    ------
    ValueError
        If ``attention_type`` is not recognised.
    """
    if attention_type not in _VALID_ATTENTION_TYPES:
        msg = f"Unknown attention type {attention_type!r}. Choose from: {sorted(_VALID_ATTENTION_TYPES)}"
        raise ValueError(msg)

    if attention_type == "standard":
        return nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    if attention_type == "flash":
        return FlashSelfAttention(d_model, nhead, dropout=dropout)

    if attention_type == "gene_bias":
        return GeneGeneAttention(
            d_model, nhead, dropout=dropout, max_genes=max_genes, gene_bias_init_std=gene_bias_init_std,
        )

    # linear
    return LinearAttention(d_model, nhead, dropout=dropout)
