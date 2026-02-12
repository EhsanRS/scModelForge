"""Gene expression embedding module."""

from __future__ import annotations

import torch
import torch.nn as nn

from scmodelforge._constants import PAD_TOKEN_ID


class GeneExpressionEmbedding(nn.Module):
    """Combined gene token + positional + optional expression value embedding.

    Parameters
    ----------
    vocab_size
        Number of tokens in the gene vocabulary (including special tokens).
    hidden_dim
        Embedding dimension.
    max_seq_len
        Maximum sequence length for learned positional embeddings.
    dropout
        Dropout probability applied after the final LayerNorm.
    use_expression_values
        If ``True``, project scalar expression values to ``hidden_dim`` and add
        them to the token embeddings.
    layer_norm_eps
        Epsilon for LayerNorm.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        *,
        use_expression_values: bool = True,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.gene_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN_ID)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.use_expression_values = use_expression_values
        if use_expression_values:
            self.expression_proj = nn.Linear(1, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute embeddings.

        Parameters
        ----------
        input_ids
            Token IDs of shape ``(B, S)``.
        values
            Optional expression values of shape ``(B, S)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(B, S, H)``.
        """
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        emb = self.gene_embedding(input_ids) + self.position_embedding(position_ids)

        if self.use_expression_values and values is not None:
            expr_emb = self.expression_proj(values.unsqueeze(-1))
            emb = emb + expr_emb

        return self.dropout(self.layer_norm(emb))
