"""Gene embedding tokenizer — uses pretrained gene representations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from scmodelforge._constants import CLS_TOKEN_ID
from scmodelforge.tokenizers._utils import ensure_tensor, load_gene_embeddings
from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell
from scmodelforge.tokenizers.registry import register_tokenizer

if TYPE_CHECKING:
    import numpy as np

    from scmodelforge.data.gene_vocab import GeneVocab


@register_tokenizer("gene_embedding")
class GeneEmbeddingTokenizer(BaseTokenizer):
    """Tokenizer that carries a pretrained gene embedding matrix.

    Follows the same tokenization flow as
    :class:`~scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer`
    (filter zeros, truncate, optional CLS) but additionally stores a gene
    embedding matrix that downstream models can access via the
    :attr:`gene_embeddings` property.

    Parameters
    ----------
    gene_vocab
        Gene vocabulary.
    max_len
        Maximum sequence length (including CLS).
    prepend_cls
        Whether to prepend a ``[CLS]`` token.
    embedding_path
        Path to a pretrained gene embedding file (``.pt``, ``.pth``, or
        ``.npy``). If ``None``, no embeddings are loaded (can be set later
        via :meth:`set_gene_embeddings`).
    embedding_dim
        Expected embedding dimension (used for validation).
    """

    def __init__(
        self,
        gene_vocab: GeneVocab,
        max_len: int = 2048,
        prepend_cls: bool = True,
        embedding_path: str | None = None,
        embedding_dim: int = 200,
    ) -> None:
        super().__init__(gene_vocab, max_len)
        self.prepend_cls = prepend_cls
        self._embedding_dim = embedding_dim

        if embedding_path is not None:
            self._embedding_matrix: torch.Tensor | None = load_gene_embeddings(
                embedding_path, gene_vocab,
            )
            self._embedding_dim = self._embedding_matrix.shape[1]
        else:
            self._embedding_matrix = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.gene_vocab)

    @property
    def strategy_name(self) -> str:
        return "gene_embedding"

    @property
    def gene_embeddings(self) -> torch.Tensor | None:
        """Pretrained gene embedding matrix, or ``None``."""
        return self._embedding_matrix

    @property
    def embedding_dim(self) -> int | None:
        """Embedding dimension, or ``None`` if no embeddings are loaded."""
        if self._embedding_matrix is not None:
            return self._embedding_matrix.shape[1]
        return None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def set_gene_embeddings(self, matrix: torch.Tensor) -> None:
        """Programmatically set the gene embedding matrix.

        Parameters
        ----------
        matrix
            Tensor of shape ``(len(gene_vocab), embedding_dim)``.
        """
        if matrix.shape[0] != len(self.gene_vocab):
            msg = (
                f"Embedding matrix rows ({matrix.shape[0]}) must match "
                f"gene_vocab size ({len(self.gene_vocab)})"
            )
            raise ValueError(msg)
        self._embedding_matrix = matrix
        self._embedding_dim = matrix.shape[1]

    def tokenize(
        self,
        expression: np.ndarray | torch.Tensor,
        gene_indices: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        """Tokenize a single cell — filter zeros, truncate, optional CLS."""
        expr_t = ensure_tensor(expression, torch.float32)
        genes_t = ensure_tensor(gene_indices, torch.long)

        # Filter zero-expression genes
        nonzero_mask = expr_t > 0
        expr_t = expr_t[nonzero_mask]
        genes_t = genes_t[nonzero_mask]

        # Truncate (leave room for CLS if needed)
        effective_max = self.max_len - (1 if self.prepend_cls else 0)
        expr_t = expr_t[:effective_max]
        genes_t = genes_t[:effective_max]

        # Prepend CLS
        if self.prepend_cls:
            cls_id = torch.tensor([CLS_TOKEN_ID], dtype=torch.long)
            input_ids = torch.cat([cls_id, genes_t])
            values = torch.cat([torch.zeros(1, dtype=torch.float32), expr_t])
            gene_indices_out = torch.cat([cls_id, genes_t])
        else:
            input_ids = genes_t
            values = expr_t
            gene_indices_out = genes_t.clone()

        attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long)

        return TokenizedCell(
            input_ids=input_ids,
            attention_mask=attention_mask,
            values=values,
            gene_indices=gene_indices_out,
            metadata=metadata or {},
        )
