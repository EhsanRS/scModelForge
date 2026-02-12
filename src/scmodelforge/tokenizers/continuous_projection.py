"""TranscriptFormer-style continuous projection tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from scmodelforge._constants import CLS_TOKEN_ID
from scmodelforge.tokenizers._utils import ensure_tensor
from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell
from scmodelforge.tokenizers.registry import register_tokenizer

if TYPE_CHECKING:
    import numpy as np

    from scmodelforge.data.gene_vocab import GeneVocab


@register_tokenizer("continuous_projection")
class ContinuousProjectionTokenizer(BaseTokenizer):
    """TranscriptFormer-style tokenization: fixed gene order with continuous values.

    The simplest tokenizer — packages expression values and gene indices
    without discretization.  Models using this strategy project the
    continuous ``values`` tensor directly into the hidden space.

    Parameters
    ----------
    gene_vocab
        Gene vocabulary.
    max_len
        Maximum sequence length (including CLS).
    prepend_cls
        Whether to prepend a ``[CLS]`` token.
    include_zero_genes
        Whether to include zero-expression genes.
    log_transform
        If ``True``, apply ``log1p`` to expression values at tokenize time.
    """

    def __init__(
        self,
        gene_vocab: GeneVocab,
        max_len: int = 2048,
        prepend_cls: bool = True,
        include_zero_genes: bool = True,
        log_transform: bool = False,
    ) -> None:
        super().__init__(gene_vocab, max_len)
        self.prepend_cls = prepend_cls
        self.include_zero_genes = include_zero_genes
        self.log_transform = log_transform

    @property
    def vocab_size(self) -> int:
        return len(self.gene_vocab)

    @property
    def strategy_name(self) -> str:
        return "continuous_projection"

    def tokenize(
        self,
        expression: np.ndarray | torch.Tensor,
        gene_indices: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        expr_t = ensure_tensor(expression, torch.float32)
        genes_t = ensure_tensor(gene_indices, torch.long)

        # Filter zero genes if requested
        if not self.include_zero_genes:
            nonzero_mask = expr_t > 0
            expr_t = expr_t[nonzero_mask]
            genes_t = genes_t[nonzero_mask]

        # Optional log transform
        if self.log_transform:
            expr_t = torch.log1p(expr_t)

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
