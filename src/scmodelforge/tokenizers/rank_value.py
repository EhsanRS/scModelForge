"""Geneformer-style rank-value tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from scmodelforge._constants import CLS_TOKEN_ID
from scmodelforge.tokenizers._utils import ensure_tensor, rank_genes_by_expression
from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell
from scmodelforge.tokenizers.registry import register_tokenizer

if TYPE_CHECKING:
    import numpy as np

    from scmodelforge.data.gene_vocab import GeneVocab


@register_tokenizer("rank_value")
class RankValueTokenizer(BaseTokenizer):
    """Geneformer-style tokenization: rank genes by expression.

    Steps
    -----
    1. Filter to non-zero expressed genes.
    2. Rank genes by expression value (descending, stable).
    3. Truncate to ``max_len - 1`` (reserving space for CLS if enabled).
    4. Optionally prepend a CLS token.
    5. Build ``input_ids`` (gene vocab indices), ``attention_mask``,
       ``values`` (expression values), and ``gene_indices``.

    Parameters
    ----------
    gene_vocab
        Gene vocabulary.
    max_len
        Maximum sequence length (including CLS).
    prepend_cls
        Whether to prepend a ``[CLS]`` token.
    """

    def __init__(
        self,
        gene_vocab: GeneVocab,
        max_len: int = 2048,
        prepend_cls: bool = True,
    ) -> None:
        super().__init__(gene_vocab, max_len)
        self.prepend_cls = prepend_cls

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.gene_vocab)

    @property
    def strategy_name(self) -> str:
        return "rank_value"

    def tokenize(
        self,
        expression: np.ndarray | torch.Tensor,
        gene_indices: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        expr_t = ensure_tensor(expression, torch.float32)
        genes_t = ensure_tensor(gene_indices, torch.long)

        # Rank non-zero genes by expression (descending)
        ranked_genes, ranked_values = rank_genes_by_expression(expr_t, genes_t)

        # Truncate (leave room for CLS if needed)
        effective_max = self.max_len - (1 if self.prepend_cls else 0)
        ranked_genes = ranked_genes[:effective_max]
        ranked_values = ranked_values[:effective_max]

        # Prepend CLS
        if self.prepend_cls:
            cls_id = torch.tensor([CLS_TOKEN_ID], dtype=torch.long)
            input_ids = torch.cat([cls_id, ranked_genes])
            values = torch.cat([torch.zeros(1, dtype=torch.float32), ranked_values])
        else:
            input_ids = ranked_genes
            values = ranked_values

        attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long)

        return TokenizedCell(
            input_ids=input_ids,
            attention_mask=attention_mask,
            values=values,
            gene_indices=input_ids.clone(),
            metadata=metadata or {},
        )
