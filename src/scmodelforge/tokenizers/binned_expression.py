"""scGPT-style binned expression tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from scmodelforge._constants import CLS_TOKEN_ID
from scmodelforge.tokenizers._utils import compute_bin_edges, digitize_expression, ensure_tensor
from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell
from scmodelforge.tokenizers.registry import register_tokenizer

if TYPE_CHECKING:
    import numpy as np

    from scmodelforge.data.gene_vocab import GeneVocab


@register_tokenizer("binned_expression")
class BinnedExpressionTokenizer(BaseTokenizer):
    """scGPT-style tokenization: fixed gene order with discretized expression bins.

    Each gene position gets two parallel representations:
    - ``input_ids`` / ``gene_indices``: gene vocabulary indices (fixed order)
    - ``bin_ids``: discretized expression bin index
    - ``values``: original continuous expression (kept for models that want both)

    Parameters
    ----------
    gene_vocab
        Gene vocabulary.
    max_len
        Maximum sequence length (including CLS).
    n_bins
        Number of expression bins.
    binning_method
        ``"uniform"`` or ``"quantile"``.
    bin_edges
        Pre-computed bin edges. If provided, *n_bins* and *binning_method*
        are ignored.
    value_max
        Upper bound for uniform binning.
    prepend_cls
        Whether to prepend a ``[CLS]`` token.
    include_zero_genes
        Whether to include zero-expression genes.
    """

    def __init__(
        self,
        gene_vocab: GeneVocab,
        max_len: int = 2048,
        n_bins: int = 51,
        binning_method: str = "uniform",
        bin_edges: np.ndarray | None = None,
        value_max: float = 10.0,
        prepend_cls: bool = True,
        include_zero_genes: bool = True,
    ) -> None:
        super().__init__(gene_vocab, max_len)
        self.n_bins = n_bins
        self.binning_method = binning_method
        self.value_max = value_max
        self.prepend_cls = prepend_cls
        self.include_zero_genes = include_zero_genes

        if bin_edges is not None:
            self._bin_edges = bin_edges
        elif binning_method == "uniform":
            self._bin_edges = compute_bin_edges(n_bins=n_bins, method="uniform", value_max=value_max)
        else:
            # Quantile binning requires fit() first
            self._bin_edges: np.ndarray | None = None  # type: ignore[no-redef]

    def fit(self, expression_values: np.ndarray) -> BinnedExpressionTokenizer:
        """Compute bin edges from data (for quantile binning).

        Parameters
        ----------
        expression_values
            1-D array of expression values to compute quantiles from.

        Returns
        -------
        self
        """
        self._bin_edges = compute_bin_edges(
            values=expression_values,
            n_bins=self.n_bins,
            method=self.binning_method,
            value_max=self.value_max,
        )
        return self

    @property
    def bin_edges(self) -> np.ndarray | None:
        """Currently stored bin edges (``None`` if quantile fit not yet called)."""
        return self._bin_edges

    @property
    def n_bin_tokens(self) -> int:
        """Number of bin tokens for model embedding table construction."""
        return self.n_bins

    @property
    def vocab_size(self) -> int:
        return len(self.gene_vocab)

    @property
    def strategy_name(self) -> str:
        return "binned_expression"

    def tokenize(
        self,
        expression: np.ndarray | torch.Tensor,
        gene_indices: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        if self._bin_edges is None:
            raise RuntimeError("Bin edges not set. Call fit() first for quantile binning.")

        expr_t = ensure_tensor(expression, torch.float32)
        genes_t = ensure_tensor(gene_indices, torch.long)

        # Filter zero genes if requested
        if not self.include_zero_genes:
            nonzero_mask = expr_t > 0
            expr_t = expr_t[nonzero_mask]
            genes_t = genes_t[nonzero_mask]

        # Truncate (leave room for CLS if needed)
        effective_max = self.max_len - (1 if self.prepend_cls else 0)
        expr_t = expr_t[:effective_max]
        genes_t = genes_t[:effective_max]

        # Compute bin IDs
        bin_ids = digitize_expression(expr_t, self._bin_edges)

        # Prepend CLS
        if self.prepend_cls:
            cls_id = torch.tensor([CLS_TOKEN_ID], dtype=torch.long)
            input_ids = torch.cat([cls_id, genes_t])
            values = torch.cat([torch.zeros(1, dtype=torch.float32), expr_t])
            bin_ids = torch.cat([torch.zeros(1, dtype=torch.long), bin_ids])
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
            bin_ids=bin_ids,
            gene_indices=gene_indices_out,
            metadata=metadata or {},
        )
