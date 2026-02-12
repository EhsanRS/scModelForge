"""Batch-level gene selection collator."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from scmodelforge.tokenizers.base import BaseTokenizer
    from scmodelforge.tokenizers.masking import MaskingStrategy


class GeneSelectionCollator:
    """Batch-level gene selection + tokenization + masking collator.

    When ``strategy="all"`` (default), tokenises and collates identically
    to ``TokenizedCellDataset`` + ``tokenizer._collate``.

    When ``strategy="most_expressed"`` or ``"random_expressed"``, selects
    a subset of genes at the batch level *before* tokenisation.

    Parameters
    ----------
    tokenizer
        Tokenizer to convert raw cell data into model inputs.
    masking
        Optional masking strategy applied after tokenisation.
    strategy
        Gene selection strategy: ``"all"``, ``"most_expressed"``, or
        ``"random_expressed"``.
    n_genes
        Number of genes to keep per batch. Required when
        ``strategy != "all"``.
    """

    _VALID_STRATEGIES = {"all", "most_expressed", "random_expressed"}

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        masking: MaskingStrategy | None = None,
        strategy: str = "all",
        n_genes: int | None = None,
    ) -> None:
        if strategy not in self._VALID_STRATEGIES:
            msg = f"Unknown gene selection strategy: {strategy!r}. Must be one of {self._VALID_STRATEGIES}."
            raise ValueError(msg)

        if strategy != "all" and n_genes is None:
            msg = "n_genes is required when strategy != 'all'"
            raise ValueError(msg)

        self.tokenizer = tokenizer
        self.masking = masking
        self.strategy = strategy
        self.n_genes = n_genes

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of raw CellDataset dicts.

        Parameters
        ----------
        batch
            List of dicts from ``CellDataset.__getitem__`` with keys
            ``expression``, ``gene_indices``, ``n_genes``, ``metadata``.

        Returns
        -------
        dict[str, torch.Tensor]
            Collated batch ready for model input.
        """
        if not batch:
            return {}

        # 1. Gene selection
        if self.strategy == "most_expressed":
            selected = self._select_genes_most_expressed(batch)
            batch = [self._filter_cell(item, selected) for item in batch]
        elif self.strategy == "random_expressed":
            selected = self._select_genes_random(batch)
            batch = [self._filter_cell(item, selected) for item in batch]

        # 2. Tokenize each cell
        cells = []
        for item in batch:
            cell = self.tokenizer.tokenize(
                expression=item["expression"],
                gene_indices=item["gene_indices"],
                metadata=item.get("metadata"),
            )
            if self.masking is not None:
                cell = self.masking.apply(cell)
            cells.append(cell)

        # 3. Collate
        return self.tokenizer._collate(cells)

    def _select_genes_most_expressed(self, batch: list[dict[str, Any]]) -> set[int]:
        """Select genes with the highest total expression across the batch.

        Returns a set of gene vocabulary indices to keep.
        """
        totals: dict[int, float] = {}
        for item in batch:
            gene_indices = item["gene_indices"]
            expression = item["expression"]
            for gi, expr in zip(
                gene_indices.tolist() if isinstance(gene_indices, torch.Tensor) else gene_indices,
                expression.tolist() if isinstance(expression, torch.Tensor) else expression,
                strict=True,
            ):
                totals[gi] = totals.get(gi, 0.0) + expr

        # Take top n_genes by total expression
        n = self.n_genes  # type: ignore[assignment]
        if len(totals) <= n:
            return set(totals.keys())

        sorted_genes = sorted(totals, key=totals.__getitem__, reverse=True)
        return set(sorted_genes[:n])

    def _select_genes_random(self, batch: list[dict[str, Any]]) -> set[int]:
        """Randomly sample expressed genes from the batch.

        Returns a set of gene vocabulary indices.
        """
        all_genes: set[int] = set()
        for item in batch:
            gene_indices = item["gene_indices"]
            if isinstance(gene_indices, torch.Tensor):
                all_genes.update(gene_indices.tolist())
            else:
                all_genes.update(int(g) for g in gene_indices)

        n = self.n_genes  # type: ignore[assignment]
        if len(all_genes) <= n:
            return all_genes

        return set(random.sample(sorted(all_genes), n))

    @staticmethod
    def _filter_cell(item: dict[str, Any], selected_genes: set[int]) -> dict[str, Any]:
        """Keep only genes in *selected_genes*.

        Returns a new dict with filtered ``expression``, ``gene_indices``,
        and updated ``n_genes``.
        """
        gene_indices = item["gene_indices"]
        expression = item["expression"]

        if isinstance(gene_indices, torch.Tensor):
            mask = torch.tensor([int(g) in selected_genes for g in gene_indices.tolist()], dtype=torch.bool)
            filtered_genes = gene_indices[mask]
            filtered_expr = expression[mask]
        else:
            import numpy as np

            mask = np.array([int(g) in selected_genes for g in gene_indices], dtype=bool)
            filtered_genes = gene_indices[mask]
            filtered_expr = expression[mask]

        return {
            "expression": filtered_expr,
            "gene_indices": filtered_genes,
            "n_genes": len(filtered_expr),
            "metadata": item.get("metadata", {}),
        }
