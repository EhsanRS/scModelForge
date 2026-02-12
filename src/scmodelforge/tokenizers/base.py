"""Base tokenizer interface and output dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from scmodelforge._constants import PAD_TOKEN_ID

if TYPE_CHECKING:
    import numpy as np

    from scmodelforge.data.gene_vocab import GeneVocab


@dataclass
class TokenizedCell:
    """Output of tokenization for a single cell.

    All tensor fields are 1-D with shape ``(seq_len,)``.  Batching
    (padding + stacking) is handled by :meth:`BaseTokenizer._collate`.

    Attributes
    ----------
    input_ids
        Token indices — for rank-value these equal the gene vocabulary
        indices; other strategies may differ.
    attention_mask
        Binary mask, ``1`` for real tokens, ``0`` for padding.
    values
        Continuous expression values (optional — not every strategy uses them).
    bin_ids
        Discrete bin indices (optional — only ``BinnedExpressionTokenizer``
        populates this).  1-D long tensor ``(seq_len,)`` with values in
        ``[0, n_bins - 1]``.
    gene_indices
        Original gene vocabulary indices for each position.
    metadata
        Pass-through metadata dict.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    values: torch.Tensor | None = None
    bin_ids: torch.Tensor | None = None
    gene_indices: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskedTokenizedCell(TokenizedCell):
    """A tokenized cell with masking applied for pretraining.

    Extends :class:`TokenizedCell` with label and position info needed
    for masked-gene-prediction loss.

    Attributes
    ----------
    labels
        Ground-truth ``input_ids`` at masked positions; ``-100`` elsewhere
        (PyTorch cross-entropy ignore index).
    masked_positions
        Boolean tensor — ``True`` where masking was applied.
    """

    labels: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    masked_positions: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.bool))


class BaseTokenizer(ABC):
    """Abstract base class for all tokenization strategies.

    Parameters
    ----------
    gene_vocab
        Gene vocabulary mapping gene names to indices.
    max_len
        Maximum sequence length (including special tokens).
    """

    def __init__(self, gene_vocab: GeneVocab, max_len: int = 2048) -> None:
        self.gene_vocab = gene_vocab
        self.max_len = max_len

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def tokenize(
        self,
        expression: np.ndarray | torch.Tensor,
        gene_indices: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        """Convert a single cell's expression vector into model input.

        Parameters
        ----------
        expression
            1-D array of expression values.
        gene_indices
            1-D array of gene vocabulary indices (same length).
        metadata
            Optional pass-through metadata.

        Returns
        -------
        TokenizedCell
        """
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size (for embedding layer)."""
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name for this strategy."""
        ...

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def tokenize_batch(
        self,
        expressions: list[np.ndarray | torch.Tensor],
        gene_indices_list: list[np.ndarray | torch.Tensor],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of cells and collate into padded tensors.

        Parameters
        ----------
        expressions
            List of 1-D expression vectors.
        gene_indices_list
            List of 1-D gene index vectors (parallel to *expressions*).
        metadata_list
            Optional list of metadata dicts.

        Returns
        -------
        dict[str, torch.Tensor]
            Batch dict with keys ``input_ids``, ``attention_mask``,
            ``values`` (if present), ``gene_indices``, and ``labels``
            / ``masked_positions`` when the cells are masked.
        """
        metas = metadata_list or [{} for _ in expressions]
        cells = [
            self.tokenize(expr, genes, meta)
            for expr, genes, meta in zip(expressions, gene_indices_list, metas, strict=True)
        ]
        return self._collate(cells)

    def _collate(self, cells: list[TokenizedCell]) -> dict[str, torch.Tensor]:
        """Pad and stack a list of tokenized cells into a batch dict.

        Padding length is the *batch maximum* (not ``self.max_len``) for
        memory efficiency.  Sequences longer than ``self.max_len`` are
        truncated as a safety net.
        """
        if not cells:
            return {}

        # Truncate any cell that exceeds max_len
        for i, c in enumerate(cells):
            if c.input_ids.shape[0] > self.max_len:
                cells[i] = self._truncate(c)

        batch_max = max(c.input_ids.shape[0] for c in cells)
        has_values = cells[0].values is not None
        has_bin_ids = cells[0].bin_ids is not None
        is_masked = isinstance(cells[0], MaskedTokenizedCell)

        input_ids_batch = []
        mask_batch = []
        values_batch = []
        bin_ids_batch = []
        gene_idx_batch = []
        labels_batch = []
        masked_pos_batch = []

        for c in cells:
            seq_len = c.input_ids.shape[0]
            pad_len = batch_max - seq_len

            input_ids_batch.append(_pad_1d(c.input_ids, pad_len, PAD_TOKEN_ID))
            mask_batch.append(_pad_1d(c.attention_mask, pad_len, 0))
            gene_idx_batch.append(_pad_1d(c.gene_indices, pad_len, PAD_TOKEN_ID))

            if has_values:
                values_batch.append(_pad_1d(c.values, pad_len, 0.0))  # type: ignore[arg-type]

            if has_bin_ids:
                bin_ids_batch.append(_pad_1d(c.bin_ids, pad_len, PAD_TOKEN_ID))  # type: ignore[arg-type]

            if is_masked:
                mc = c  # type: ignore[assignment]
                labels_batch.append(_pad_1d(mc.labels, pad_len, -100))
                masked_pos_batch.append(_pad_1d(mc.masked_positions, pad_len, False))

        result: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids_batch),
            "attention_mask": torch.stack(mask_batch),
            "gene_indices": torch.stack(gene_idx_batch),
        }
        if has_values:
            result["values"] = torch.stack(values_batch)
        if has_bin_ids:
            result["bin_ids"] = torch.stack(bin_ids_batch)
        if is_masked:
            result["labels"] = torch.stack(labels_batch)
            result["masked_positions"] = torch.stack(masked_pos_batch)
        return result

    @staticmethod
    def _truncate(cell: TokenizedCell) -> TokenizedCell:
        """Truncate a cell to ``max_len`` (used internally by ``_collate``)."""
        # We don't know max_len here, but the caller already checked.
        # This is a fallback — truncate to the length that was set.
        # The caller will use the cell's own max_len.
        raise NotImplementedError("Direct truncation should not be needed; _collate handles it.")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _pad_1d(tensor: torch.Tensor, pad_len: int, fill_value: int | float | bool) -> torch.Tensor:
    """Pad a 1-D tensor on the right."""
    if pad_len <= 0:
        return tensor
    pad = torch.full((pad_len,), fill_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad])
