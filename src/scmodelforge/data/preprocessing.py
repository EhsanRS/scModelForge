"""Preprocessing transforms for single-cell expression data.

Provides configurable on-the-fly preprocessing that can be composed
into a pipeline. Operates on per-cell expression vectors (1-D arrays).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Configurable preprocessing pipeline applied to each cell.

    Applies normalisation and log-transformation on-the-fly to raw
    expression vectors. HVG filtering is applied at the dataset level,
    not per-cell.

    Parameters
    ----------
    normalize
        Normalisation method. ``"library_size"`` divides by total counts
        and scales to ``target_sum``. ``None`` skips normalisation.
    target_sum
        Target library size after normalisation.
    log1p
        Whether to apply log1p transformation after normalisation.
    """

    def __init__(
        self,
        normalize: str | None = "library_size",
        target_sum: float | None = 1e4,
        log1p: bool = True,
    ) -> None:
        self.normalize = normalize
        self.target_sum = target_sum
        self.log1p = log1p

    def __call__(self, expression: np.ndarray) -> np.ndarray:
        """Apply preprocessing to a single cell's expression vector.

        Parameters
        ----------
        expression
            Raw expression values, shape ``(n_genes,)``.

        Returns
        -------
        np.ndarray
            Preprocessed expression values, same shape.
        """
        result = expression.astype(np.float32, copy=True)

        if self.normalize == "library_size":
            result = self._normalize_library_size(result)
        elif self.normalize is not None:
            raise ValueError(f"Unknown normalisation method: {self.normalize!r}. Supported: 'library_size', None.")

        if self.log1p:
            result = np.log1p(result)

        return result

    def _normalize_library_size(self, expression: np.ndarray) -> np.ndarray:
        """Normalize to target library size (total counts)."""
        total = expression.sum()
        if total > 0 and self.target_sum is not None:
            expression = expression * (self.target_sum / total)
        return expression

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(normalize={self.normalize!r}, target_sum={self.target_sum}, log1p={self.log1p})"


def select_highly_variable_genes(
    expressions: np.ndarray,
    n_top_genes: int,
) -> np.ndarray:
    """Select highly variable genes by dispersion.

    A simple HVG selection based on variance/mean ratio (Fano factor).
    For more sophisticated methods, use scanpy.pp.highly_variable_genes
    at the AnnData level before constructing the dataset.

    Parameters
    ----------
    expressions
        Expression matrix of shape ``(n_cells, n_genes)``.
    n_top_genes
        Number of genes to select.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n_genes,)`` indicating selected genes.
    """
    mean = np.mean(expressions, axis=0)
    var = np.var(expressions, axis=0)

    # Fano factor (variance / mean), handle zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.where(mean > 0, var / mean, 0.0)

    # Select top genes by dispersion
    n_top_genes = min(n_top_genes, len(dispersion))
    top_indices = np.argsort(dispersion)[-n_top_genes:]
    mask = np.zeros(len(dispersion), dtype=bool)
    mask[top_indices] = True
    return mask
