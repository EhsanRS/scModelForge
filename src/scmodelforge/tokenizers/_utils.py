"""Internal utilities for the tokenizers module."""

from __future__ import annotations

import numpy as np
import torch


def ensure_tensor(x: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a numpy array or tensor to a torch tensor with the given dtype.

    Parameters
    ----------
    x
        Input array or tensor.
    dtype
        Desired torch dtype.

    Returns
    -------
    torch.Tensor
        Tensor with the specified dtype.
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(x, dtype=dtype)


def rank_genes_by_expression(
    expression: torch.Tensor,
    gene_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank genes by expression value in descending order.

    Filters out zero-expression genes, then sorts the remaining genes
    from highest to lowest expression. Uses stable sort so that tied
    values maintain their original relative order.

    Parameters
    ----------
    expression
        1-D float tensor of expression values.
    gene_indices
        1-D int tensor of gene vocabulary indices (same length as expression).

    Returns
    -------
    ranked_genes
        Gene indices sorted by descending expression.
    ranked_values
        Corresponding expression values in descending order.
    """
    # Filter non-zero
    nonzero_mask = expression > 0
    expr_nz = expression[nonzero_mask]
    genes_nz = gene_indices[nonzero_mask]

    if expr_nz.numel() == 0:
        return genes_nz, expr_nz

    # Sort descending (negate for argsort, stable for reproducibility)
    order = torch.argsort(-expr_nz, stable=True)
    return genes_nz[order], expr_nz[order]


def compute_bin_edges(
    values: np.ndarray | None = None,
    n_bins: int = 51,
    method: str = "uniform",
    value_max: float = 10.0,
) -> np.ndarray:
    """Compute bin edges for expression discretization.

    Parameters
    ----------
    values
        1-D array of expression values (required for ``"quantile"`` method).
    n_bins
        Number of bins.
    method
        ``"uniform"`` for evenly spaced edges or ``"quantile"`` for
        data-driven quantile edges.
    value_max
        Upper bound for uniform binning.

    Returns
    -------
    np.ndarray
        Bin edges of shape ``(n_bins + 1,)``.

    Raises
    ------
    ValueError
        If *method* is unknown or *values* is missing for quantile binning.
    """
    if method == "uniform":
        return np.linspace(0.0, value_max, n_bins + 1)

    if method == "quantile":
        if values is None:
            raise ValueError("values are required for quantile binning")
        nonzero = values[values > 0]
        if nonzero.size == 0:
            return np.linspace(0.0, value_max, n_bins + 1)
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(nonzero, quantiles)
        # Deduplicate while preserving order
        edges = np.unique(edges)
        return edges

    raise ValueError(f"Unknown binning method '{method}'. Choose 'uniform' or 'quantile'.")


def digitize_expression(values: torch.Tensor, bin_edges: np.ndarray) -> torch.Tensor:
    """Map continuous expression values to discrete bin indices.

    Zero values are always mapped to bin 0. Non-zero values are
    assigned to bins ``[1, n_bins - 1]`` via :func:`torch.bucketize`.

    Parameters
    ----------
    values
        1-D float tensor of expression values.
    bin_edges
        Bin edges from :func:`compute_bin_edges`.

    Returns
    -------
    torch.Tensor
        1-D long tensor of bin indices in ``[0, n_bins - 1]``.
    """
    n_bins = len(bin_edges) - 1
    edges_tensor = torch.as_tensor(bin_edges[1:-1], dtype=values.dtype)
    # bucketize returns indices in [0, len(edges_tensor)]
    bin_ids = torch.bucketize(values, edges_tensor, right=False)
    # Clamp to valid range
    bin_ids = bin_ids.clamp(0, n_bins - 1)
    # Zero expression always maps to bin 0
    bin_ids[values == 0] = 0
    return bin_ids.long()
