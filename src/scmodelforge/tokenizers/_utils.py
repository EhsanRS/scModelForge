"""Internal utilities for the tokenizers module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np


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
