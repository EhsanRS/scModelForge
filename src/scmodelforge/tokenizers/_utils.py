"""Internal utilities for the tokenizers module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from scmodelforge.data.gene_vocab import GeneVocab

logger = logging.getLogger(__name__)


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


def load_gene_embeddings(path: str, gene_vocab: GeneVocab) -> torch.Tensor:
    """Load pretrained gene embeddings and align to a gene vocabulary.

    Supported formats:

    - ``.pt`` / ``.pth``: expects a dict with ``"gene_names"`` (list of str)
      and ``"embeddings"`` (Tensor of shape ``(n, d)``).
    - ``.npy``: expects a NumPy ``.npy`` file saved from a dict with the
      same two keys (use :func:`numpy.save` on the dict).

    Genes present in the file but absent from *gene_vocab* are ignored.
    Genes in *gene_vocab* but missing from the file receive zero vectors.

    Parameters
    ----------
    path
        File path to the embedding file.
    gene_vocab
        Target gene vocabulary.

    Returns
    -------
    torch.Tensor
        Embedding matrix of shape ``(len(gene_vocab), embedding_dim)``.

    Raises
    ------
    ValueError
        If the file extension is unsupported or the data lacks required keys.
    """
    if path.endswith((".pt", ".pth")):
        # torch.load is used here intentionally for loading pretrained gene
        # embedding weights (a trusted local artifact, not user-uploaded data).
        data = torch.load(path, map_location="cpu", weights_only=False)
    elif path.endswith(".npy"):
        # numpy.load with allow_pickle is needed to load dict-style .npy files
        # containing gene embedding data (trusted local artifact).
        data = np.load(path, allow_pickle=True).item()  # noqa: S301
        data["embeddings"] = torch.as_tensor(data["embeddings"], dtype=torch.float32)
    else:
        msg = f"Unsupported embedding file format: {path!r}. Use .pt, .pth, or .npy."
        raise ValueError(msg)

    for key in ("gene_names", "embeddings"):
        if key not in data:
            msg = f"Embedding file must contain key '{key}', got: {sorted(data.keys())}"
            raise ValueError(msg)

    gene_names: list[str] = data["gene_names"]
    embeddings: torch.Tensor = data["embeddings"]
    embedding_dim = embeddings.shape[1]

    # Build aligned matrix
    aligned = torch.zeros(len(gene_vocab), embedding_dim, dtype=torch.float32)
    n_matched = 0
    for i, name in enumerate(gene_names):
        if name in gene_vocab:
            idx = gene_vocab[name]
            aligned[idx] = embeddings[i]
            n_matched += 1

    logger.info(
        "Loaded gene embeddings: %d/%d genes matched (dim=%d)",
        n_matched,
        len(gene_vocab),
        embedding_dim,
    )
    return aligned
