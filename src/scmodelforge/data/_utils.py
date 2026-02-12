"""Internal utilities for the data module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch

if TYPE_CHECKING:
    import anndata as ad

    from scmodelforge.config.schema import DataConfig

logger = logging.getLogger(__name__)


def sparse_to_dense(x: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Convert a sparse matrix or array to dense ndarray.

    Parameters
    ----------
    x
        Sparse matrix (CSR, CSC, COO) or dense ndarray.

    Returns
    -------
    np.ndarray
        Dense array.
    """
    if sp.issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def get_row_as_dense(X: sp.spmatrix | np.ndarray, idx: int) -> np.ndarray:
    """Extract a single row from a matrix as a 1-D dense array.

    Handles both sparse and dense matrices efficiently.

    Parameters
    ----------
    X
        Expression matrix (n_cells x n_genes), sparse or dense.
    idx
        Row index.

    Returns
    -------
    np.ndarray
        1-D dense array of shape (n_genes,).
    """
    row = X[idx]
    if sp.issparse(row):
        return np.asarray(row.todense()).ravel()
    return np.asarray(row).ravel()


def get_nonzero_indices_and_values(
    expression: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get indices and values of non-zero entries in an expression vector.

    Parameters
    ----------
    expression
        1-D expression vector.

    Returns
    -------
    indices
        Indices of non-zero elements.
    values
        Non-zero expression values.
    """
    mask = expression > 0
    return np.where(mask)[0], expression[mask]


def collate_cells(batch: list[dict[str, Any]], pad_value: int = 0) -> dict[str, Any]:
    """Custom collate function for variable-length cell data.

    Pads expression and gene_indices tensors to the maximum sequence
    length in the batch. Builds an attention mask (1 = real, 0 = pad).

    Parameters
    ----------
    batch
        List of dicts from CellDataset.__getitem__, each with keys
        ``expression``, ``gene_indices``, ``n_genes``, and optionally
        ``metadata``.
    pad_value
        Value used to pad shorter sequences (default 0 = PAD token).

    Returns
    -------
    dict
        Batched tensors with keys: ``expression``, ``gene_indices``,
        ``attention_mask``, ``n_genes``, ``metadata``.
    """
    max_len = max(item["n_genes"] for item in batch)

    expressions = []
    gene_indices = []
    attention_masks = []
    n_genes_list = []
    metadata_list = []

    for item in batch:
        n = item["n_genes"]
        expr = item["expression"]
        genes = item["gene_indices"]

        # Pad to max_len
        pad_len = max_len - n
        if pad_len > 0:
            expr = torch.cat([expr, torch.zeros(pad_len, dtype=expr.dtype)])
            genes = torch.cat([genes, torch.full((pad_len,), pad_value, dtype=genes.dtype)])

        mask = torch.zeros(max_len, dtype=torch.long)
        mask[:n] = 1

        expressions.append(expr)
        gene_indices.append(genes)
        attention_masks.append(mask)
        n_genes_list.append(item["n_genes"])
        metadata_list.append(item.get("metadata", {}))

    return {
        "expression": torch.stack(expressions),
        "gene_indices": torch.stack(gene_indices),
        "attention_mask": torch.stack(attention_masks),
        "n_genes": torch.tensor(n_genes_list, dtype=torch.long),
        "metadata": metadata_list,
    }


def load_adata(
    data_config: DataConfig,
    adata: Any | None = None,
    obs_keys: list[str] | None = None,
) -> ad.AnnData:
    """Load AnnData from the configured source.

    Dispatches on ``data_config.source``:

    * ``"local"`` — reads ``.h5ad`` files from ``data_config.paths``
    * ``"cellxgene_census"`` — queries CELLxGENE Census via
      :func:`~scmodelforge.data.census.load_census_adata`

    Parameters
    ----------
    data_config
        Data configuration (source, paths, census settings, etc.).
    adata
        Optional pre-loaded AnnData.  If provided, returned directly
        (useful for testing / programmatic use).
    obs_keys
        Extra ``obs`` column names forwarded to Census loading.

    Returns
    -------
    anndata.AnnData

    Raises
    ------
    ValueError
        If *source* is not recognized.
    """
    if adata is not None:
        return adata

    if data_config.source == "cellxgene_census":
        from scmodelforge.data.census import load_census_adata

        return load_census_adata(data_config.census, obs_keys=obs_keys)

    if data_config.source == "local":
        import anndata as ad

        from scmodelforge.data.cloud import is_cloud_path
        from scmodelforge.data.cloud import read_h5ad as cloud_read_h5ad

        cloud_cfg = data_config.cloud
        adata_list = []
        for p in data_config.paths:
            if is_cloud_path(p):
                adata_list.append(cloud_read_h5ad(
                    p,
                    storage_options=cloud_cfg.storage_options or None,
                    cache_dir=cloud_cfg.cache_dir,
                ))
            else:
                adata_list.append(ad.read_h5ad(p))
        if len(adata_list) == 1:
            return adata_list[0]
        return ad.concat(adata_list)

    msg = f"Unknown data source: {data_config.source!r}. Expected 'local' or 'cellxgene_census'."
    raise ValueError(msg)
