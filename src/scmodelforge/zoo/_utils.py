"""Shared utilities for external model adapters."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

    from scmodelforge.zoo.base import GeneOverlapReport

logger = logging.getLogger(__name__)


def require_package(package_name: str, pip_name: str | None = None) -> None:
    """Check that *package_name* is importable; raise a clear error if not.

    Parameters
    ----------
    package_name
        Python import name (e.g. ``"geneformer"``).
    pip_name
        pip install name if different from *package_name*
        (e.g. ``"transformers>=4.30"``).

    Raises
    ------
    ImportError
        With install instructions.
    """
    import importlib

    try:
        importlib.import_module(package_name)
    except ImportError as exc:
        install = pip_name or package_name
        msg = f"'{package_name}' is required but not installed. Install with: pip install '{install}'"
        raise ImportError(msg) from exc


def compute_gene_overlap(
    adata_genes: list[str],
    model_genes: list[str],
) -> GeneOverlapReport:
    """Compute overlap statistics between two gene sets.

    Parameters
    ----------
    adata_genes
        Gene identifiers from the AnnData object.
    model_genes
        Gene identifiers from the model's vocabulary.

    Returns
    -------
    GeneOverlapReport
    """
    from scmodelforge.zoo.base import GeneOverlapReport

    adata_set = set(adata_genes)
    model_set = set(model_genes)

    matched = adata_set & model_set
    missing = model_set - adata_set
    extra = adata_set - model_set

    model_vocab_size = len(model_set)
    coverage = len(matched) / model_vocab_size if model_vocab_size > 0 else 0.0

    return GeneOverlapReport(
        matched=len(matched),
        missing=len(missing),
        extra=len(extra),
        coverage=coverage,
        model_vocab_size=model_vocab_size,
        adata_n_genes=len(adata_set),
    )


def align_genes_to_model(
    adata: AnnData,
    model_genes: list[str],
    *,
    fill_missing: float = 0.0,
) -> AnnData:
    """Reindex *adata* columns to match a model's gene vocabulary.

    Sparse-friendly: uses sparse slicing and ``scipy.sparse.hstack`` for
    missing genes.  Never materializes a full dense matrix.

    Parameters
    ----------
    adata
        Input AnnData object.
    model_genes
        Ordered list of gene identifiers expected by the model.
    fill_missing
        Fill value for genes missing from *adata* (default 0.0).

    Returns
    -------
    AnnData
        New AnnData with columns matching *model_genes* in order.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    adata_gene_set = set(adata.var_names)
    model_gene_set = set(model_genes)
    overlap = adata_gene_set & model_gene_set

    n_matched = len(overlap)
    n_missing = len(model_gene_set) - n_matched
    n_extra = len(adata_gene_set) - n_matched

    logger.info(
        "Gene alignment: %d matched, %d missing from data, %d extra in data (%.1f%% coverage)",
        n_matched,
        n_missing,
        n_extra,
        n_matched / len(model_gene_set) * 100 if model_gene_set else 0.0,
    )

    # Build column indices for model gene ordering
    n_cells = adata.n_obs
    columns = []
    adata_var_index = {g: i for i, g in enumerate(adata.var_names)}

    X = adata.X
    is_sparse = sp.issparse(X)

    for gene in model_genes:
        if gene in adata_var_index:
            idx = adata_var_index[gene]
            col = X[:, idx]
            if is_sparse:
                col = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
            else:
                col = np.asarray(col).ravel()
            columns.append(col)
        else:
            columns.append(np.full(n_cells, fill_missing, dtype=np.float32))

    # Stack into dense or sparse matrix
    new_X = np.column_stack(columns) if columns else np.empty((n_cells, 0), dtype=np.float32)
    new_X_sparse = sp.csr_matrix(new_X)

    var = pd.DataFrame(index=model_genes)
    new_adata = ad.AnnData(X=new_X_sparse, obs=adata.obs.copy(), var=var)

    return new_adata


def ensure_raw_counts(adata: AnnData) -> AnnData:
    """Return an AnnData with raw counts in ``.X``.

    Uses ``adata.raw`` if available, otherwise returns *adata* unchanged
    (assumes ``.X`` already contains raw counts).

    Parameters
    ----------
    adata
        Input AnnData object.

    Returns
    -------
    AnnData
        AnnData with raw counts in ``.X``.
    """
    if adata.raw is not None:
        logger.info("Using adata.raw for raw counts (%d genes)", adata.raw.n_vars)
        return adata.raw.to_adata()
    logger.info("No adata.raw found; assuming adata.X contains raw counts")
    return adata
