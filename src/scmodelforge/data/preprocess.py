"""Cached offline preprocessing for AnnData files.

Applies scanpy-level preprocessing (normalization, log1p, HVG selection)
to an entire .h5ad file and writes the result to disk. This avoids
repeated on-the-fly preprocessing and enables batch-level operations
like highly variable gene selection that require the full matrix.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def preprocess_h5ad(
    input_path: str,
    output_path: str,
    normalize: str | None = "library_size",
    target_sum: float | None = 1e4,
    log1p: bool = True,
    hvg_n_top_genes: int | None = None,
) -> str:
    """Preprocess an .h5ad file and write the result to disk.

    Applies scanpy preprocessing at the AnnData level:
    library-size normalization, log1p, and (optionally) highly variable
    gene filtering.  This is a one-shot offline operation — the output
    file can then be used directly for training or fine-tuning without
    on-the-fly preprocessing.

    Parameters
    ----------
    input_path
        Path to the input .h5ad file. Supports cloud URLs
        (``s3://``, ``gs://``, etc.) if fsspec is installed.
    output_path
        Path to write the preprocessed .h5ad file.
    normalize
        Normalization method. ``"library_size"`` applies
        :func:`scanpy.pp.normalize_total`. ``None`` skips normalization.
    target_sum
        Target sum for library-size normalization.
    log1p
        Whether to apply :func:`scanpy.pp.log1p`.
    hvg_n_top_genes
        If set, select the top *N* highly variable genes via
        :func:`scanpy.pp.highly_variable_genes` and subset the data.

    Returns
    -------
    str
        The *output_path* for convenience (e.g. chaining).
    """
    import scanpy as sc

    from scmodelforge.data.cloud import is_cloud_path
    from scmodelforge.data.cloud import read_h5ad as cloud_read_h5ad

    # --- Load ---
    if is_cloud_path(input_path):
        adata = cloud_read_h5ad(input_path)
    else:
        import anndata as ad

        adata = ad.read_h5ad(input_path)

    logger.info("Loaded %d cells x %d genes from %s", adata.n_obs, adata.n_vars, input_path)

    # --- Normalize ---
    if normalize == "library_size":
        sc.pp.normalize_total(adata, target_sum=target_sum)
        logger.info("Normalized to target_sum=%s", target_sum)
    elif normalize is not None:
        msg = f"Unknown normalization method: {normalize!r}. Supported: 'library_size', None."
        raise ValueError(msg)

    # --- Log1p ---
    if log1p:
        sc.pp.log1p(adata)
        logger.info("Applied log1p transformation")

    # --- HVG selection ---
    if hvg_n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg_n_top_genes)
        n_before = adata.n_vars
        adata = adata[:, adata.var["highly_variable"]].copy()
        logger.info("HVG selection: %d -> %d genes", n_before, adata.n_vars)

    # --- Write ---
    adata.write_h5ad(output_path)
    logger.info("Wrote preprocessed data to %s", output_path)

    return output_path
