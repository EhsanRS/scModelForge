"""Convert .h5ad files to memory-mapped shard directories.

The shard format pre-aligns genes to a vocabulary at conversion time,
so reading at training time requires no alignment logic and can be
fully memory-mapped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scmodelforge.data._utils import get_row_as_dense

if TYPE_CHECKING:
    import anndata as ad

    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.data.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


def convert_to_shards(
    sources: list[str | Path | ad.AnnData],
    gene_vocab: GeneVocab,
    output_dir: str | Path,
    shard_size: int = 500_000,
    obs_keys: list[str] | None = None,
    preprocessing: PreprocessingPipeline | None = None,
    storage_options: dict | None = None,
) -> Path:
    """Convert .h5ad files to a shard directory of memory-mapped arrays.

    Genes are aligned to *gene_vocab* at conversion time.  Each cell
    stores only its non-zero, vocab-aligned genes, padded to the
    per-shard maximum for rectangular memmap arrays.

    Parameters
    ----------
    sources
        List of .h5ad file paths or AnnData objects.
    gene_vocab
        Gene vocabulary to align to.
    output_dir
        Directory to write shards into.
    shard_size
        Maximum number of cells per shard.
    obs_keys
        Metadata columns to preserve in ``obs.parquet``.
    preprocessing
        Optional preprocessing applied before writing.

    Returns
    -------
    Path
        The *output_dir* path.
    """
    import anndata as _ad
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_keys = obs_keys or []

    # Collect all cells
    all_expressions: list[np.ndarray] = []
    all_gene_indices: list[np.ndarray] = []
    all_obs_rows: list[dict[str, str]] = []

    for source in sources:
        if isinstance(source, _ad.AnnData):
            adata: _ad.AnnData = source
        else:
            from scmodelforge.data.cloud import is_cloud_path
            from scmodelforge.data.cloud import read_h5ad as cloud_read_h5ad

            source_str = str(source)
            if is_cloud_path(source_str):
                adata = cloud_read_h5ad(source_str, storage_options=storage_options)
            else:
                adata = _ad.read_h5ad(Path(source))

        source_idx, vocab_idx = gene_vocab.get_alignment_indices(
            list(adata.var_names)
        )

        logger.info(
            "Processing %s: %d cells, %d/%d genes overlap",
            getattr(source, "name", source),
            adata.n_obs,
            len(source_idx),
            adata.n_vars,
        )

        for i in range(adata.n_obs):
            row = get_row_as_dense(adata.X, i)
            aligned_expr = row[source_idx]
            aligned_genes = vocab_idx

            # Filter to non-zero
            nonzero_mask = aligned_expr > 0
            expression = aligned_expr[nonzero_mask].astype(np.float32)
            gene_indices = aligned_genes[nonzero_mask]

            if preprocessing is not None:
                expression = preprocessing(expression)

            all_expressions.append(expression)
            all_gene_indices.append(gene_indices)

            # Metadata
            meta: dict[str, str] = {}
            for key in obs_keys:
                if key in adata.obs.columns:
                    meta[key] = str(adata.obs.iloc[i][key])
                else:
                    meta[key] = "unknown"
            all_obs_rows.append(meta)

    total_cells = len(all_expressions)
    n_shards = max(1, (total_cells + shard_size - 1) // shard_size)
    shard_sizes: list[int] = []

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, total_cells)
        shard_n = end - start

        shard_exprs = all_expressions[start:end]
        shard_genes = all_gene_indices[start:end]
        shard_obs = all_obs_rows[start:end]

        # Find max gene count in this shard
        n_genes_arr = np.array([len(e) for e in shard_exprs], dtype=np.int64)
        max_genes = int(n_genes_arr.max()) if shard_n > 0 else 0

        # Build padded arrays
        X = np.zeros((shard_n, max_genes), dtype=np.float32)
        G = np.zeros((shard_n, max_genes), dtype=np.int64)

        for i, (expr, genes) in enumerate(zip(shard_exprs, shard_genes, strict=True)):
            n = len(expr)
            X[i, :n] = expr
            G[i, :n] = genes

        # Write shard directory
        shard_path = output_dir / f"shard_{shard_idx:03d}"
        shard_path.mkdir(parents=True, exist_ok=True)

        np.save(shard_path / "X.npy", X)
        np.save(shard_path / "gene_indices.npy", G)
        np.save(shard_path / "n_genes.npy", n_genes_arr)

        # Write obs as parquet
        if obs_keys:
            obs_df = pd.DataFrame(shard_obs)
            obs_df.to_parquet(shard_path / "obs.parquet", index=False)
        else:
            # Write empty parquet
            pd.DataFrame().to_parquet(shard_path / "obs.parquet", index=False)

        # Shard metadata
        shard_meta = {
            "n_cells": shard_n,
            "max_genes": max_genes,
        }
        with open(shard_path / "metadata.json", "w") as f:
            json.dump(shard_meta, f, indent=2)

        shard_sizes.append(shard_n)
        logger.info(
            "Wrote shard %d: %d cells, max_genes=%d",
            shard_idx, shard_n, max_genes,
        )

    # Write manifest
    import hashlib

    vocab_hash = hashlib.md5(
        json.dumps(gene_vocab._gene_to_idx, sort_keys=True).encode()
    ).hexdigest()

    manifest = {
        "n_shards": n_shards,
        "total_cells": total_cells,
        "shard_sizes": shard_sizes,
        "vocab_hash": vocab_hash,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Shard conversion complete: %d shards, %d total cells, dir=%s",
        n_shards, total_cells, output_dir,
    )

    return output_dir


def validate_shard_dir(shard_dir: str | Path) -> bool:
    """Validate a shard directory for completeness.

    Checks that the manifest exists, shard count matches, and all
    required array files are present with correct shapes.

    Parameters
    ----------
    shard_dir
        Path to the shard directory.

    Returns
    -------
    bool
        ``True`` if the shard directory is valid.
    """
    shard_dir = Path(shard_dir)
    manifest_path = shard_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error("Missing manifest.json in %s", shard_dir)
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_shards = manifest.get("n_shards", 0)
    shard_sizes = manifest.get("shard_sizes", [])

    if len(shard_sizes) != n_shards:
        logger.error("Shard count mismatch: expected %d, got %d sizes", n_shards, len(shard_sizes))
        return False

    for shard_idx in range(n_shards):
        shard_path = shard_dir / f"shard_{shard_idx:03d}"
        if not shard_path.is_dir():
            logger.error("Missing shard directory: %s", shard_path)
            return False

        for name in ("X.npy", "gene_indices.npy", "n_genes.npy"):
            if not (shard_path / name).exists():
                logger.error("Missing %s in %s", name, shard_path)
                return False

        # Check shapes
        n_genes = np.load(shard_path / "n_genes.npy", mmap_mode="r")
        expected_n = shard_sizes[shard_idx]
        if len(n_genes) != expected_n:
            logger.error(
                "Shard %d: expected %d cells, got %d",
                shard_idx, expected_n, len(n_genes),
            )
            return False

    total = manifest.get("total_cells", 0)
    if sum(shard_sizes) != total:
        logger.error("Total cells mismatch: manifest=%d, sum=%d", total, sum(shard_sizes))
        return False

    return True
