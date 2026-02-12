"""Lazy AnnData loading and indexing.

Manages one or more .h5ad files, providing a unified view with gene
alignment to a shared vocabulary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np

from scmodelforge.data._utils import get_row_as_dense

if TYPE_CHECKING:
    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.data.ortholog_mapper import OrthologMapper

logger = logging.getLogger(__name__)


class AnnDataStore:
    """Manages one or more AnnData objects with gene alignment.

    Provides a unified interface to access cells by global index across
    multiple datasets, with automatic gene alignment to a shared
    vocabulary.

    Parameters
    ----------
    adatas
        List of AnnData objects or file paths.
    gene_vocab
        Shared gene vocabulary. Each dataset's genes are aligned to
        this vocabulary at construction time.
    obs_keys
        Observation metadata keys to carry through (e.g.,
        ``["cell_type", "batch"]``). Keys not present in a dataset
        are filled with ``"unknown"``.
    ortholog_mapper
        Optional :class:`~scmodelforge.data.ortholog_mapper.OrthologMapper`
        for translating gene names before alignment.
    source_organism
        Organism of the input datasets (e.g. ``"mouse"``). Required
        when *ortholog_mapper* is provided.
    """

    def __init__(
        self,
        adatas: list[ad.AnnData | str | Path],
        gene_vocab: GeneVocab,
        obs_keys: list[str] | None = None,
        ortholog_mapper: OrthologMapper | None = None,
        source_organism: str | None = None,
    ) -> None:
        self.gene_vocab = gene_vocab
        self.obs_keys = obs_keys or []
        self._ortholog_mapper = ortholog_mapper
        self._source_organism = source_organism

        self._adatas: list[ad.AnnData] = []
        self._alignments: list[tuple[np.ndarray, np.ndarray]] = []
        self._cumulative_sizes: list[int] = []

        total = 0
        for adata_or_path in adatas:
            adata = self._load(adata_or_path)
            self._adatas.append(adata)

            # Translate gene names if multi-species mapping is configured
            gene_names = list(adata.var_names)
            if self._ortholog_mapper is not None and self._source_organism is not None:
                gene_names = self._ortholog_mapper.translate_gene_names(
                    gene_names, self._source_organism
                )

            # Compute alignment: which columns in this adata map to which
            # positions in the gene_vocab
            source_idx, vocab_idx = gene_vocab.get_alignment_indices(gene_names)
            self._alignments.append((source_idx, vocab_idx))

            n_overlap = len(source_idx)
            n_genes = adata.n_vars
            logger.info(
                "Loaded %s: %d cells, %d/%d genes overlap with vocabulary",
                getattr(adata_or_path, "name", adata_or_path),
                adata.n_obs,
                n_overlap,
                n_genes,
            )

            total += adata.n_obs
            self._cumulative_sizes.append(total)

    @staticmethod
    def _load(
        adata_or_path: ad.AnnData | str | Path,
        storage_options: dict | None = None,
        cache_dir: str | None = None,
    ) -> ad.AnnData:
        """Load AnnData from a path (local or cloud) or return as-is."""
        if isinstance(adata_or_path, ad.AnnData):
            return adata_or_path
        path_str = str(adata_or_path)
        from scmodelforge.data.cloud import is_cloud_path

        if is_cloud_path(path_str):
            from scmodelforge.data.cloud import read_h5ad as cloud_read_h5ad

            return cloud_read_h5ad(path_str, storage_options=storage_options, cache_dir=cache_dir)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"AnnData file not found: {path}")
        return ad.read_h5ad(path)

    def __len__(self) -> int:
        """Total number of cells across all datasets."""
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def get_cell(self, global_idx: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
        """Retrieve a single cell by global index.

        Returns the expression values and corresponding gene vocabulary
        indices for genes that exist in both the dataset and vocabulary.
        Only non-zero expressed genes that are in the vocabulary are
        returned.

        Parameters
        ----------
        global_idx
            Global cell index (0 to len-1).

        Returns
        -------
        expression
            Non-zero expression values for genes in the vocabulary.
        gene_indices
            Vocabulary indices corresponding to the expression values.
        metadata
            Cell metadata dict with keys from ``obs_keys``.
        """
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(f"Cell index {global_idx} out of range [0, {len(self)})")

        # Find which dataset this cell belongs to
        ds_idx, local_idx = self._resolve_index(global_idx)
        adata = self._adatas[ds_idx]
        source_idx, vocab_idx = self._alignments[ds_idx]

        # Get the full row as dense
        row = get_row_as_dense(adata.X, local_idx)

        # Extract only the genes that are in the vocabulary
        aligned_expr = row[source_idx]
        aligned_genes = vocab_idx

        # Filter to non-zero expressed genes
        nonzero_mask = aligned_expr > 0
        expression = aligned_expr[nonzero_mask].astype(np.float32)
        gene_indices = aligned_genes[nonzero_mask]

        # Collect metadata
        metadata: dict[str, str] = {}
        for key in self.obs_keys:
            if key in adata.obs.columns:
                val = adata.obs.iloc[local_idx][key]
                metadata[key] = str(val)
            else:
                metadata[key] = "unknown"

        return expression, gene_indices, metadata

    def _resolve_index(self, global_idx: int) -> tuple[int, int]:
        """Map a global cell index to (dataset_index, local_index)."""
        for ds_idx, cum_size in enumerate(self._cumulative_sizes):
            if global_idx < cum_size:
                prev = self._cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0
                return ds_idx, global_idx - prev
        raise IndexError(f"Index {global_idx} out of range")

    @property
    def n_datasets(self) -> int:
        """Number of loaded datasets."""
        return len(self._adatas)

    def __repr__(self) -> str:
        return f"AnnDataStore(n_datasets={self.n_datasets}, n_cells={len(self)}, vocab_size={len(self.gene_vocab)})"
