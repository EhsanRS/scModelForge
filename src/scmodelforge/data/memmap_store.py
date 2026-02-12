"""Memory-mapped shard store for large-scale single-cell data.

Provides :class:`MemoryMappedStore` which reads pre-aligned shard
directories of numpy memory-mapped arrays, giving the same
``get_cell()`` interface as :class:`AnnDataStore` without loading
all data into memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyarrow.parquet as pq

    from scmodelforge.data.gene_vocab import GeneVocab

logger = logging.getLogger(__name__)


class MemoryMappedStore:
    """Read-only store backed by memory-mapped numpy shards.

    Each shard directory contains:
    - ``X.npy`` — expression values ``(n_cells, n_aligned_genes)``
    - ``gene_indices.npy`` — vocab indices ``(n_cells, n_aligned_genes)``
    - ``n_genes.npy`` — actual gene count per cell ``(n_cells,)``
    - ``obs.parquet`` — cell metadata
    - ``metadata.json`` — shard-level info

    A top-level ``manifest.json`` indexes all shards.

    Parameters
    ----------
    shard_dir
        Path to the shard directory containing ``manifest.json``.
    gene_vocab
        Optional gene vocabulary (for validation).
    """

    def __init__(
        self,
        shard_dir: str | Path,
        gene_vocab: GeneVocab | None = None,
    ) -> None:
        self._shard_dir = Path(shard_dir)
        self.gene_vocab = gene_vocab

        manifest_path = self._shard_dir / "manifest.json"
        if not manifest_path.exists():
            msg = f"Manifest not found: {manifest_path}"
            raise FileNotFoundError(msg)

        with open(manifest_path) as f:
            self._manifest = json.load(f)

        self._n_shards: int = self._manifest["n_shards"]
        self._shard_sizes: list[int] = self._manifest["shard_sizes"]
        self._total_cells: int = self._manifest["total_cells"]

        # Build cumulative sizes for index resolution
        self._cumulative_sizes: list[int] = []
        total = 0
        for s in self._shard_sizes:
            total += s
            self._cumulative_sizes.append(total)

        # Lazy-loaded memmaps per shard
        self._X_maps: dict[int, np.ndarray] = {}
        self._gene_idx_maps: dict[int, np.ndarray] = {}
        self._n_genes_maps: dict[int, np.ndarray] = {}
        self._obs_tables: dict[int, pq.ParquetFile] = {}

        logger.info(
            "Opened MemoryMappedStore: %d shards, %d cells, dir=%s",
            self._n_shards,
            self._total_cells,
            self._shard_dir,
        )

    def _get_shard_memmaps(
        self, shard_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lazily open memmaps for a specific shard."""
        if shard_idx not in self._X_maps:
            shard_path = self._shard_dir / f"shard_{shard_idx:03d}"
            self._X_maps[shard_idx] = np.load(
                shard_path / "X.npy", mmap_mode="r"
            )
            self._gene_idx_maps[shard_idx] = np.load(
                shard_path / "gene_indices.npy", mmap_mode="r"
            )
            self._n_genes_maps[shard_idx] = np.load(
                shard_path / "n_genes.npy", mmap_mode="r"
            )
        return (
            self._X_maps[shard_idx],
            self._gene_idx_maps[shard_idx],
            self._n_genes_maps[shard_idx],
        )

    def _get_obs_metadata(self, shard_idx: int, local_idx: int) -> dict[str, str]:
        """Read obs metadata for a single cell from parquet."""
        import pandas as pd

        if shard_idx not in self._obs_tables:
            shard_path = self._shard_dir / f"shard_{shard_idx:03d}"
            obs_path = shard_path / "obs.parquet"
            if obs_path.exists():
                self._obs_tables[shard_idx] = pd.read_parquet(obs_path)
            else:
                self._obs_tables[shard_idx] = pd.DataFrame()

        df = self._obs_tables[shard_idx]
        if len(df) == 0 or local_idx >= len(df):
            return {}
        row = df.iloc[local_idx]
        return {k: str(v) for k, v in row.items()}

    def __len__(self) -> int:
        """Total number of cells across all shards."""
        return self._total_cells

    def get_cell(
        self, global_idx: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
        """Retrieve a single cell by global index.

        Returns the same format as :meth:`AnnDataStore.get_cell`:
        non-zero expression values, vocabulary indices, and metadata.

        Parameters
        ----------
        global_idx
            Global cell index (0 to len-1).

        Returns
        -------
        expression
            Non-zero expression values.
        gene_indices
            Vocabulary indices.
        metadata
            Cell metadata dict.
        """
        if global_idx < 0 or global_idx >= self._total_cells:
            raise IndexError(
                f"Cell index {global_idx} out of range [0, {self._total_cells})"
            )

        shard_idx, local_idx = self._resolve_index(global_idx)
        X, gene_idx, n_genes_arr = self._get_shard_memmaps(shard_idx)

        n = int(n_genes_arr[local_idx])
        expression = np.array(X[local_idx, :n], dtype=np.float32)
        gene_indices = np.array(gene_idx[local_idx, :n], dtype=np.int64)

        metadata = self._get_obs_metadata(shard_idx, local_idx)

        return expression, gene_indices, metadata

    def _resolve_index(self, global_idx: int) -> tuple[int, int]:
        """Map a global index to (shard_index, local_index)."""
        for shard_idx, cum_size in enumerate(self._cumulative_sizes):
            if global_idx < cum_size:
                prev = self._cumulative_sizes[shard_idx - 1] if shard_idx > 0 else 0
                return shard_idx, global_idx - prev
        raise IndexError(f"Index {global_idx} out of range")

    @property
    def n_shards(self) -> int:
        """Number of shards."""
        return self._n_shards

    @property
    def shard_sizes(self) -> list[int]:
        """Number of cells per shard."""
        return list(self._shard_sizes)

    def close(self) -> None:
        """Release all memory-mapped file references."""
        self._X_maps.clear()
        self._gene_idx_maps.clear()
        self._n_genes_maps.clear()
        self._obs_tables.clear()

    def __repr__(self) -> str:
        return (
            f"MemoryMappedStore(n_shards={self._n_shards}, "
            f"n_cells={self._total_cells}, dir={self._shard_dir})"
        )
