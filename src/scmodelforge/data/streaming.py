"""Iterable streaming dataset for large-scale single-cell data."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.data.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


class StreamingCellDataset(IterableDataset):
    """Iterable dataset that streams cells from H5AD files chunk-by-chunk.

    Reads each file with ``anndata.read_h5ad(backed='r')`` and iterates
    in chunks, avoiding loading the entire dataset into memory.  A shuffle
    buffer provides local randomisation across chunks.

    When used with multiple DataLoader workers, files are sharded across
    workers so each worker processes a disjoint subset.

    Output format is identical to :class:`CellDataset.__getitem__`:
    ``{expression, gene_indices, n_genes, metadata}``.

    Parameters
    ----------
    file_paths
        Paths to ``.h5ad`` files.
    gene_vocab
        Gene vocabulary for index alignment.
    preprocessing
        Optional preprocessing pipeline applied to each cell.
    chunk_size
        Number of cells to read per chunk from each file.
    shuffle_buffer_size
        Size of the in-memory shuffle buffer.  Set to ``0`` to disable.
    obs_keys
        Observation metadata keys to pass through.
    seed
        Random seed for shuffle buffer.
    """

    def __init__(
        self,
        file_paths: list[str],
        gene_vocab: GeneVocab,
        preprocessing: PreprocessingPipeline | None = None,
        chunk_size: int = 10_000,
        shuffle_buffer_size: int = 10_000,
        obs_keys: list[str] | None = None,
        seed: int = 0,
        storage_options: dict | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.file_paths = list(file_paths)
        self.gene_vocab = gene_vocab
        self.preprocessing = preprocessing
        self.chunk_size = chunk_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.obs_keys = obs_keys or []
        self.seed = seed
        self.storage_options = storage_options
        self.cache_dir = cache_dir

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield cells with worker-aware file sharding."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Shard files across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = [f for i, f in enumerate(self.file_paths) if i % num_workers == worker_id]
        else:
            files = self.file_paths

        rng = np.random.default_rng(self.seed)
        buffer: deque[dict[str, Any]] = deque()

        for cell in self._iter_files(files):
            if self.shuffle_buffer_size <= 0:
                yield cell
                continue

            buffer.append(cell)
            if len(buffer) >= self.shuffle_buffer_size:
                idx = int(rng.integers(0, len(buffer)))
                # Swap selected item to end and pop
                buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                yield buffer.pop()

        # Drain remaining buffer in random order
        items = list(buffer)
        rng.shuffle(items)
        yield from items

    def _iter_files(
        self, files: list[str],
    ) -> Iterator[dict[str, Any]]:
        """Iterate through files, reading chunks and aligning genes."""
        import anndata as ad

        from scmodelforge.data.cloud import is_cloud_path
        from scmodelforge.data.cloud import read_h5ad as cloud_read_h5ad

        for file_path in files:
            try:
                if is_cloud_path(file_path):
                    adata = cloud_read_h5ad(
                        file_path,
                        storage_options=self.storage_options,
                        backed="r",
                        cache_dir=self.cache_dir,
                    )
                else:
                    adata = ad.read_h5ad(file_path, backed="r")
            except Exception:
                logger.warning("Failed to open %s, skipping.", file_path)
                continue

            n_cells = adata.n_obs
            if n_cells == 0:
                continue

            # Build gene alignment: map file gene names -> vocab indices
            var_names = list(adata.var_names)
            file_gene_mask = []
            vocab_indices = []
            for j, name in enumerate(var_names):
                if name in self.gene_vocab:
                    file_gene_mask.append(j)
                    vocab_indices.append(self.gene_vocab[name])

            if not file_gene_mask:
                logger.warning(
                    "No genes in %s match the vocabulary, skipping.", file_path,
                )
                continue

            file_gene_mask_arr = np.array(file_gene_mask, dtype=np.intp)
            vocab_indices_arr = np.array(vocab_indices, dtype=np.int64)

            # Read in chunks
            for start in range(0, n_cells, self.chunk_size):
                end = min(start + self.chunk_size, n_cells)
                chunk = adata[start:end]
                X_chunk = chunk.X

                # Densify if sparse
                import scipy.sparse as sp

                if sp.issparse(X_chunk):
                    X_chunk = X_chunk.toarray()
                X_chunk = np.asarray(X_chunk, dtype=np.float32)

                # Extract obs metadata for chunk
                obs_meta = {}
                for key in self.obs_keys:
                    if key in chunk.obs.columns:
                        obs_meta[key] = list(chunk.obs[key])

                for i in range(X_chunk.shape[0]):
                    row = X_chunk[i]
                    # Select aligned genes
                    expression = row[file_gene_mask_arr]
                    gene_indices = vocab_indices_arr.copy()

                    # Filter non-zero
                    nonzero = expression > 0
                    expression = expression[nonzero]
                    gene_indices = gene_indices[nonzero]

                    if len(expression) == 0:
                        continue

                    # Apply preprocessing
                    if self.preprocessing is not None:
                        expression = self.preprocessing(expression)

                    # Build metadata dict
                    metadata = {}
                    for key, vals in obs_meta.items():
                        metadata[key] = str(vals[i])

                    yield {
                        "expression": torch.from_numpy(expression),
                        "gene_indices": torch.from_numpy(gene_indices),
                        "n_genes": len(expression),
                        "metadata": metadata,
                    }

    def __repr__(self) -> str:
        return (
            f"StreamingCellDataset(n_files={len(self.file_paths)}, "
            f"chunk_size={self.chunk_size}, buffer={self.shuffle_buffer_size})"
        )
