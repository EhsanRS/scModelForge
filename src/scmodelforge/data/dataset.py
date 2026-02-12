"""PyTorch Dataset for single-cell expression data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from scmodelforge.data.anndata_store import AnnDataStore

if TYPE_CHECKING:
    from pathlib import Path

    import anndata as ad

    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.data.memmap_store import MemoryMappedStore
    from scmodelforge.data.preprocessing import PreprocessingPipeline


class CellDataset(Dataset):
    """Map-style PyTorch Dataset that yields individual cells.

    Each cell is returned as a dict containing expression values,
    gene vocabulary indices, and metadata. Variable-length sequences
    are padded at collation time by :class:`CellDataLoader`.

    Parameters
    ----------
    adata
        AnnData object(s) or path(s) to .h5ad file(s).
    gene_vocab
        Gene vocabulary for index alignment.
    preprocessing
        Optional preprocessing pipeline applied on-the-fly.
    obs_keys
        Observation metadata keys to pass through.
    """

    def __init__(
        self,
        adata: ad.AnnData | str | Path | list[ad.AnnData | str | Path],
        gene_vocab: GeneVocab,
        preprocessing: PreprocessingPipeline | None = None,
        obs_keys: list[str] | None = None,
    ) -> None:
        # Normalise to list
        if not isinstance(adata, list):
            adata = [adata]

        self.gene_vocab = gene_vocab
        self.preprocessing = preprocessing
        self.store = AnnDataStore(adata, gene_vocab, obs_keys=obs_keys)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single cell.

        Returns
        -------
        dict
            - ``expression``: ``torch.Tensor`` of shape ``(n_genes,)``
              with expression values for non-zero, vocab-aligned genes.
            - ``gene_indices``: ``torch.Tensor`` of shape ``(n_genes,)``
              with vocabulary indices.
            - ``n_genes``: ``int``, number of genes (before padding).
            - ``metadata``: ``dict[str, str]`` with cell metadata.
        """
        expression, gene_indices, metadata = self.store.get_cell(idx)

        # Apply preprocessing
        if self.preprocessing is not None:
            expression = self.preprocessing(expression)

        return {
            "expression": torch.from_numpy(expression),
            "gene_indices": torch.from_numpy(gene_indices),
            "n_genes": len(expression),
            "metadata": metadata,
        }

    def __repr__(self) -> str:
        return (
            f"CellDataset(n_cells={len(self)}, n_datasets={self.store.n_datasets}, preprocessing={self.preprocessing})"
        )


class ShardedCellDataset(Dataset):
    """Map-style PyTorch Dataset backed by memory-mapped shards.

    Provides the same interface as :class:`CellDataset` but reads from
    a :class:`~scmodelforge.data.memmap_store.MemoryMappedStore` instead
    of in-memory AnnData objects.

    Parameters
    ----------
    shard_dir
        Path to the shard directory (or a pre-opened store).
    gene_vocab
        Optional gene vocabulary (for validation only).
    preprocessing
        Optional preprocessing pipeline applied on-the-fly.
    """

    def __init__(
        self,
        shard_dir: str | Path | MemoryMappedStore,
        gene_vocab: GeneVocab | None = None,
        preprocessing: PreprocessingPipeline | None = None,
    ) -> None:
        from scmodelforge.data.memmap_store import MemoryMappedStore as _MMS

        if isinstance(shard_dir, _MMS):
            self.store = shard_dir
        else:
            self.store = _MMS(shard_dir, gene_vocab=gene_vocab)

        self.gene_vocab = gene_vocab
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single cell.

        Returns the same dict format as :class:`CellDataset`.
        """
        expression, gene_indices, metadata = self.store.get_cell(idx)

        if self.preprocessing is not None:
            expression = self.preprocessing(expression)

        return {
            "expression": torch.from_numpy(expression),
            "gene_indices": torch.from_numpy(gene_indices),
            "n_genes": len(expression),
            "metadata": metadata,
        }

    def __repr__(self) -> str:
        return f"ShardedCellDataset(n_cells={len(self)}, n_shards={self.store.n_shards})"
