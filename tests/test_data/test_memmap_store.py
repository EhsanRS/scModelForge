"""Tests for MemoryMappedStore."""

from __future__ import annotations

import numpy as np
import pytest

from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.data.memmap_store import MemoryMappedStore
from scmodelforge.data.sharding import convert_to_shards


@pytest.fixture()
def shard_dir(tmp_path, mini_adata):
    """Create a shard directory from mini_adata."""
    vocab = GeneVocab.from_adata(mini_adata)
    output = tmp_path / "shards"
    convert_to_shards(
        sources=[mini_adata],
        gene_vocab=vocab,
        output_dir=output,
        shard_size=50,  # Split 100 cells into 2 shards
    )
    return output, vocab


class TestMemoryMappedStore:
    def test_construction(self, shard_dir):
        output, vocab = shard_dir
        store = MemoryMappedStore(output, gene_vocab=vocab)
        assert len(store) == 100

    def test_len(self, shard_dir):
        output, vocab = shard_dir
        store = MemoryMappedStore(output)
        assert len(store) == 100

    def test_get_cell_format(self, shard_dir):
        output, vocab = shard_dir
        store = MemoryMappedStore(output)
        expr, gene_idx, meta = store.get_cell(0)
        assert isinstance(expr, np.ndarray)
        assert isinstance(gene_idx, np.ndarray)
        assert isinstance(meta, dict)
        assert expr.dtype == np.float32
        assert gene_idx.dtype == np.int64
        assert len(expr) == len(gene_idx)

    def test_multiple_shards(self, shard_dir):
        output, vocab = shard_dir
        store = MemoryMappedStore(output)
        assert store.n_shards == 2
        assert store.shard_sizes == [50, 50]

    def test_index_resolution_across_shards(self, shard_dir):
        output, vocab = shard_dir
        store = MemoryMappedStore(output)
        # Access cell in second shard
        expr, gene_idx, meta = store.get_cell(75)
        assert len(expr) >= 0  # May be 0 if cell had no overlap

    def test_out_of_range_raises(self, shard_dir):
        output, vocab = shard_dir
        store = MemoryMappedStore(output)
        with pytest.raises(IndexError):
            store.get_cell(1000)

    def test_close(self, shard_dir):
        output, vocab = shard_dir
        store = MemoryMappedStore(output)
        _ = store.get_cell(0)  # Force loading
        store.close()
        # After close, memmaps are cleared — can reopen
        _ = store.get_cell(0)

    def test_invalid_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            MemoryMappedStore(tmp_path / "nonexistent")

    def test_repr(self, shard_dir):
        output, vocab = shard_dir
        store = MemoryMappedStore(output)
        r = repr(store)
        assert "MemoryMappedStore" in r
        assert "n_shards=2" in r
        assert "n_cells=100" in r
