"""Tests for DistributedShardSampler."""

from __future__ import annotations

import pytest

from scmodelforge.data.distributed import DistributedShardSampler
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.data.memmap_store import MemoryMappedStore
from scmodelforge.data.sharding import convert_to_shards


@pytest.fixture()
def store_4shards(tmp_path, mini_adata):
    """Create a store with 4 shards (25 cells each)."""
    vocab = GeneVocab.from_adata(mini_adata)
    output = tmp_path / "shards"
    convert_to_shards(
        sources=[mini_adata],
        gene_vocab=vocab,
        output_dir=output,
        shard_size=25,
    )
    return MemoryMappedStore(output)


class TestDistributedShardSampler:
    def test_assigns_shards_to_ranks(self, store_4shards):
        """Two ranks should each get 2 of 4 shards."""
        sampler0 = DistributedShardSampler(
            store_4shards, num_replicas=2, rank=0, shuffle=False
        )
        sampler1 = DistributedShardSampler(
            store_4shards, num_replicas=2, rank=1, shuffle=False
        )
        indices0 = list(sampler0)
        indices1 = list(sampler1)
        # Together they should cover all cells
        assert len(indices0) + len(indices1) == 100
        # Each rank should have 50 cells (2 shards of 25)
        assert len(indices0) == 50
        assert len(indices1) == 50

    def test_no_overlap_between_ranks(self, store_4shards):
        sampler0 = DistributedShardSampler(
            store_4shards, num_replicas=2, rank=0, shuffle=False
        )
        sampler1 = DistributedShardSampler(
            store_4shards, num_replicas=2, rank=1, shuffle=False
        )
        set0 = set(sampler0)
        set1 = set(sampler1)
        assert len(set0 & set1) == 0

    def test_set_epoch_changes_order(self, store_4shards):
        sampler = DistributedShardSampler(
            store_4shards, num_replicas=1, rank=0, shuffle=True, seed=0
        )
        sampler.set_epoch(0)
        indices_e0 = list(sampler)
        sampler.set_epoch(1)
        indices_e1 = list(sampler)
        # Different epochs should produce different orderings
        assert indices_e0 != indices_e1
        # But same total count
        assert len(indices_e0) == len(indices_e1)

    def test_len_correct(self, store_4shards):
        sampler = DistributedShardSampler(
            store_4shards, num_replicas=2, rank=0, shuffle=False
        )
        assert len(sampler) == 50

    def test_single_rank(self, store_4shards):
        """With one rank, all cells are yielded."""
        sampler = DistributedShardSampler(
            store_4shards, num_replicas=1, rank=0, shuffle=False
        )
        indices = list(sampler)
        assert len(indices) == 100
        assert set(indices) == set(range(100))
