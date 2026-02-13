"""Tests for DistributedShardSampler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scmodelforge.data.distributed import DistributedShardSampler
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.data.memmap_store import MemoryMappedStore
from scmodelforge.data.sharding import convert_to_shards


def _mock_store(shard_sizes: list[int]) -> MagicMock:
    """Create a lightweight mock store with given shard sizes."""
    store = MagicMock()
    store.n_shards = len(shard_sizes)
    store.shard_sizes = list(shard_sizes)
    return store


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


class TestDropLastEqualisation:
    """Tests for drop_last with imbalanced shard sizes."""

    def test_drop_last_equalises_ranks_imbalanced(self):
        """With imbalanced shards, drop_last truncates to min per-rank count."""
        # 3 shards: sizes 100, 50, 30.  With 2 ranks (no shuffle):
        #   rank 0 gets shards [0, 2] = 100+30 = 130
        #   rank 1 gets shard  [1]    = 50
        # drop_last should truncate both to 50.
        store = _mock_store([100, 50, 30])
        s0 = DistributedShardSampler(store, num_replicas=2, rank=0, shuffle=False, drop_last=True)
        s1 = DistributedShardSampler(store, num_replicas=2, rank=1, shuffle=False, drop_last=True)

        assert len(s0) == len(s1) == 50
        assert len(list(s0)) == len(list(s1)) == 50

    def test_drop_last_no_effect_when_balanced(self, store_4shards):
        """With equal-sized shards, drop_last doesn't reduce count."""
        s0 = DistributedShardSampler(store_4shards, num_replicas=2, rank=0, shuffle=False, drop_last=True)
        s1 = DistributedShardSampler(store_4shards, num_replicas=2, rank=1, shuffle=False, drop_last=True)
        assert len(s0) == len(s1) == 50
        assert len(list(s0)) == len(list(s1)) == 50

    def test_drop_last_single_rank_noop(self):
        """drop_last is a no-op with a single rank."""
        store = _mock_store([100, 50])
        sampler = DistributedShardSampler(store, num_replicas=1, rank=0, shuffle=False, drop_last=True)
        # Single rank gets all cells, drop_last shouldn't truncate
        assert len(sampler) == 150
        assert len(list(sampler)) == 150

    def test_drop_last_len_matches_iter(self):
        """__len__ must match actual iteration count when drop_last=True."""
        store = _mock_store([80, 60, 40, 20])
        for rank in range(3):
            sampler = DistributedShardSampler(
                store, num_replicas=3, rank=rank, shuffle=False, drop_last=True,
            )
            assert len(list(sampler)) == len(sampler)

    def test_drop_last_with_shuffle(self):
        """All ranks produce equal counts even with epoch-based shuffling."""
        store = _mock_store([100, 70, 30, 20])
        for epoch in range(5):
            counts = []
            for rank in range(2):
                sampler = DistributedShardSampler(
                    store, num_replicas=2, rank=rank, shuffle=True, seed=42, drop_last=True,
                )
                sampler.set_epoch(epoch)
                counts.append(len(list(sampler)))
            assert counts[0] == counts[1], f"Epoch {epoch}: ranks have different counts {counts}"
