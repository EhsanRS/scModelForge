"""Tests for CellDataLoader and collation."""

from __future__ import annotations

import torch

from scmodelforge.data._utils import collate_cells
from scmodelforge.data.dataloader import CellDataLoader
from scmodelforge.data.dataset import CellDataset
from scmodelforge.data.gene_vocab import GeneVocab

# ------------------------------------------------------------------
# Collation
# ------------------------------------------------------------------


class TestCollation:
    def test_pads_to_max_length(self):
        batch = [
            {"expression": torch.tensor([1.0, 2.0, 3.0]), "gene_indices": torch.tensor([4, 5, 6]), "n_genes": 3},
            {"expression": torch.tensor([7.0, 8.0]), "gene_indices": torch.tensor([9, 10]), "n_genes": 2},
        ]
        result = collate_cells(batch)
        assert result["expression"].shape == (2, 3)
        assert result["gene_indices"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)

    def test_attention_mask_correct(self):
        batch = [
            {"expression": torch.tensor([1.0, 2.0, 3.0]), "gene_indices": torch.tensor([4, 5, 6]), "n_genes": 3},
            {"expression": torch.tensor([7.0]), "gene_indices": torch.tensor([9]), "n_genes": 1},
        ]
        result = collate_cells(batch)
        # First item: all real tokens
        assert result["attention_mask"][0].tolist() == [1, 1, 1]
        # Second item: 1 real + 2 padding
        assert result["attention_mask"][1].tolist() == [1, 0, 0]

    def test_padding_values(self):
        batch = [
            {"expression": torch.tensor([5.0]), "gene_indices": torch.tensor([10]), "n_genes": 1},
            {"expression": torch.tensor([1.0, 2.0]), "gene_indices": torch.tensor([3, 4]), "n_genes": 2},
        ]
        result = collate_cells(batch)
        # First item padded with 0s
        assert result["expression"][0, 1].item() == 0.0
        assert result["gene_indices"][0, 1].item() == 0

    def test_n_genes_preserved(self):
        batch = [
            {"expression": torch.tensor([1.0, 2.0]), "gene_indices": torch.tensor([4, 5]), "n_genes": 2},
            {"expression": torch.tensor([3.0]), "gene_indices": torch.tensor([6]), "n_genes": 1},
        ]
        result = collate_cells(batch)
        assert result["n_genes"].tolist() == [2, 1]

    def test_metadata_passed_through(self):
        batch = [
            {
                "expression": torch.tensor([1.0]),
                "gene_indices": torch.tensor([4]),
                "n_genes": 1,
                "metadata": {"cell_type": "T cell"},
            },
            {
                "expression": torch.tensor([2.0]),
                "gene_indices": torch.tensor([5]),
                "n_genes": 1,
                "metadata": {"cell_type": "B cell"},
            },
        ]
        result = collate_cells(batch)
        assert len(result["metadata"]) == 2
        assert result["metadata"][0]["cell_type"] == "T cell"
        assert result["metadata"][1]["cell_type"] == "B cell"

    def test_single_item_batch(self):
        batch = [
            {"expression": torch.tensor([1.0, 2.0]), "gene_indices": torch.tensor([4, 5]), "n_genes": 2},
        ]
        result = collate_cells(batch)
        assert result["expression"].shape == (1, 2)


# ------------------------------------------------------------------
# CellDataLoader
# ------------------------------------------------------------------


class TestCellDataLoader:
    def test_basic_iteration(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        loader = CellDataLoader(ds, batch_size=16, num_workers=0, shuffle=False, drop_last=False)

        batches = list(loader)
        assert len(batches) > 0

        batch = batches[0]
        assert "expression" in batch
        assert "gene_indices" in batch
        assert "attention_mask" in batch
        assert batch["expression"].shape[0] <= 16

    def test_batch_size(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        loader = CellDataLoader(ds, batch_size=32, num_workers=0, shuffle=False, drop_last=True)

        for batch in loader:
            assert batch["expression"].shape[0] == 32

    def test_drop_last(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)

        loader_drop = CellDataLoader(ds, batch_size=32, num_workers=0, shuffle=False, drop_last=True)
        loader_keep = CellDataLoader(ds, batch_size=32, num_workers=0, shuffle=False, drop_last=False)

        n_drop = sum(1 for _ in loader_drop)
        n_keep = sum(1 for _ in loader_keep)
        assert n_drop <= n_keep

    def test_reproducible_with_seed(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)

        loader1 = CellDataLoader(ds, batch_size=16, num_workers=0, shuffle=True, seed=42, drop_last=False)
        loader2 = CellDataLoader(ds, batch_size=16, num_workers=0, shuffle=True, seed=42, drop_last=False)

        for b1, b2 in zip(loader1, loader2, strict=False):
            torch.testing.assert_close(b1["gene_indices"], b2["gene_indices"])

    def test_len(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        loader = CellDataLoader(ds, batch_size=32, num_workers=0, shuffle=False, drop_last=True)
        assert len(loader) == 100 // 32  # 3 batches (drop_last=True)

    def test_repr(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        loader = CellDataLoader(ds, batch_size=16, num_workers=0, drop_last=False)
        r = repr(loader)
        assert "n_cells=100" in r
        assert "batch_size=16" in r
