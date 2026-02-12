"""Tests for GeneSelectionCollator."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from torch.utils.data import DataLoader

from scmodelforge.data.dataset import CellDataset
from scmodelforge.data.gene_selection import GeneSelectionCollator
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.tokenizers.masking import MaskingStrategy
from scmodelforge.tokenizers.registry import get_tokenizer

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def gs_adata() -> AnnData:
    """20 cells x 50 genes, sparse counts."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 20, 50
    data = rng.poisson(lam=2, size=(n_cells, n_genes)).astype(np.float32)
    X = sp.csr_matrix(data)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    return AnnData(X=X, var={"gene_name": gene_names}, obs={"cell_type": ["A"] * n_cells})


@pytest.fixture()
def gs_vocab(gs_adata: AnnData) -> GeneVocab:
    return GeneVocab.from_adata(gs_adata)


@pytest.fixture()
def gs_tokenizer(gs_vocab: GeneVocab):
    return get_tokenizer("rank_value", gene_vocab=gs_vocab, max_len=64, prepend_cls=True)


@pytest.fixture()
def gs_masking(gs_vocab: GeneVocab) -> MaskingStrategy:
    return MaskingStrategy(mask_ratio=0.15, vocab_size=len(gs_vocab))


@pytest.fixture()
def gs_dataset(gs_adata: AnnData, gs_vocab: GeneVocab) -> CellDataset:
    return CellDataset(gs_adata, gs_vocab)


@pytest.fixture()
def raw_batch(gs_dataset: CellDataset) -> list[dict]:
    """A batch of 4 raw CellDataset dicts."""
    return [gs_dataset[i] for i in range(4)]


# ------------------------------------------------------------------
# Collator tests
# ------------------------------------------------------------------


class TestGeneSelectionCollator:
    """Tests for GeneSelectionCollator."""

    def test_strategy_all_same_as_tokenized(
        self, gs_tokenizer, gs_masking, gs_dataset, raw_batch,
    ) -> None:
        """strategy='all' produces same keys as standard pipeline."""
        collator = GeneSelectionCollator(
            tokenizer=gs_tokenizer, masking=gs_masking, strategy="all",
        )
        result = collator(raw_batch)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["input_ids"].ndim == 2
        assert result["input_ids"].shape[0] == 4

    def test_most_expressed_limits_genes(
        self, gs_tokenizer, gs_masking, raw_batch,
    ) -> None:
        """most_expressed with n_genes=5 limits genes per cell."""
        collator = GeneSelectionCollator(
            tokenizer=gs_tokenizer, masking=gs_masking,
            strategy="most_expressed", n_genes=5,
        )
        result = collator(raw_batch)
        assert "input_ids" in result
        # Sequence length should be at most n_genes + 1 (CLS token)
        assert result["input_ids"].shape[1] <= 5 + 1

    def test_most_expressed_selects_highest_sum(
        self, gs_tokenizer, raw_batch,
    ) -> None:
        """Selected genes should be the highest total expression."""
        collator = GeneSelectionCollator(
            tokenizer=gs_tokenizer, masking=None,
            strategy="most_expressed", n_genes=3,
        )
        # Compute expected top genes manually
        totals: dict[int, float] = {}
        for item in raw_batch:
            for gi, expr in zip(item["gene_indices"].tolist(), item["expression"].tolist(), strict=True):
                totals[gi] = totals.get(gi, 0.0) + expr
        expected_top = set(sorted(totals, key=totals.__getitem__, reverse=True)[:3])

        selected = collator._select_genes_most_expressed(raw_batch)
        assert selected == expected_top

    def test_random_expressed_limits_genes(
        self, gs_tokenizer, gs_masking, raw_batch,
    ) -> None:
        """random_expressed with n_genes=5 limits genes per cell."""
        collator = GeneSelectionCollator(
            tokenizer=gs_tokenizer, masking=gs_masking,
            strategy="random_expressed", n_genes=5,
        )
        result = collator(raw_batch)
        assert "input_ids" in result
        assert result["input_ids"].shape[1] <= 5 + 1

    def test_random_expressed_varies_across_calls(
        self, gs_tokenizer, raw_batch,
    ) -> None:
        """Random selection should (usually) differ across calls."""
        collator = GeneSelectionCollator(
            tokenizer=gs_tokenizer, masking=None,
            strategy="random_expressed", n_genes=5,
        )
        sets = [collator._select_genes_random(raw_batch) for _ in range(10)]
        # At least 2 different sets across 10 attempts
        unique_sets = {frozenset(s) for s in sets}
        assert len(unique_sets) > 1

    def test_filter_preserves_correspondence(
        self, gs_tokenizer, raw_batch,
    ) -> None:
        """Filtering keeps expression/gene_indices aligned."""
        selected = {raw_batch[0]["gene_indices"][0].item()}
        filtered = GeneSelectionCollator._filter_cell(raw_batch[0], selected)
        assert len(filtered["expression"]) == len(filtered["gene_indices"])
        assert filtered["n_genes"] == len(filtered["expression"])
        # Filtered gene should be in selected set
        for gi in filtered["gene_indices"].tolist():
            assert gi in selected

    def test_empty_batch_returns_empty(self, gs_tokenizer) -> None:
        collator = GeneSelectionCollator(tokenizer=gs_tokenizer, strategy="all")
        result = collator([])
        assert result == {}

    def test_n_genes_larger_than_available_keeps_all(
        self, gs_tokenizer, raw_batch,
    ) -> None:
        """When n_genes > available genes, keep all."""
        collator = GeneSelectionCollator(
            tokenizer=gs_tokenizer, masking=None,
            strategy="most_expressed", n_genes=9999,
        )
        all_genes: set[int] = set()
        for item in raw_batch:
            all_genes.update(item["gene_indices"].tolist())

        selected = collator._select_genes_most_expressed(raw_batch)
        assert selected == all_genes

    def test_invalid_strategy_raises(self, gs_tokenizer) -> None:
        with pytest.raises(ValueError, match="Unknown gene selection strategy"):
            GeneSelectionCollator(tokenizer=gs_tokenizer, strategy="invalid")

    def test_n_genes_required_when_not_all(self, gs_tokenizer) -> None:
        with pytest.raises(ValueError, match="n_genes is required"):
            GeneSelectionCollator(tokenizer=gs_tokenizer, strategy="most_expressed")

    def test_works_with_dataloader(
        self, gs_tokenizer, gs_masking, gs_dataset,
    ) -> None:
        """End-to-end: GeneSelectionCollator works as DataLoader collate_fn."""
        collator = GeneSelectionCollator(
            tokenizer=gs_tokenizer, masking=gs_masking,
            strategy="most_expressed", n_genes=5,
        )
        dl = DataLoader(gs_dataset, batch_size=4, collate_fn=collator, shuffle=False)
        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 4
