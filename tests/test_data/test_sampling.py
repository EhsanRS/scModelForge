"""Tests for WeightedCellSampler and extract_labels_from_dataset."""

from __future__ import annotations

import collections

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch.utils.data import DataLoader, random_split

from scmodelforge.data.dataset import CellDataset
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.data.sampling import WeightedCellSampler, extract_labels_from_dataset

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def imbalanced_labels() -> list[str]:
    """100 labels: 90 type_A, 10 type_B."""
    return ["type_A"] * 90 + ["type_B"] * 10


@pytest.fixture()
def balanced_labels() -> list[str]:
    """20 labels: 10 each."""
    return ["type_A"] * 10 + ["type_B"] * 10


@pytest.fixture()
def imbalanced_adata() -> AnnData:
    """30 cells x 10 genes: 24 type_A, 6 type_B."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 30, 10
    X = sp.csr_matrix(rng.poisson(2, (n_cells, n_genes)).astype(np.float32))
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    cell_types = ["type_A"] * 24 + ["type_B"] * 6
    return AnnData(X=X, var={"gene_name": gene_names}, obs={"cell_type": cell_types})


# ------------------------------------------------------------------
# WeightedCellSampler
# ------------------------------------------------------------------


class TestWeightedCellSampler:
    """Tests for WeightedCellSampler."""

    def test_produces_correct_number_of_indices(self, imbalanced_labels: list[str]) -> None:
        sampler = WeightedCellSampler(imbalanced_labels, seed=0)
        indices = list(sampler)
        assert len(indices) == len(imbalanced_labels)

    def test_len_returns_dataset_size(self, imbalanced_labels: list[str]) -> None:
        sampler = WeightedCellSampler(imbalanced_labels)
        assert len(sampler) == 100

    def test_rare_class_gets_higher_weight(self, imbalanced_labels: list[str]) -> None:
        sampler = WeightedCellSampler(imbalanced_labels)
        weights = sampler.class_weights
        assert weights["type_B"] > weights["type_A"]
        # type_B is 9x rarer, so weight should be 9x higher
        assert abs(weights["type_B"] / weights["type_A"] - 9.0) < 0.01

    def test_class_weights_property(self, balanced_labels: list[str]) -> None:
        sampler = WeightedCellSampler(balanced_labels)
        weights = sampler.class_weights
        assert set(weights.keys()) == {"type_A", "type_B"}
        # Balanced: both have same weight
        assert abs(weights["type_A"] - weights["type_B"]) < 0.01

    def test_deterministic_with_seed(self, imbalanced_labels: list[str]) -> None:
        s1 = WeightedCellSampler(imbalanced_labels, seed=123)
        s2 = WeightedCellSampler(imbalanced_labels, seed=123)
        assert list(s1) == list(s2)

    def test_different_seeds_differ(self, imbalanced_labels: list[str]) -> None:
        s1 = WeightedCellSampler(imbalanced_labels, seed=0)
        s2 = WeightedCellSampler(imbalanced_labels, seed=999)
        # Very unlikely to be identical with different seeds
        assert list(s1) != list(s2)

    def test_set_epoch_changes_effective_weights(self, imbalanced_labels: list[str]) -> None:
        sampler = WeightedCellSampler(imbalanced_labels, seed=0, curriculum_warmup_epochs=5)
        sampler.set_epoch(0)
        indices_epoch0 = list(sampler)
        sampler.set_epoch(5)
        indices_epoch5 = list(sampler)
        # Different epochs should (very likely) produce different samples
        assert indices_epoch0 != indices_epoch5

    def test_curriculum_warmup_epoch0_near_uniform(self, imbalanced_labels: list[str]) -> None:
        """At epoch 0, curriculum should produce near-uniform sampling."""
        sampler = WeightedCellSampler(imbalanced_labels, seed=0, curriculum_warmup_epochs=10)
        sampler.set_epoch(0)
        # At epoch 0, alpha=0, so weights are uniform
        effective = sampler._effective_weights()
        # All weights should be equal for uniform
        assert torch.allclose(effective, effective[0].expand_as(effective))

    def test_curriculum_disabled_full_weighting(self, imbalanced_labels: list[str]) -> None:
        """With warmup=0, full weighting from the start."""
        sampler = WeightedCellSampler(imbalanced_labels, seed=42, curriculum_warmup_epochs=0)
        # Sample many times and check type_B is well-represented
        indices = list(sampler)
        type_b_count = sum(1 for i in indices if imbalanced_labels[i] == "type_B")
        # With full inverse-frequency weighting, type_B should get ~50%
        assert type_b_count > 20  # At least 20% (generous lower bound)

    def test_empty_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            WeightedCellSampler([])

    def test_works_with_dataloader(self, balanced_labels: list[str]) -> None:
        """Integration: sampler works with a DataLoader."""
        dataset = torch.utils.data.TensorDataset(torch.arange(len(balanced_labels)))
        sampler = WeightedCellSampler(balanced_labels, seed=0)
        dl = DataLoader(dataset, batch_size=4, sampler=sampler)
        batches = list(dl)
        total = sum(b[0].shape[0] for b in batches)
        assert total == len(balanced_labels)


# ------------------------------------------------------------------
# extract_labels_from_dataset
# ------------------------------------------------------------------


class TestExtractLabels:
    """Tests for extract_labels_from_dataset."""

    def test_from_cell_dataset(self, imbalanced_adata: AnnData) -> None:
        vocab = GeneVocab.from_adata(imbalanced_adata)
        ds = CellDataset(imbalanced_adata, vocab, obs_keys=["cell_type"])
        labels = extract_labels_from_dataset(ds, label_key="cell_type")
        assert len(labels) == 30
        assert collections.Counter(labels) == {"type_A": 24, "type_B": 6}

    def test_from_subset(self, imbalanced_adata: AnnData) -> None:
        vocab = GeneVocab.from_adata(imbalanced_adata)
        ds = CellDataset(imbalanced_adata, vocab, obs_keys=["cell_type"])
        generator = torch.Generator().manual_seed(42)
        subset, _ = random_split(ds, [20, 10], generator=generator)
        labels = extract_labels_from_dataset(subset, label_key="cell_type")
        assert len(labels) == 20
        assert all(lbl in {"type_A", "type_B"} for lbl in labels)

    def test_missing_label_key_returns_unknown(self, imbalanced_adata: AnnData) -> None:
        vocab = GeneVocab.from_adata(imbalanced_adata)
        ds = CellDataset(imbalanced_adata, vocab)
        labels = extract_labels_from_dataset(ds, label_key="nonexistent_key")
        assert all(lbl == "unknown" for lbl in labels)
        assert len(labels) == 30
