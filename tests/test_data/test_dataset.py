"""Tests for CellDataset and AnnDataStore."""

from __future__ import annotations

import anndata as ad
import pytest
import scipy.sparse as sp

from scmodelforge.data.dataset import CellDataset
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.data.preprocessing import PreprocessingPipeline

# ------------------------------------------------------------------
# CellDataset basics
# ------------------------------------------------------------------


class TestCellDataset:
    def test_from_adata(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        assert len(ds) == 100

    def test_from_h5ad_path(self, tmp_h5ad, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(tmp_h5ad, gene_vocab=vocab)
        assert len(ds) == 100

    def test_getitem_returns_correct_keys(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        item = ds[0]
        assert "expression" in item
        assert "gene_indices" in item
        assert "n_genes" in item
        assert "metadata" in item

    def test_getitem_shapes_match(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        item = ds[0]
        assert item["expression"].shape == item["gene_indices"].shape
        assert item["n_genes"] == len(item["expression"])
        assert item["n_genes"] > 0  # Sparse matrix should have non-zero entries

    def test_getitem_expression_is_float(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        item = ds[0]
        assert item["expression"].is_floating_point()

    def test_getitem_gene_indices_are_valid(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        item = ds[0]
        assert (item["gene_indices"] >= 0).all()
        assert (item["gene_indices"] < len(vocab)).all()

    def test_with_preprocessing(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        pipeline = PreprocessingPipeline(normalize="library_size", target_sum=1e4, log1p=True)
        ds = CellDataset(mini_adata, gene_vocab=vocab, preprocessing=pipeline)
        item = ds[0]
        # After log1p, all values should be >= 0
        assert (item["expression"] >= 0).all()

    def test_with_obs_keys(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab, obs_keys=["cell_type", "batch"])
        item = ds[0]
        assert "cell_type" in item["metadata"]
        assert "batch" in item["metadata"]
        assert item["metadata"]["cell_type"] in ["T cell", "B cell", "Monocyte"]

    def test_missing_obs_key_returns_unknown(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab, obs_keys=["nonexistent_key"])
        item = ds[0]
        assert item["metadata"]["nonexistent_key"] == "unknown"

    def test_out_of_range_index_raises(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset(mini_adata, gene_vocab=vocab)
        with pytest.raises(IndexError):
            ds[1000]


# ------------------------------------------------------------------
# Multiple datasets
# ------------------------------------------------------------------


class TestMultiDataset:
    def test_multiple_adatas(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset([mini_adata, mini_adata], gene_vocab=vocab)
        assert len(ds) == 200

    def test_multiple_h5ad_paths(self, tmp_h5ad, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset([tmp_h5ad, tmp_h5ad], gene_vocab=vocab)
        assert len(ds) == 200

    def test_cross_dataset_indexing(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        ds = CellDataset([mini_adata, mini_adata], gene_vocab=vocab)
        # Index into second dataset
        item = ds[150]
        assert item["n_genes"] > 0


# ------------------------------------------------------------------
# Gene alignment
# ------------------------------------------------------------------


class TestGeneAlignment:
    def test_partial_vocab_overlap(self):
        """Dataset with genes not in vocab should still work."""
        n_obs, n_vars = 50, 100
        X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=42)
        adata = ad.AnnData(X=X, var={"gene_name": [f"G_{i}" for i in range(n_vars)]})
        adata.var_names = [f"G_{i}" for i in range(n_vars)]

        # Vocab only contains half the genes
        vocab = GeneVocab.from_genes([f"G_{i}" for i in range(50)])
        ds = CellDataset(adata, gene_vocab=vocab)
        item = ds[0]
        # All gene indices should be in vocab range
        assert (item["gene_indices"] >= 0).all()
        assert (item["gene_indices"] < len(vocab)).all()

    def test_no_vocab_overlap(self):
        """Dataset with no genes in vocab should return empty sequences."""
        X = sp.random(10, 5, density=0.5, format="csr", random_state=42)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"DATASET_GENE_{i}" for i in range(5)]

        vocab = GeneVocab.from_genes(["VOCAB_GENE_0", "VOCAB_GENE_1"])
        ds = CellDataset(adata, gene_vocab=vocab)
        item = ds[0]
        assert item["n_genes"] == 0
