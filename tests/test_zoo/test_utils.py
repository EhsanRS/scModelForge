"""Tests for zoo utility functions."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scmodelforge.zoo._utils import align_genes_to_model, compute_gene_overlap, ensure_raw_counts, require_package
from scmodelforge.zoo.base import GeneOverlapReport


class TestRequirePackage:
    def test_installed_package(self):
        """No error for an installed package."""
        require_package("numpy")

    def test_missing_package_raises(self):
        with pytest.raises(ImportError, match="pip install 'nonexistent_pkg_xyz'"):
            require_package("nonexistent_pkg_xyz")

    def test_custom_pip_name(self):
        with pytest.raises(ImportError, match="pip install 'my-special-pkg>=1.0'"):
            require_package("nonexistent_pkg_xyz", "my-special-pkg>=1.0")


class TestComputeGeneOverlap:
    def test_full_overlap(self):
        genes = [f"g{i}" for i in range(10)]
        report = compute_gene_overlap(genes, genes)
        assert report.matched == 10
        assert report.missing == 0
        assert report.extra == 0
        assert report.coverage == pytest.approx(1.0)

    def test_no_overlap(self):
        adata_genes = [f"a{i}" for i in range(5)]
        model_genes = [f"m{i}" for i in range(8)]
        report = compute_gene_overlap(adata_genes, model_genes)
        assert report.matched == 0
        assert report.missing == 8
        assert report.extra == 5
        assert report.coverage == pytest.approx(0.0)

    def test_partial_overlap(self):
        adata_genes = ["g0", "g1", "g2", "extra1"]
        model_genes = ["g0", "g1", "g2", "missing1", "missing2"]
        report = compute_gene_overlap(adata_genes, model_genes)
        assert report.matched == 3
        assert report.missing == 2
        assert report.extra == 1
        assert report.coverage == pytest.approx(3 / 5)
        assert report.model_vocab_size == 5
        assert report.adata_n_genes == 4

    def test_empty_model_genes(self):
        report = compute_gene_overlap(["g0", "g1"], [])
        assert report.matched == 0
        assert report.coverage == pytest.approx(0.0)
        assert report.model_vocab_size == 0

    def test_returns_correct_type(self):
        report = compute_gene_overlap(["g0"], ["g0"])
        assert isinstance(report, GeneOverlapReport)


class TestAlignGenesToModel:
    def _make_adata(self, gene_names, n_cells=5, sparse=True):
        rng = np.random.default_rng(42)
        X = rng.poisson(lam=2.0, size=(n_cells, len(gene_names))).astype(np.float32)
        if sparse:
            X = sp.csr_matrix(X)
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
        var = pd.DataFrame(index=gene_names)
        return ad.AnnData(X=X, obs=obs, var=var)

    def test_identical_genes(self):
        genes = ["g0", "g1", "g2"]
        adata = self._make_adata(genes)
        result = align_genes_to_model(adata, genes)
        assert list(result.var_names) == genes
        assert result.n_obs == 5
        assert result.n_vars == 3

    def test_reorders_genes(self):
        adata = self._make_adata(["g0", "g1", "g2"])
        result = align_genes_to_model(adata, ["g2", "g0", "g1"])
        assert list(result.var_names) == ["g2", "g0", "g1"]

    def test_missing_genes_filled_with_zeros(self):
        adata = self._make_adata(["g0", "g1"])
        result = align_genes_to_model(adata, ["g0", "g1", "g_missing"])
        assert list(result.var_names) == ["g0", "g1", "g_missing"]
        # Missing gene column should be all zeros
        missing_col = result[:, "g_missing"].X
        if sp.issparse(missing_col):
            missing_col = missing_col.toarray()
        assert np.all(missing_col == 0.0)

    def test_extra_genes_dropped(self):
        adata = self._make_adata(["g0", "g1", "g2", "extra"])
        result = align_genes_to_model(adata, ["g0", "g2"])
        assert list(result.var_names) == ["g0", "g2"]
        assert result.n_vars == 2

    def test_preserves_obs(self):
        adata = self._make_adata(["g0", "g1"])
        adata.obs["label"] = ["A", "B", "A", "B", "A"]
        result = align_genes_to_model(adata, ["g0"])
        assert list(result.obs["label"]) == ["A", "B", "A", "B", "A"]

    def test_sparse_output(self):
        adata = self._make_adata(["g0", "g1"], sparse=True)
        result = align_genes_to_model(adata, ["g0", "g1"])
        assert sp.issparse(result.X)

    def test_dense_input_works(self):
        adata = self._make_adata(["g0", "g1"], sparse=False)
        result = align_genes_to_model(adata, ["g1", "g0"])
        assert list(result.var_names) == ["g1", "g0"]

    def test_zero_overlap(self):
        adata = self._make_adata(["a", "b"])
        result = align_genes_to_model(adata, ["x", "y", "z"])
        assert result.n_vars == 3
        X_dense = result.X.toarray() if sp.issparse(result.X) else result.X
        assert np.all(X_dense == 0.0)

    def test_custom_fill_value(self):
        adata = self._make_adata(["g0"])
        result = align_genes_to_model(adata, ["g0", "g_missing"], fill_missing=-1.0)
        missing_col = result[:, "g_missing"].X
        if sp.issparse(missing_col):
            missing_col = missing_col.toarray()
        assert np.all(missing_col == -1.0)


class TestEnsureRawCounts:
    def test_uses_raw_when_available(self):
        rng = np.random.default_rng(42)
        X = rng.poisson(lam=5.0, size=(10, 20)).astype(np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(X))
        # Store raw, then overwrite X with normalized
        adata.raw = adata.copy()
        adata.X = sp.csr_matrix(np.log1p(X))

        result = ensure_raw_counts(adata)
        # Result should have raw counts, not log-transformed
        result_X = result.X.toarray() if sp.issparse(result.X) else result.X
        np.testing.assert_array_almost_equal(result_X, X)

    def test_returns_adata_when_no_raw(self):
        rng = np.random.default_rng(42)
        X = rng.poisson(lam=5.0, size=(10, 20)).astype(np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(X))
        result = ensure_raw_counts(adata)
        assert result is adata
