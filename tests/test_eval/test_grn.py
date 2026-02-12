"""Tests for GRN inference benchmark."""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from scmodelforge.eval.grn import (
    GRNBenchmark,
    compute_gene_representations,
    compute_gene_similarity_scores,
    load_ground_truth_network,
)

# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestLoadGroundTruthNetwork:
    def test_loads_tsv(self, tmp_path):
        tsv = tmp_path / "network.tsv"
        tsv.write_text("gene1\tgene2\tweight\nGENE_A\tGENE_B\t1.0\nGENE_C\tGENE_D\t0.0\n")
        edges = load_ground_truth_network(str(tsv))
        assert len(edges) == 2
        assert edges[0] == ("GENE_A", "GENE_B", 1.0)
        assert edges[1] == ("GENE_C", "GENE_D", 0.0)


class TestComputeGeneRepresentations:
    def _make_adata(self, X, gene_names):
        import pandas as pd

        return ad.AnnData(
            X=X,
            var=pd.DataFrame(index=gene_names),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(X.shape[0])]),
        )

    def test_basic(self):
        X = np.array([
            [5.0, 0.0],
            [3.0, 1.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        adata = self._make_adata(X, ["G1", "G2"])
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)

        reps = compute_gene_representations(embeddings, adata, min_cells=1)
        # G1 is expressed in cells 0 and 1
        assert "G1" in reps
        np.testing.assert_allclose(reps["G1"], [0.5, 0.5])
        # G2 is expressed in cells 1 and 2
        assert "G2" in reps
        np.testing.assert_allclose(reps["G2"], [0.5, 1.0])

    def test_min_cells_filters(self):
        X = np.array([
            [5.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)
        adata = self._make_adata(X, ["G1", "G2"])
        embeddings = np.ones((3, 4), dtype=np.float32)

        reps = compute_gene_representations(embeddings, adata, min_cells=2)
        assert "G1" not in reps
        assert "G2" not in reps

    def test_sparse_input(self):
        X_dense = np.array([
            [5.0, 0.0],
            [3.0, 1.0],
        ], dtype=np.float32)
        X_sparse = sp.csr_matrix(X_dense)
        adata = self._make_adata(X_sparse, ["G1", "G2"])
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        reps = compute_gene_representations(embeddings, adata, min_cells=1)
        assert "G1" in reps
        np.testing.assert_allclose(reps["G1"], [0.5, 0.5])


class TestComputeGeneSimilarityScores:
    def test_basic(self):
        reps = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
            "C": np.array([1.0, 0.0]),
        }
        pairs = [("A", "B"), ("A", "C"), ("B", "C")]
        scores, evaluable = compute_gene_similarity_scores(reps, pairs)
        assert len(scores) == 3
        assert len(evaluable) == 3
        # A-B are orthogonal -> 0.0
        np.testing.assert_allclose(scores[0], 0.0, atol=1e-6)
        # A-C are identical -> 1.0
        np.testing.assert_allclose(scores[1], 1.0, atol=1e-6)

    def test_missing_genes_skipped(self):
        reps = {"A": np.array([1.0, 0.0])}
        pairs = [("A", "MISSING")]
        scores, evaluable = compute_gene_similarity_scores(reps, pairs)
        assert len(scores) == 0
        assert len(evaluable) == 0


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestGRNBenchmark:
    def _make_adata(self, n_cells=100, n_genes=20, seed=42):
        import pandas as pd

        rng = np.random.default_rng(seed)
        gene_names = [f"GENE_{i}" for i in range(n_genes)]
        X = rng.random((n_cells, n_genes)).astype(np.float32)
        # Make some entries zero to create variability
        X[X < 0.3] = 0.0
        return ad.AnnData(
            X=X,
            var=pd.DataFrame(index=gene_names),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        )

    def test_name(self):
        gt = [("A", "B", 1.0)]
        bench = GRNBenchmark(ground_truth=gt)
        assert bench.name == "grn_inference"

    def test_required_obs_keys_empty(self):
        gt = [("A", "B", 1.0)]
        bench = GRNBenchmark(ground_truth=gt)
        assert bench.required_obs_keys == []

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Either network_file or ground_truth"):
            GRNBenchmark()

    def test_run_returns_result(self):
        adata = self._make_adata(n_cells=100, n_genes=20)
        embeddings = np.random.default_rng(0).standard_normal((100, 32)).astype(np.float32)

        # Create ground truth with positive and negative edges
        gene_names = list(adata.var_names)
        gt = [
            (gene_names[0], gene_names[1], 1.0),
            (gene_names[2], gene_names[3], 1.0),
            (gene_names[4], gene_names[5], 0.0),
            (gene_names[6], gene_names[7], 0.0),
            (gene_names[8], gene_names[9], 1.0),
            (gene_names[10], gene_names[11], 0.0),
        ]

        bench = GRNBenchmark(ground_truth=gt, min_cells_per_gene=5)
        result = bench.run(embeddings, adata, "test_dataset")

        assert result.benchmark_name == "grn_inference"
        assert result.dataset_name == "test_dataset"
        assert "auroc" in result.metrics
        assert "auprc" in result.metrics
        assert "n_genes_evaluated" in result.metrics
        assert "n_edges_evaluated" in result.metrics

    def test_few_pairs_graceful(self):
        adata = self._make_adata(n_cells=50, n_genes=5)
        embeddings = np.random.default_rng(1).standard_normal((50, 16)).astype(np.float32)

        # Only one edge with same label => cannot compute AUROC
        gt = [("GENE_0", "GENE_1", 1.0)]
        bench = GRNBenchmark(ground_truth=gt, min_cells_per_gene=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = bench.run(embeddings, adata, "test")

        assert result.metrics["auroc"] == 0.5
        assert result.metrics["auprc"] == 0.5

    def test_from_file(self, tmp_path):
        tsv = tmp_path / "network.tsv"
        adata = self._make_adata(n_cells=100, n_genes=20)
        gene_names = list(adata.var_names)
        lines = ["gene1\tgene2\tweight"]
        lines.append(f"{gene_names[0]}\t{gene_names[1]}\t1.0")
        lines.append(f"{gene_names[2]}\t{gene_names[3]}\t0.0")
        lines.append(f"{gene_names[4]}\t{gene_names[5]}\t1.0")
        lines.append(f"{gene_names[6]}\t{gene_names[7]}\t0.0")
        tsv.write_text("\n".join(lines) + "\n")

        embeddings = np.random.default_rng(2).standard_normal((100, 16)).astype(np.float32)
        bench = GRNBenchmark(network_file=str(tsv), min_cells_per_gene=5)
        result = bench.run(embeddings, adata, "file_test")
        assert result.benchmark_name == "grn_inference"

    def test_registry_registered(self):
        from scmodelforge.eval.registry import list_benchmarks

        assert "grn_inference" in list_benchmarks()
