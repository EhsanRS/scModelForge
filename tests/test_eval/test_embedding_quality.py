"""Tests for eval.embedding_quality — EmbeddingQualityBenchmark."""

from __future__ import annotations

import pytest

scib = pytest.importorskip("scib", reason="scib not installed")

from scmodelforge.eval.base import BenchmarkResult  # noqa: E402
from scmodelforge.eval.embedding_quality import EmbeddingQualityBenchmark  # noqa: E402


class TestEmbeddingQualityBenchmark:
    """Tests for EmbeddingQualityBenchmark (requires scib)."""

    def test_name(self):
        bench = EmbeddingQualityBenchmark()
        assert bench.name == "embedding_quality"

    def test_required_obs_keys_with_batch(self):
        bench = EmbeddingQualityBenchmark(cell_type_key="ct", batch_key="b")
        assert "ct" in bench.required_obs_keys
        assert "b" in bench.required_obs_keys

    def test_required_obs_keys_without_batch(self):
        bench = EmbeddingQualityBenchmark(batch_key=None)
        assert "batch" not in bench.required_obs_keys

    def test_run_returns_benchmark_result(self, tiny_adata, synthetic_embeddings):
        bench = EmbeddingQualityBenchmark(
            cell_type_key="cell_type", batch_key="batch", n_neighbors=5,
        )
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "embedding_quality"

    def test_run_contains_bio_metrics(self, tiny_adata, synthetic_embeddings):
        bench = EmbeddingQualityBenchmark(
            cell_type_key="cell_type", batch_key="batch", n_neighbors=5,
        )
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert "nmi" in result.metrics
        assert "ari" in result.metrics
        assert "asw_cell_type" in result.metrics
        assert "overall" in result.metrics

    def test_run_contains_batch_metrics(self, tiny_adata, synthetic_embeddings):
        bench = EmbeddingQualityBenchmark(
            cell_type_key="cell_type", batch_key="batch", n_neighbors=5,
        )
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert "asw_batch" in result.metrics
        assert "graph_connectivity" in result.metrics

    def test_run_without_batch_key(self, tiny_adata, synthetic_embeddings):
        bench = EmbeddingQualityBenchmark(
            cell_type_key="cell_type", batch_key=None, n_neighbors=5,
        )
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert "nmi" in result.metrics
        assert "asw_batch" not in result.metrics
        # Overall should equal bio_mean when no batch metrics
        assert "overall" in result.metrics

    def test_does_not_mutate_adata(self, tiny_adata, synthetic_embeddings):
        original_keys = set(tiny_adata.obsm.keys())
        bench = EmbeddingQualityBenchmark(
            cell_type_key="cell_type", batch_key="batch", n_neighbors=5,
        )
        bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        # Original adata should not have new obsm keys
        assert set(tiny_adata.obsm.keys()) == original_keys

    def test_metadata_populated(self, tiny_adata, synthetic_embeddings):
        bench = EmbeddingQualityBenchmark(
            cell_type_key="cell_type", batch_key="batch", n_neighbors=5,
        )
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert result.metadata["n_cells"] == 40
        assert result.metadata["n_neighbors"] == 5

    def test_missing_obs_key_raises(self, tiny_adata, synthetic_embeddings):
        bench = EmbeddingQualityBenchmark(cell_type_key="nonexistent")
        with pytest.raises(ValueError, match="nonexistent"):
            bench.run(synthetic_embeddings, tiny_adata, "test_ds")
