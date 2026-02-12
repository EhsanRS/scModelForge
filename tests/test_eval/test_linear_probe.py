"""Tests for eval.linear_probe — LinearProbeBenchmark."""

from __future__ import annotations

import pytest

from scmodelforge.eval.base import BenchmarkResult
from scmodelforge.eval.linear_probe import LinearProbeBenchmark


class TestLinearProbeBenchmark:
    """Tests for LinearProbeBenchmark."""

    def test_name(self):
        bench = LinearProbeBenchmark()
        assert bench.name == "linear_probe"

    def test_required_obs_keys(self):
        bench = LinearProbeBenchmark(cell_type_key="my_key")
        assert bench.required_obs_keys == ["my_key"]

    def test_run_returns_benchmark_result(self, tiny_adata, synthetic_embeddings):
        bench = LinearProbeBenchmark(cell_type_key="cell_type", seed=42)
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "linear_probe"
        assert result.dataset_name == "test_ds"

    def test_run_returns_expected_metrics(self, tiny_adata, synthetic_embeddings):
        bench = LinearProbeBenchmark(cell_type_key="cell_type", seed=42)
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert "accuracy" in result.metrics
        assert "f1_macro" in result.metrics
        assert "f1_weighted" in result.metrics
        # All metrics should be in [0, 1]
        for v in result.metrics.values():
            assert 0.0 <= v <= 1.0

    def test_linearly_separable_high_accuracy(self, tiny_adata, synthetic_embeddings):
        bench = LinearProbeBenchmark(cell_type_key="cell_type", seed=42)
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        # With well-separated embeddings, accuracy should be high
        assert result.metrics["accuracy"] >= 0.8

    def test_metadata_populated(self, tiny_adata, synthetic_embeddings):
        bench = LinearProbeBenchmark(cell_type_key="cell_type", seed=42)
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert result.metadata["n_cells"] == 40
        assert result.metadata["n_classes"] == 2
        assert result.metadata["test_size"] == 0.2

    def test_custom_cell_type_key(self, tiny_adata, synthetic_embeddings):
        tiny_adata.obs["custom_label"] = tiny_adata.obs["cell_type"]
        bench = LinearProbeBenchmark(cell_type_key="custom_label")
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert result.metadata["cell_type_key"] == "custom_label"

    def test_missing_obs_key_raises(self, tiny_adata, synthetic_embeddings):
        bench = LinearProbeBenchmark(cell_type_key="nonexistent")
        with pytest.raises(ValueError, match="nonexistent"):
            bench.run(synthetic_embeddings, tiny_adata, "test_ds")

    def test_deterministic(self, tiny_adata, synthetic_embeddings):
        bench = LinearProbeBenchmark(cell_type_key="cell_type", seed=42)
        r1 = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        r2 = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert r1.metrics["accuracy"] == r2.metrics["accuracy"]
