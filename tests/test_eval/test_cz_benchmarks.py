"""Tests for cz-benchmarks adapter benchmarks.

cz-benchmarks is an optional dependency so we mock it entirely using
``patch.dict("sys.modules", ...)``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from scmodelforge.eval.base import BenchmarkResult

# ---------------------------------------------------------------------------
# Mock cz-benchmarks types
# ---------------------------------------------------------------------------

class _MockMetricType(Enum):
    """Mirrors ``czbenchmarks.MetricType`` for the metrics we care about."""

    adjusted_rand_index = "adjusted_rand_index"
    normalized_mutual_info = "normalized_mutual_info"
    silhouette_score = "silhouette_score"
    mean_fold_accuracy = "mean_fold_accuracy"
    mean_fold_f1 = "mean_fold_f1"
    mean_fold_precision = "mean_fold_precision"
    mean_fold_recall = "mean_fold_recall"
    mean_fold_auroc = "mean_fold_auroc"
    entropy_per_cell = "entropy_per_cell"
    batch_silhouette = "batch_silhouette"


@dataclass
class _MockMetricResult:
    """Mirrors ``czbenchmarks.MetricResult``."""

    metric_type: _MockMetricType
    value: float


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _mock_czbenchmarks():
    """Patch ``sys.modules`` so cz-benchmarks imports succeed with mocks.

    Yields a dict of the mock task classes for assertions.
    """
    # Build mock modules
    mock_cz = ModuleType("czbenchmarks")
    mock_tasks = ModuleType("czbenchmarks.tasks")

    # --- ClusteringTask ---
    mock_clustering_task_cls = MagicMock()
    mock_clustering_task_instance = MagicMock()
    mock_clustering_task_instance.run.return_value = [
        _MockMetricResult(_MockMetricType.adjusted_rand_index, 0.85),
        _MockMetricResult(_MockMetricType.normalized_mutual_info, 0.90),
    ]
    mock_clustering_task_cls.return_value = mock_clustering_task_instance

    # --- EmbeddingTask ---
    mock_embedding_task_cls = MagicMock()
    mock_embedding_task_instance = MagicMock()
    mock_embedding_task_instance.run.return_value = [
        _MockMetricResult(_MockMetricType.silhouette_score, 0.72),
    ]
    mock_embedding_task_cls.return_value = mock_embedding_task_instance

    # --- MetadataLabelPredictionTask ---
    mock_label_pred_task_cls = MagicMock()
    mock_label_pred_task_instance = MagicMock()
    mock_label_pred_task_instance.run.return_value = [
        _MockMetricResult(_MockMetricType.mean_fold_accuracy, 0.91),
        _MockMetricResult(_MockMetricType.mean_fold_f1, 0.89),
        _MockMetricResult(_MockMetricType.mean_fold_precision, 0.88),
        _MockMetricResult(_MockMetricType.mean_fold_recall, 0.87),
        _MockMetricResult(_MockMetricType.mean_fold_auroc, 0.95),
    ]
    mock_label_pred_task_cls.return_value = mock_label_pred_task_instance

    # --- BatchIntegrationTask ---
    mock_batch_task_cls = MagicMock()
    mock_batch_task_instance = MagicMock()
    mock_batch_task_instance.run.return_value = [
        _MockMetricResult(_MockMetricType.entropy_per_cell, 0.78),
        _MockMetricResult(_MockMetricType.batch_silhouette, 0.65),
    ]
    mock_batch_task_cls.return_value = mock_batch_task_instance

    # Wire up mock_tasks attributes
    mock_tasks.ClusteringTask = mock_clustering_task_cls
    mock_tasks.ClusteringTaskInput = MagicMock()
    mock_tasks.EmbeddingTask = mock_embedding_task_cls
    mock_tasks.EmbeddingTaskInput = MagicMock()
    mock_tasks.MetadataLabelPredictionTask = mock_label_pred_task_cls
    mock_tasks.MetadataLabelPredictionTaskInput = MagicMock()
    mock_tasks.BatchIntegrationTask = mock_batch_task_cls
    mock_tasks.BatchIntegrationTaskInput = MagicMock()

    # Wire parent module
    mock_cz.tasks = mock_tasks

    modules_patch = {
        "czbenchmarks": mock_cz,
        "czbenchmarks.tasks": mock_tasks,
    }

    with patch.dict(sys.modules, modules_patch):
        yield {
            "ClusteringTask": mock_clustering_task_cls,
            "EmbeddingTask": mock_embedding_task_cls,
            "MetadataLabelPredictionTask": mock_label_pred_task_cls,
            "BatchIntegrationTask": mock_batch_task_cls,
        }


# ---------------------------------------------------------------------------
# CZClusteringBenchmark
# ---------------------------------------------------------------------------

class TestCZClusteringBenchmark:
    def test_name(self):
        from scmodelforge.eval.cz_benchmarks import CZClusteringBenchmark

        bench = CZClusteringBenchmark()
        assert bench.name == "cz_clustering"

    def test_required_obs_keys(self):
        from scmodelforge.eval.cz_benchmarks import CZClusteringBenchmark

        bench = CZClusteringBenchmark(label_key="custom_label")
        assert bench.required_obs_keys == ["custom_label"]

    def test_run_returns_benchmark_result(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        from scmodelforge.eval.cz_benchmarks import CZClusteringBenchmark

        bench = CZClusteringBenchmark()
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "cz_clustering"
        assert result.dataset_name == "test_ds"

    def test_metrics_from_cz_task(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        from scmodelforge.eval.cz_benchmarks import CZClusteringBenchmark

        bench = CZClusteringBenchmark()
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert result.metrics["adjusted_rand_index"] == pytest.approx(0.85)
        assert result.metrics["normalized_mutual_info"] == pytest.approx(0.90)

    def test_metadata_populated(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        from scmodelforge.eval.cz_benchmarks import CZClusteringBenchmark

        bench = CZClusteringBenchmark()
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert result.metadata["n_cells"] == 40
        assert result.metadata["label_key"] == "cell_type"

    def test_missing_obs_key_raises(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        from scmodelforge.eval.cz_benchmarks import CZClusteringBenchmark

        bench = CZClusteringBenchmark(label_key="nonexistent_key")
        with pytest.raises(ValueError, match="missing"):
            bench.run(synthetic_embeddings, tiny_adata, "test_ds")


# ---------------------------------------------------------------------------
# CZEmbeddingBenchmark
# ---------------------------------------------------------------------------

class TestCZEmbeddingBenchmark:
    def test_name(self):
        from scmodelforge.eval.cz_benchmarks import CZEmbeddingBenchmark

        bench = CZEmbeddingBenchmark()
        assert bench.name == "cz_embedding"

    def test_run_returns_silhouette(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        from scmodelforge.eval.cz_benchmarks import CZEmbeddingBenchmark

        bench = CZEmbeddingBenchmark()
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert isinstance(result, BenchmarkResult)
        assert result.metrics["silhouette_score"] == pytest.approx(0.72)

    def test_metadata(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        from scmodelforge.eval.cz_benchmarks import CZEmbeddingBenchmark

        bench = CZEmbeddingBenchmark()
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert result.metadata["label_key"] == "cell_type"
        assert result.metadata["n_cells"] == 40


# ---------------------------------------------------------------------------
# CZLabelPredictionBenchmark
# ---------------------------------------------------------------------------

class TestCZLabelPredictionBenchmark:
    def test_name(self):
        from scmodelforge.eval.cz_benchmarks import CZLabelPredictionBenchmark

        bench = CZLabelPredictionBenchmark()
        assert bench.name == "cz_label_prediction"

    def test_run_returns_metrics(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        from scmodelforge.eval.cz_benchmarks import CZLabelPredictionBenchmark

        bench = CZLabelPredictionBenchmark()
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert isinstance(result, BenchmarkResult)
        assert result.metrics["mean_fold_accuracy"] == pytest.approx(0.91)
        assert result.metrics["mean_fold_f1"] == pytest.approx(0.89)
        assert result.metrics["mean_fold_precision"] == pytest.approx(0.88)
        assert result.metrics["mean_fold_recall"] == pytest.approx(0.87)
        assert result.metrics["mean_fold_auroc"] == pytest.approx(0.95)

    def test_custom_label_key(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        # Add a custom obs column
        tiny_adata.obs["lineage"] = tiny_adata.obs["cell_type"]

        from scmodelforge.eval.cz_benchmarks import CZLabelPredictionBenchmark

        bench = CZLabelPredictionBenchmark(label_key="lineage")
        assert bench.required_obs_keys == ["lineage"]
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert result.metadata["label_key"] == "lineage"


# ---------------------------------------------------------------------------
# CZBatchIntegrationBenchmark
# ---------------------------------------------------------------------------

class TestCZBatchIntegrationBenchmark:
    def test_name(self):
        from scmodelforge.eval.cz_benchmarks import CZBatchIntegrationBenchmark

        bench = CZBatchIntegrationBenchmark()
        assert bench.name == "cz_batch_integration"

    def test_required_obs_keys_both(self):
        from scmodelforge.eval.cz_benchmarks import CZBatchIntegrationBenchmark

        bench = CZBatchIntegrationBenchmark(label_key="ct", batch_key="donor")
        assert bench.required_obs_keys == ["ct", "donor"]

    def test_run_returns_metrics(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        from scmodelforge.eval.cz_benchmarks import CZBatchIntegrationBenchmark

        bench = CZBatchIntegrationBenchmark()
        result = bench.run(synthetic_embeddings, tiny_adata, "test_ds")
        assert isinstance(result, BenchmarkResult)
        assert result.metrics["entropy_per_cell"] == pytest.approx(0.78)
        assert result.metrics["batch_silhouette"] == pytest.approx(0.65)

    def test_missing_batch_key_raises(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        from scmodelforge.eval.cz_benchmarks import CZBatchIntegrationBenchmark

        bench = CZBatchIntegrationBenchmark(batch_key="nonexistent_batch")
        with pytest.raises(ValueError, match="missing"):
            bench.run(synthetic_embeddings, tiny_adata, "test_ds")


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_import_error_without_package(self, tiny_adata, synthetic_embeddings):
        """When cz-benchmarks is not installed, run() raises ImportError."""
        from scmodelforge.eval.cz_benchmarks import CZClusteringBenchmark

        # Ensure czbenchmarks is NOT in sys.modules
        with patch.dict(sys.modules, {"czbenchmarks": None}):
            bench = CZClusteringBenchmark()
            with pytest.raises(ImportError, match="cz-benchmarks"):
                bench.run(synthetic_embeddings, tiny_adata, "test_ds")


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_all_four_registered(self):
        from scmodelforge.eval.registry import list_benchmarks

        names = list_benchmarks()
        assert "cz_clustering" in names
        assert "cz_embedding" in names
        assert "cz_label_prediction" in names
        assert "cz_batch_integration" in names

    def test_get_benchmark_with_kwargs(self):
        from scmodelforge.eval.registry import get_benchmark

        bench = get_benchmark("cz_clustering", label_key="custom")
        assert bench.required_obs_keys == ["custom"]


# ---------------------------------------------------------------------------
# Harness integration
# ---------------------------------------------------------------------------

class TestHarnessIntegration:
    def test_from_config_dict_spec(self, _mock_czbenchmarks, tiny_adata, synthetic_embeddings):
        """YAML-style dict spec works with EvalHarness.from_config()."""
        from scmodelforge.config.schema import EvalConfig
        from scmodelforge.eval.harness import EvalHarness

        config = EvalConfig(
            every_n_epochs=1,
            batch_size=16,
            benchmarks=[
                "cz_clustering",
                {"name": "cz_batch_integration", "label_key": "cell_type", "batch_key": "batch"},
            ],
        )
        harness = EvalHarness.from_config(config)
        assert len(harness.benchmarks) == 2
        assert harness.benchmarks[0].name == "cz_clustering"
        assert harness.benchmarks[1].name == "cz_batch_integration"

        # Actually run them
        results = harness.run_on_embeddings(synthetic_embeddings, tiny_adata, "test_ds")
        assert len(results) == 2
        assert all(isinstance(r, BenchmarkResult) for r in results)
