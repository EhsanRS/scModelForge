"""Adapter benchmarks wrapping cz-benchmarks tasks.

Provides thin wrappers around CZI's standardized evaluation framework
(``cz-benchmarks`` on PyPI) so that its tasks integrate seamlessly with
scModelForge's :class:`~scmodelforge.eval.harness.EvalHarness`,
:class:`~scmodelforge.eval.callback.AssessmentCallback`, and CLI.

Each adapter:
- Extends :class:`BaseBenchmark` (``name``, ``required_obs_keys``, ``run()``)
- Is registered via ``@register_benchmark("cz_*")``
- Lazy-imports ``czbenchmarks`` inside ``run()`` for a clear install error

Install::

    pip install cz-benchmarks
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.registry import register_benchmark

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _require_czbenchmarks() -> None:
    """Raise a clear ``ImportError`` if ``cz-benchmarks`` is not installed."""
    try:
        import czbenchmarks  # noqa: F401
    except ImportError as exc:
        msg = (
            "cz-benchmarks is required for CZ benchmark adapters. "
            "Install with: pip install 'cz-benchmarks>=0.9'"
        )
        raise ImportError(msg) from exc


# ---------------------------------------------------------------------------
# Result conversion helper
# ---------------------------------------------------------------------------

def _convert_results(
    results: list[Any],
    benchmark_name: str,
    dataset_name: str,
    n_cells: int,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Convert a list of cz-benchmarks ``MetricResult`` to :class:`BenchmarkResult`.

    Parameters
    ----------
    results
        List of ``czbenchmarks.MetricResult`` objects.
    benchmark_name
        Name for the benchmark result.
    dataset_name
        Dataset identifier.
    n_cells
        Number of cells evaluated.
    metadata
        Extra metadata to include.
    """
    metrics: dict[str, float] = {}
    for r in results:
        metrics[r.metric_type.value] = float(r.value)

    meta: dict[str, Any] = {"n_cells": n_cells}
    if metadata:
        meta.update(metadata)

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        dataset_name=dataset_name,
        metrics=metrics,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Adapter benchmarks
# ---------------------------------------------------------------------------

@register_benchmark("cz_clustering")
class CZClusteringBenchmark(BaseBenchmark):
    """Clustering quality via ``czbenchmarks.tasks.ClusteringTask``.

    Metrics: ``adjusted_rand_index``, ``normalized_mutual_info``.

    Parameters
    ----------
    label_key
        Column in ``adata.obs`` with cell-type labels.
    """

    def __init__(self, label_key: str = "cell_type") -> None:
        self._label_key = label_key

    @property
    def name(self) -> str:
        return "cz_clustering"

    @property
    def required_obs_keys(self) -> list[str]:
        return [self._label_key]

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        _require_czbenchmarks()
        from czbenchmarks.tasks import ClusteringTask, ClusteringTaskInput

        self.validate_adata(adata)

        task_input = ClusteringTaskInput(input_labels=adata.obs[self._label_key].values)
        task = ClusteringTask()
        results = task.run(cell_representation=embeddings, task_input=task_input)

        return _convert_results(
            results,
            benchmark_name=self.name,
            dataset_name=dataset_name,
            n_cells=len(embeddings),
            metadata={"label_key": self._label_key},
        )


@register_benchmark("cz_embedding")
class CZEmbeddingBenchmark(BaseBenchmark):
    """Embedding quality via ``czbenchmarks.tasks.EmbeddingTask``.

    Metrics: ``silhouette_score``.

    Parameters
    ----------
    label_key
        Column in ``adata.obs`` with cell-type labels.
    """

    def __init__(self, label_key: str = "cell_type") -> None:
        self._label_key = label_key

    @property
    def name(self) -> str:
        return "cz_embedding"

    @property
    def required_obs_keys(self) -> list[str]:
        return [self._label_key]

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        _require_czbenchmarks()
        from czbenchmarks.tasks import EmbeddingTask, EmbeddingTaskInput

        self.validate_adata(adata)

        task_input = EmbeddingTaskInput(input_labels=adata.obs[self._label_key].values)
        task = EmbeddingTask()
        results = task.run(cell_representation=embeddings, task_input=task_input)

        return _convert_results(
            results,
            benchmark_name=self.name,
            dataset_name=dataset_name,
            n_cells=len(embeddings),
            metadata={"label_key": self._label_key},
        )


@register_benchmark("cz_label_prediction")
class CZLabelPredictionBenchmark(BaseBenchmark):
    """Label prediction via ``czbenchmarks.tasks.MetadataLabelPredictionTask``.

    Metrics: ``mean_fold_accuracy``, ``mean_fold_f1``, ``mean_fold_precision``,
    ``mean_fold_recall``, ``mean_fold_auroc``.

    Parameters
    ----------
    label_key
        Column in ``adata.obs`` with cell-type labels.
    """

    def __init__(self, label_key: str = "cell_type") -> None:
        self._label_key = label_key

    @property
    def name(self) -> str:
        return "cz_label_prediction"

    @property
    def required_obs_keys(self) -> list[str]:
        return [self._label_key]

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        _require_czbenchmarks()
        from czbenchmarks.tasks import MetadataLabelPredictionTask, MetadataLabelPredictionTaskInput

        self.validate_adata(adata)

        task_input = MetadataLabelPredictionTaskInput(labels=adata.obs[self._label_key].values)
        task = MetadataLabelPredictionTask()
        results = task.run(cell_representation=embeddings, task_input=task_input)

        return _convert_results(
            results,
            benchmark_name=self.name,
            dataset_name=dataset_name,
            n_cells=len(embeddings),
            metadata={"label_key": self._label_key},
        )


@register_benchmark("cz_batch_integration")
class CZBatchIntegrationBenchmark(BaseBenchmark):
    """Batch integration quality via ``czbenchmarks.tasks.BatchIntegrationTask``.

    Metrics: ``entropy_per_cell``, ``batch_silhouette``.

    Parameters
    ----------
    label_key
        Column in ``adata.obs`` with cell-type labels.
    batch_key
        Column in ``adata.obs`` with batch labels.
    """

    def __init__(self, label_key: str = "cell_type", batch_key: str = "batch") -> None:
        self._label_key = label_key
        self._batch_key = batch_key

    @property
    def name(self) -> str:
        return "cz_batch_integration"

    @property
    def required_obs_keys(self) -> list[str]:
        return [self._label_key, self._batch_key]

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        _require_czbenchmarks()
        from czbenchmarks.tasks import BatchIntegrationTask, BatchIntegrationTaskInput

        self.validate_adata(adata)

        task_input = BatchIntegrationTaskInput(
            labels=adata.obs[self._label_key].values,
            batch_labels=adata.obs[self._batch_key].values,
        )
        task = BatchIntegrationTask()
        results = task.run(cell_representation=embeddings, task_input=task_input)

        return _convert_results(
            results,
            benchmark_name=self.name,
            dataset_name=dataset_name,
            n_cells=len(embeddings),
            metadata={"label_key": self._label_key, "batch_key": self._batch_key},
        )
