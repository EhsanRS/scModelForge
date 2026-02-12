"""Evaluation benchmarks and harnesses for single-cell models."""

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.callback import AssessmentCallback
from scmodelforge.eval.cz_benchmarks import (
    CZBatchIntegrationBenchmark,
    CZClusteringBenchmark,
    CZEmbeddingBenchmark,
    CZLabelPredictionBenchmark,
)
from scmodelforge.eval.embedding_quality import EmbeddingQualityBenchmark
from scmodelforge.eval.grn import GRNBenchmark
from scmodelforge.eval.harness import EvalHarness
from scmodelforge.eval.linear_probe import LinearProbeBenchmark
from scmodelforge.eval.perturbation import PerturbationBenchmark
from scmodelforge.eval.registry import get_benchmark, list_benchmarks, register_benchmark

__all__ = [
    "AssessmentCallback",
    "BaseBenchmark",
    "BenchmarkResult",
    "CZBatchIntegrationBenchmark",
    "CZClusteringBenchmark",
    "CZEmbeddingBenchmark",
    "CZLabelPredictionBenchmark",
    "EmbeddingQualityBenchmark",
    "EvalHarness",
    "GRNBenchmark",
    "LinearProbeBenchmark",
    "PerturbationBenchmark",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
]
