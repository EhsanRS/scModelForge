"""Tests for eval.harness — EvalHarness orchestrator."""

from __future__ import annotations

import pytest

from scmodelforge.config.schema import EvalConfig
from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.harness import EvalHarness
from scmodelforge.eval.linear_probe import LinearProbeBenchmark


class _DummyBenchmark(BaseBenchmark):
    """Minimal benchmark for testing harness orchestration."""

    def __init__(self, bench_name: str = "dummy") -> None:
        self._name = bench_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_obs_keys(self) -> list[str]:
        return []

    def run(self, embeddings, adata, dataset_name):
        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=dataset_name,
            metrics={"score": float(embeddings.mean())},
            metadata={"n_cells": len(embeddings)},
        )


class TestEvalHarness:
    """Tests for EvalHarness."""

    def test_init(self):
        harness = EvalHarness([_DummyBenchmark()])
        assert len(harness.benchmarks) == 1

    def test_run_on_embeddings(self, tiny_adata, synthetic_embeddings):
        harness = EvalHarness([_DummyBenchmark()])
        results = harness.run_on_embeddings(synthetic_embeddings, tiny_adata, "test_ds")
        assert len(results) == 1
        assert results[0].benchmark_name == "dummy"
        assert results[0].dataset_name == "test_ds"

    def test_run_on_embeddings_multiple_benchmarks(self, tiny_adata, synthetic_embeddings):
        benchmarks = [_DummyBenchmark("bench_a"), _DummyBenchmark("bench_b")]
        harness = EvalHarness(benchmarks)
        results = harness.run_on_embeddings(synthetic_embeddings, tiny_adata, "test_ds")
        assert len(results) == 2
        assert results[0].benchmark_name == "bench_a"
        assert results[1].benchmark_name == "bench_b"

    def test_run_extracts_embeddings(self, tiny_model, tiny_adata, tiny_tokenizer):
        harness = EvalHarness([_DummyBenchmark()])
        results = harness.run(
            tiny_model,
            {"test": tiny_adata},
            tiny_tokenizer,
            batch_size=16,
        )
        assert len(results) == 1
        assert results[0].dataset_name == "test"
        assert results[0].metadata["n_cells"] == 40

    def test_run_multiple_datasets(self, tiny_model, tiny_adata, tiny_tokenizer):
        harness = EvalHarness([_DummyBenchmark()])
        datasets = {"ds1": tiny_adata, "ds2": tiny_adata}
        results = harness.run(
            tiny_model,
            datasets,
            tiny_tokenizer,
            batch_size=16,
        )
        assert len(results) == 2
        ds_names = {r.dataset_name for r in results}
        assert ds_names == {"ds1", "ds2"}

    def test_from_config_default(self):
        config = EvalConfig()
        harness = EvalHarness.from_config(config)
        # Default should fall back to linear_probe
        assert len(harness.benchmarks) == 1
        assert harness.benchmarks[0].name == "linear_probe"

    def test_from_config_with_string_list(self):
        config = EvalConfig(benchmarks=["linear_probe"])
        harness = EvalHarness.from_config(config)
        assert len(harness.benchmarks) == 1
        assert isinstance(harness.benchmarks[0], LinearProbeBenchmark)

    def test_from_config_with_dict_spec(self):
        config = EvalConfig(benchmarks=[{"name": "linear_probe", "seed": 123}])
        harness = EvalHarness.from_config(config)
        assert len(harness.benchmarks) == 1
        assert harness.benchmarks[0]._seed == 123

    def test_from_config_invalid_spec_raises(self):
        config = EvalConfig(benchmarks=[42])
        with pytest.raises(ValueError, match="Invalid benchmark spec"):
            EvalHarness.from_config(config)

    def test_from_config_unknown_benchmark_raises(self):
        config = EvalConfig(benchmarks=["nonexistent_benchmark"])
        with pytest.raises(ValueError, match="Unknown benchmark"):
            EvalHarness.from_config(config)
