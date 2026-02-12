"""Tests for eval.base — BenchmarkResult and BaseBenchmark ABC."""

from __future__ import annotations

import numpy as np
import pytest

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation(self):
        result = BenchmarkResult(
            benchmark_name="test_bench",
            dataset_name="test_ds",
            metrics={"acc": 0.95, "f1": 0.90},
        )
        assert result.benchmark_name == "test_bench"
        assert result.dataset_name == "test_ds"
        assert result.metrics["acc"] == 0.95
        assert result.metadata == {}

    def test_creation_with_metadata(self):
        result = BenchmarkResult(
            benchmark_name="test_bench",
            dataset_name="test_ds",
            metrics={"acc": 0.95},
            metadata={"n_cells": 100},
        )
        assert result.metadata["n_cells"] == 100

    def test_to_dict(self):
        result = BenchmarkResult(
            benchmark_name="test_bench",
            dataset_name="test_ds",
            metrics={"acc": 0.95},
            metadata={"n": 10},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["benchmark_name"] == "test_bench"
        assert d["dataset_name"] == "test_ds"
        assert d["metrics"] == {"acc": 0.95}
        assert d["metadata"] == {"n": 10}

    def test_to_dict_returns_copies(self):
        metrics = {"acc": 0.95}
        result = BenchmarkResult(
            benchmark_name="b", dataset_name="d", metrics=metrics,
        )
        d = result.to_dict()
        d["metrics"]["acc"] = 0.0
        # Original should be unchanged
        assert result.metrics["acc"] == 0.95

    def test_summary(self):
        result = BenchmarkResult(
            benchmark_name="test_bench",
            dataset_name="test_ds",
            metrics={"acc": 0.9512, "f1": 0.9023},
        )
        s = result.summary()
        assert "test_bench" in s
        assert "test_ds" in s
        assert "acc=0.9512" in s
        assert "f1=0.9023" in s

    def test_summary_empty_metrics(self):
        result = BenchmarkResult(
            benchmark_name="b", dataset_name="d", metrics={},
        )
        s = result.summary()
        assert "b" in s
        assert "d" in s


class TestBaseBenchmark:
    """Tests for BaseBenchmark ABC."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseBenchmark()

    def test_concrete_subclass(self, tiny_adata):
        class DummyBenchmark(BaseBenchmark):
            @property
            def name(self):
                return "dummy"

            @property
            def required_obs_keys(self):
                return ["cell_type"]

            def run(self, embeddings, adata, dataset_name):
                return BenchmarkResult(
                    benchmark_name=self.name,
                    dataset_name=dataset_name,
                    metrics={"score": 1.0},
                )

        bench = DummyBenchmark()
        assert bench.name == "dummy"
        assert bench.required_obs_keys == ["cell_type"]

        emb = np.random.default_rng(0).standard_normal((40, 32)).astype(np.float32)
        result = bench.run(emb, tiny_adata, "test")
        assert result.benchmark_name == "dummy"
        assert result.metrics["score"] == 1.0

    def test_validate_adata_passes(self, tiny_adata):
        class DummyBenchmark(BaseBenchmark):
            @property
            def name(self):
                return "dummy"

            @property
            def required_obs_keys(self):
                return ["cell_type", "batch"]

            def run(self, embeddings, adata, dataset_name):
                return BenchmarkResult("d", "d", {})

        bench = DummyBenchmark()
        bench.validate_adata(tiny_adata)  # should not raise

    def test_validate_adata_fails(self, tiny_adata):
        class DummyBenchmark(BaseBenchmark):
            @property
            def name(self):
                return "dummy"

            @property
            def required_obs_keys(self):
                return ["cell_type", "nonexistent_key"]

            def run(self, embeddings, adata, dataset_name):
                return BenchmarkResult("d", "d", {})

        bench = DummyBenchmark()
        with pytest.raises(ValueError, match="nonexistent_key"):
            bench.validate_adata(tiny_adata)

    def test_must_implement_all_abstract_methods(self):
        with pytest.raises(TypeError):

            class IncompleteBenchmark(BaseBenchmark):
                @property
                def name(self):
                    return "incomplete"

            IncompleteBenchmark()
