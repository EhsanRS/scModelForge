"""Tests for eval.registry — benchmark registration."""

from __future__ import annotations

import pytest

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.registry import _REGISTRY, get_benchmark, list_benchmarks, register_benchmark


class TestBenchmarkRegistry:
    """Tests for benchmark registration and retrieval."""

    def test_list_benchmarks_includes_builtin(self):
        names = list_benchmarks()
        assert "linear_probe" in names

    def test_get_benchmark_linear_probe(self):
        bench = get_benchmark("linear_probe")
        assert bench.name == "linear_probe"

    def test_get_benchmark_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("nonexistent_benchmark")

    def test_get_benchmark_with_kwargs(self):
        bench = get_benchmark("linear_probe", cell_type_key="my_key", seed=123)
        assert bench._cell_type_key == "my_key"
        assert bench._seed == 123

    def test_register_new_benchmark(self):
        name = "_test_registry_bench"
        try:

            @register_benchmark(name)
            class TestBench(BaseBenchmark):
                @property
                def name(self):
                    return name

                @property
                def required_obs_keys(self):
                    return []

                def run(self, embeddings, adata, dataset_name):
                    return BenchmarkResult(self.name, dataset_name, {"x": 1.0})

            assert name in list_benchmarks()
            bench = get_benchmark(name)
            assert bench.name == name
        finally:
            _REGISTRY.pop(name, None)

    def test_register_duplicate_raises(self):
        name = "_test_dup_bench"
        try:

            @register_benchmark(name)
            class First(BaseBenchmark):
                @property
                def name(self):
                    return name

                @property
                def required_obs_keys(self):
                    return []

                def run(self, embeddings, adata, dataset_name):
                    return BenchmarkResult(self.name, dataset_name, {})

            with pytest.raises(ValueError, match="already registered"):

                @register_benchmark(name)
                class Second(BaseBenchmark):
                    @property
                    def name(self):
                        return name

                    @property
                    def required_obs_keys(self):
                        return []

                    def run(self, embeddings, adata, dataset_name):
                        return BenchmarkResult(self.name, dataset_name, {})

        finally:
            _REGISTRY.pop(name, None)

    def test_list_benchmarks_sorted(self):
        names = list_benchmarks()
        assert names == sorted(names)

    def test_builtin_benchmarks_registered(self):
        names = list_benchmarks()
        assert "linear_probe" in names
        assert "embedding_quality" in names
