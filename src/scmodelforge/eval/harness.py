"""Evaluation harness — orchestrates benchmarks across datasets."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scmodelforge.eval._utils import extract_embeddings

if TYPE_CHECKING:
    import numpy as np
    import torch.nn as nn
    from anndata import AnnData

    from scmodelforge.config.schema import EvalConfig
    from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
    from scmodelforge.tokenizers.base import BaseTokenizer
    from scmodelforge.zoo.base import BaseModelAdapter

logger = logging.getLogger(__name__)


class EvalHarness:
    """Orchestrates evaluation benchmarks.

    Extracts embeddings once per dataset, then runs all benchmarks.

    Parameters
    ----------
    benchmarks
        List of benchmark instances to run.
    """

    def __init__(self, benchmarks: list[BaseBenchmark]) -> None:
        self.benchmarks = list(benchmarks)

    @classmethod
    def from_config(cls, config: EvalConfig) -> EvalHarness:
        """Create an :class:`EvalHarness` from an :class:`EvalConfig`.

        Instantiates benchmarks from the registry using the names in
        ``config.benchmarks``.  Falls back to ``["linear_probe"]`` if
        the list is empty.

        Parameters
        ----------
        config
            Evaluation configuration.

        Returns
        -------
        EvalHarness
        """
        from scmodelforge.eval.registry import get_benchmark

        benchmark_specs = config.benchmarks or ["linear_probe"]
        benchmarks: list[BaseBenchmark] = []
        for spec in benchmark_specs:
            if isinstance(spec, str):
                benchmarks.append(get_benchmark(spec))
            elif isinstance(spec, dict):
                spec = dict(spec)  # copy to avoid mutating original
                name = spec.pop("name")
                # "dataset" is routing metadata, not a constructor kwarg
                spec.pop("dataset", None)
                # Support nested {params: {...}} and flat kwargs
                params = spec.pop("params", {})
                kwargs = {**spec, **params}
                benchmarks.append(get_benchmark(name, **kwargs))
            else:
                msg = f"Invalid benchmark spec: {spec!r}. Expected str or dict."
                raise ValueError(msg)
        return cls(benchmarks)

    def run(
        self,
        model: nn.Module,
        datasets: dict[str, AnnData],
        tokenizer: BaseTokenizer,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> list[BenchmarkResult]:
        """Run all benchmarks on all datasets.

        Extracts embeddings once per dataset, then runs every benchmark.

        Parameters
        ----------
        model
            Model with an ``encode()`` method.
        datasets
            Mapping of dataset name to AnnData.
        tokenizer
            Tokenizer for embedding extraction.
        batch_size
            Batch size for embedding extraction.
        device
            Device for inference.

        Returns
        -------
        list[BenchmarkResult]
        """
        results: list[BenchmarkResult] = []
        for ds_name, adata in datasets.items():
            logger.info("Extracting embeddings for dataset '%s' (%d cells)", ds_name, adata.n_obs)
            embeddings = extract_embeddings(model, adata, tokenizer, batch_size=batch_size, device=device)
            ds_results = self.run_on_embeddings(embeddings, adata, ds_name)
            results.extend(ds_results)
        return results

    def run_on_embeddings(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> list[BenchmarkResult]:
        """Run all benchmarks on precomputed embeddings.

        Parameters
        ----------
        embeddings
            Cell embeddings of shape ``(n_cells, hidden_dim)``.
        adata
            AnnData with matching annotations.
        dataset_name
            Name for this dataset in results.

        Returns
        -------
        list[BenchmarkResult]
        """
        results: list[BenchmarkResult] = []
        for bench in self.benchmarks:
            logger.info("Running benchmark '%s' on '%s'", bench.name, dataset_name)
            result = bench.run(embeddings, adata, dataset_name)
            logger.info("  %s", result.summary())
            results.append(result)
        return results

    def run_external(
        self,
        adapter: BaseModelAdapter,
        datasets: dict[str, AnnData],
        batch_size: int = 64,
        device: str = "cpu",
    ) -> list[BenchmarkResult]:
        """Run benchmarks using an external model adapter.

        Extracts embeddings via the adapter's own pipeline, then runs all
        configured benchmarks on the result.

        Parameters
        ----------
        adapter
            External model adapter instance.
        datasets
            Mapping of dataset name to AnnData.
        batch_size
            Batch size for embedding extraction.
        device
            Device for inference.

        Returns
        -------
        list[BenchmarkResult]
        """
        results: list[BenchmarkResult] = []
        for ds_name, adata in datasets.items():
            logger.info(
                "Extracting embeddings via '%s' for dataset '%s' (%d cells)",
                adapter.info.name,
                ds_name,
                adata.n_obs,
            )
            embeddings = adapter.extract_embeddings(adata, batch_size=batch_size, device=device)
            ds_results = self.run_on_embeddings(embeddings, adata, ds_name)
            results.extend(ds_results)
        return results
