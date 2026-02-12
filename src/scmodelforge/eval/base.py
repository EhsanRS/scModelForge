"""Base benchmark interface and result dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run on a single dataset.

    Attributes
    ----------
    benchmark_name
        Name of the benchmark that produced this result.
    dataset_name
        Name of the dataset that was evaluated.
    metrics
        Dictionary of metric name to float value.
    metadata
        Additional metadata (e.g. number of cells, parameters used).
    """

    benchmark_name: str
    dataset_name: str
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "dataset_name": self.dataset_name,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        metric_strs = [f"{k}={v:.4f}" for k, v in sorted(self.metrics.items())]
        return f"[{self.benchmark_name}] {self.dataset_name}: {', '.join(metric_strs)}"


class BaseBenchmark(ABC):
    """Abstract base class for evaluation benchmarks.

    Subclasses must implement :meth:`run`, :attr:`name`, and
    :attr:`required_obs_keys`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this benchmark."""
        ...

    @property
    @abstractmethod
    def required_obs_keys(self) -> list[str]:
        """AnnData ``.obs`` columns required by this benchmark."""
        ...

    @abstractmethod
    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        """Run the benchmark on precomputed embeddings.

        Parameters
        ----------
        embeddings
            Cell embeddings of shape ``(n_cells, hidden_dim)``.
        adata
            AnnData object with matching ``.obs`` annotations.
        dataset_name
            Name used to identify this dataset in results.

        Returns
        -------
        BenchmarkResult
        """
        ...

    def validate_adata(self, adata: AnnData) -> None:
        """Check that *adata* has all required ``.obs`` columns.

        Raises
        ------
        ValueError
            If any required key is missing from ``adata.obs``.
        """
        missing = [k for k in self.required_obs_keys if k not in adata.obs.columns]
        if missing:
            msg = (
                f"Benchmark '{self.name}' requires obs keys {self.required_obs_keys}, "
                f"but the following are missing: {missing}"
            )
            raise ValueError(msg)
