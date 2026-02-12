"""Embedding quality benchmark — scIB-style bio + batch metrics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.registry import register_benchmark

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData

logger = logging.getLogger(__name__)


@register_benchmark("embedding_quality")
class EmbeddingQualityBenchmark(BaseBenchmark):
    """Evaluate embedding quality using scIB metrics.

    Computes NMI, ARI, ASW for cell-type (bio) and optionally ASW and
    graph connectivity for batch correction.  Overall score is
    ``0.6 * bio_mean + 0.4 * batch_mean``.

    Parameters
    ----------
    cell_type_key
        Column in ``adata.obs`` for cell-type labels.
    batch_key
        Column in ``adata.obs`` for batch labels.  Set to ``None``
        to skip batch metrics.
    n_neighbors
        Number of neighbours for the scanpy neighbourhood graph.
    """

    def __init__(
        self,
        cell_type_key: str = "cell_type",
        batch_key: str | None = "batch",
        n_neighbors: int = 15,
    ) -> None:
        self._cell_type_key = cell_type_key
        self._batch_key = batch_key
        self._n_neighbors = n_neighbors

    @property
    def name(self) -> str:
        return "embedding_quality"

    @property
    def required_obs_keys(self) -> list[str]:
        keys = [self._cell_type_key]
        if self._batch_key is not None:
            keys.append(self._batch_key)
        return keys

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        """Run the embedding quality benchmark.

        Parameters
        ----------
        embeddings
            Cell embeddings of shape ``(n_cells, hidden_dim)``.
        adata
            AnnData with cell-type and batch annotations.
        dataset_name
            Dataset identifier for the result.

        Returns
        -------
        BenchmarkResult
        """
        try:
            import scanpy as sc
            import scib.metrics
        except ImportError as e:
            msg = (
                "scanpy and scib are required for EmbeddingQualityBenchmark. "
                "Install with: pip install 'scModelForge[eval]'"
            )
            raise ImportError(msg) from e

        self.validate_adata(adata)

        # Work on a copy to avoid mutating the user's object
        adata = adata.copy()
        adata.obsm["X_scmf"] = embeddings

        # Build neighbourhood graph
        sc.pp.neighbors(adata, use_rep="X_scmf", n_neighbors=self._n_neighbors)

        # Optimal-resolution clustering for NMI/ARI
        scib.metrics.cluster_optimal_resolution(
            adata,
            cluster_key="scmf_cluster",
            label_key=self._cell_type_key,
        )

        # Bio metrics
        nmi = float(scib.metrics.nmi(adata, cluster_key="scmf_cluster", label_key=self._cell_type_key))
        ari = float(scib.metrics.ari(adata, cluster_key="scmf_cluster", label_key=self._cell_type_key))
        asw_cell_type = float(
            scib.metrics.silhouette(adata, label_key=self._cell_type_key, embed="X_scmf")
        )
        bio_scores = [nmi, ari, asw_cell_type]

        metrics: dict[str, float] = {
            "nmi": nmi,
            "ari": ari,
            "asw_cell_type": asw_cell_type,
        }

        # Batch metrics (optional)
        batch_scores: list[float] = []
        if self._batch_key is not None and self._batch_key in adata.obs.columns:
            n_batches = adata.obs[self._batch_key].nunique()
            if n_batches > 1:
                asw_batch = float(
                    scib.metrics.silhouette_batch(
                        adata,
                        batch_key=self._batch_key,
                        label_key=self._cell_type_key,
                        embed="X_scmf",
                    )
                )
                graph_conn = float(
                    scib.metrics.graph_connectivity(adata, label_key=self._cell_type_key)
                )
                metrics["asw_batch"] = asw_batch
                metrics["graph_connectivity"] = graph_conn
                batch_scores = [asw_batch, graph_conn]

        # Overall score
        bio_mean = sum(bio_scores) / len(bio_scores) if bio_scores else 0.0
        batch_mean = sum(batch_scores) / len(batch_scores) if batch_scores else 0.0
        if batch_scores:
            metrics["overall"] = 0.6 * bio_mean + 0.4 * batch_mean
        else:
            metrics["overall"] = bio_mean

        metadata = {
            "n_cells": len(embeddings),
            "cell_type_key": self._cell_type_key,
            "batch_key": self._batch_key,
            "n_neighbors": self._n_neighbors,
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=dataset_name,
            metrics=metrics,
            metadata=metadata,
        )
