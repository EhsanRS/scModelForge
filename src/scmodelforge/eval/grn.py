"""Gene regulatory network inference benchmark."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.registry import register_benchmark

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_ground_truth_network(
    path: str | Path,
) -> list[tuple[str, str, float]]:
    """Load a ground-truth gene regulatory network from a TSV file.

    Expected columns: ``gene1``, ``gene2``, ``weight``.
    The file must have a header row.  Edges with ``weight > 0`` are
    treated as positives.

    Parameters
    ----------
    path
        Path to the TSV file.

    Returns
    -------
    list[tuple[str, str, float]]
        List of ``(gene1, gene2, weight)`` tuples.
    """
    import csv
    from pathlib import Path as _Path

    path = _Path(path)
    edges: list[tuple[str, str, float]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            edges.append((row["gene1"], row["gene2"], float(row["weight"])))
    return edges


def compute_gene_representations(
    embeddings: np.ndarray,
    adata: AnnData,
    min_cells: int = 10,
) -> dict[str, np.ndarray]:
    """Compute gene-level representations as mean cell embeddings.

    For each gene in ``adata.var_names``, collect all cells where the gene
    is expressed (value > 0) and average their embeddings.

    Parameters
    ----------
    embeddings
        Cell embeddings of shape ``(n_cells, hidden_dim)``.
    adata
        AnnData with expression matrix (``adata.X``).
    min_cells
        Minimum number of expressing cells required.  Genes with fewer
        cells are excluded.

    Returns
    -------
    dict[str, np.ndarray]
        Gene name → mean embedding vector.
    """
    import scipy.sparse as sp

    X = adata.X
    is_sparse = sp.issparse(X)
    gene_reps: dict[str, np.ndarray] = {}

    for j, gene_name in enumerate(adata.var_names):
        if is_sparse:
            col = X[:, j]
            expressing_mask = np.asarray(col.toarray()).ravel() > 0
        else:
            expressing_mask = np.asarray(X[:, j]).ravel() > 0

        n_expressing = expressing_mask.sum()
        if n_expressing < min_cells:
            continue

        gene_reps[gene_name] = embeddings[expressing_mask].mean(axis=0)

    return gene_reps


def compute_gene_similarity_scores(
    gene_reps: dict[str, np.ndarray],
    gene_pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Compute cosine similarity for gene pairs.

    Parameters
    ----------
    gene_reps
        Gene name → embedding vector.
    gene_pairs
        List of ``(gene1, gene2)`` pairs to score.

    Returns
    -------
    scores
        Cosine similarity scores for evaluable pairs.
    evaluable_pairs
        The subset of *gene_pairs* where both genes have representations.
    """
    scores: list[float] = []
    evaluable: list[tuple[str, str]] = []

    for g1, g2 in gene_pairs:
        if g1 not in gene_reps or g2 not in gene_reps:
            continue
        v1 = gene_reps[g1]
        v2 = gene_reps[g2]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(v1, v2) / (norm1 * norm2)))
        evaluable.append((g1, g2))

    return np.array(scores, dtype=np.float64), evaluable


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@register_benchmark("grn_inference")
class GRNBenchmark(BaseBenchmark):
    """Gene regulatory network inference benchmark.

    Evaluates how well cell embeddings capture gene–gene regulatory
    relationships.  Gene-level representations are computed as the mean
    embedding of cells expressing each gene.  Cosine similarity between
    gene representations is used as a predicted adjacency score, which is
    compared to a ground-truth network via AUROC and AUPRC.

    Parameters
    ----------
    network_file
        Path to a TSV file with columns ``gene1``, ``gene2``, ``weight``.
        Required unless *ground_truth* is provided directly.
    ground_truth
        Pre-loaded list of ``(gene1, gene2, weight)`` edges.  Takes
        precedence over *network_file*.
    min_cells_per_gene
        Minimum expressing cells to include a gene.
    """

    def __init__(
        self,
        network_file: str | None = None,
        ground_truth: list[tuple[str, str, float]] | None = None,
        min_cells_per_gene: int = 10,
    ) -> None:
        if network_file is None and ground_truth is None:
            msg = "Either network_file or ground_truth must be provided."
            raise ValueError(msg)
        self._network_file = network_file
        self._ground_truth = ground_truth
        self._min_cells_per_gene = min_cells_per_gene

    @property
    def name(self) -> str:
        return "grn_inference"

    @property
    def required_obs_keys(self) -> list[str]:
        return []

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        """Run GRN inference evaluation."""
        from sklearn.metrics import average_precision_score, roc_auc_score

        # 1. Load ground truth
        edges = (
            self._ground_truth
            if self._ground_truth is not None
            else load_ground_truth_network(self._network_file)  # type: ignore[arg-type]
        )

        # 2. Compute gene representations
        gene_reps = compute_gene_representations(
            embeddings, adata, min_cells=self._min_cells_per_gene,
        )
        logger.info("Computed representations for %d genes", len(gene_reps))

        # 3. Build pairs and labels
        gene_pairs = [(g1, g2) for g1, g2, _ in edges]
        labels_map = {(g1, g2): (1.0 if w > 0 else 0.0) for g1, g2, w in edges}

        # 4. Compute similarity scores
        scores, evaluable_pairs = compute_gene_similarity_scores(gene_reps, gene_pairs)
        labels = np.array([labels_map[p] for p in evaluable_pairs], dtype=np.float64)

        n_edges = len(evaluable_pairs)
        n_genes = len({g for pair in evaluable_pairs for g in pair})

        # 5. Handle edge cases
        if n_edges < 2 or len(np.unique(labels)) < 2:
            warnings.warn(
                f"GRN benchmark: insufficient evaluable edges ({n_edges}) or "
                f"all same label. Returning default scores.",
                stacklevel=2,
            )
            return BenchmarkResult(
                benchmark_name=self.name,
                dataset_name=dataset_name,
                metrics={
                    "auroc": 0.5,
                    "auprc": 0.5,
                    "n_genes_evaluated": n_genes,
                    "n_edges_evaluated": n_edges,
                },
                metadata={"min_cells_per_gene": self._min_cells_per_gene},
            )

        # 6. Compute metrics
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))

        logger.info(
            "GRN benchmark on %s: AUROC=%.4f, AUPRC=%.4f (%d genes, %d edges)",
            dataset_name, auroc, auprc, n_genes, n_edges,
        )

        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=dataset_name,
            metrics={
                "auroc": auroc,
                "auprc": auprc,
                "n_genes_evaluated": float(n_genes),
                "n_edges_evaluated": float(n_edges),
            },
            metadata={"min_cells_per_gene": self._min_cells_per_gene},
        )
