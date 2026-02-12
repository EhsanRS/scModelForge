"""Perturbation prediction benchmark — Ridge regression on cell embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.registry import register_benchmark

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


def compute_expression_deltas(
    adata: AnnData,
    perturbation_key: str,
    control_label: str,
) -> dict[str, np.ndarray]:
    """Compute mean expression delta per perturbation vs control.

    Returns ``{perturbation_label: mean(perturbed) - mean(control)}`` as dense
    1-D arrays.  Skips the control label itself.  Handles sparse ``adata.X``.

    Uses a sparse indicator-matrix multiplication to compute all group means
    in a single pass over ``adata.X``, avoiding per-label boolean masking
    that would scale as O(n_perturbations × n_cells).

    Parameters
    ----------
    adata
        AnnData with expression matrix and perturbation annotations.
    perturbation_key
        Column in ``adata.obs`` containing perturbation labels.
    control_label
        Value in ``perturbation_key`` that identifies control cells.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from perturbation label to expression delta vector.

    Raises
    ------
    ValueError
        If *control_label* is not found in ``adata.obs[perturbation_key]``.
    """
    import scipy.sparse as sp

    labels = adata.obs[perturbation_key].values
    unique_labels, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)

    if control_label not in unique_labels:
        msg = (
            f"Control label '{control_label}' not found in "
            f"adata.obs['{perturbation_key}']. Available: {list(unique_labels)}"
        )
        raise ValueError(msg)

    # Build (n_groups × n_cells) indicator matrix with 1/count weights so that
    # indicator @ X yields group means directly.  Single sparse matmul replaces
    # the per-label boolean-mask loop: O(nnz) instead of O(n_groups × n_cells).
    n_groups = len(unique_labels)
    weights = (1.0 / counts[inverse]).astype(np.float64)
    indicator = sp.csr_matrix(
        (weights, (inverse, np.arange(len(labels)))),
        shape=(n_groups, len(labels)),
    )

    X = adata.X
    raw = (indicator @ X).todense() if sp.issparse(X) else indicator @ np.asarray(X)
    group_means = np.asarray(raw)

    ctrl_idx = int(np.where(unique_labels == control_label)[0][0])
    ctrl_mean = group_means[ctrl_idx].ravel()

    deltas: dict[str, np.ndarray] = {}
    for i, label in enumerate(unique_labels):
        if label == control_label:
            continue
        deltas[str(label)] = (group_means[i].ravel() - ctrl_mean).astype(np.float64)

    return deltas


def find_top_degs(delta: np.ndarray, n_top: int) -> np.ndarray:
    """Return indices of top *n_top* genes by absolute expression change.

    Parameters
    ----------
    delta
        1-D array of expression changes.
    n_top
        Number of top differentially expressed genes to select.

    Returns
    -------
    np.ndarray
        Integer indices of top DEGs (sorted by decreasing |delta|).
    """
    n_top = min(n_top, len(delta))
    abs_delta = np.abs(delta)
    # argpartition is O(n) vs O(n log n) for full argsort — matters for 30K+ genes
    top_idx = np.argpartition(abs_delta, -n_top)[-n_top:]
    # Sort the selected indices by descending absolute value
    return top_idx[np.argsort(abs_delta[top_idx])[::-1]]


def mean_shift_baseline(train_deltas: dict[str, np.ndarray]) -> np.ndarray:
    """Compute the mean-shift baseline: average delta across all training perturbations.

    Parameters
    ----------
    train_deltas
        Mapping from perturbation label to expression delta.

    Returns
    -------
    np.ndarray
        Mean delta vector (1-D).
    """
    return np.mean(list(train_deltas.values()), axis=0)


@register_benchmark("perturbation")
class PerturbationBenchmark(BaseBenchmark):
    """Evaluate embeddings via perturbation-response prediction.

    Trains a Ridge regression on ``{mean_perturbed_embedding → expression_delta}``
    for training perturbations, then evaluates Pearson correlation, MSE, and
    direction accuracy on held-out perturbations. A mean-shift baseline is
    included for honest comparison.

    Parameters
    ----------
    perturbation_key
        Column in ``adata.obs`` with perturbation labels.
    control_label
        Value identifying control (unperturbed) cells.
    n_top_genes
        Number of top DEGs to evaluate metrics on.
    test_fraction
        Fraction of perturbations held out for testing.
    seed
        Random seed for train/test split and Ridge regression.
    """

    def __init__(
        self,
        perturbation_key: str = "perturbation",
        control_label: str = "control",
        n_top_genes: int = 50,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        self._perturbation_key = perturbation_key
        self._control_label = control_label
        self._n_top_genes = n_top_genes
        self._test_fraction = test_fraction
        self._seed = seed

    @property
    def name(self) -> str:
        return "perturbation"

    @property
    def required_obs_keys(self) -> list[str]:
        return [self._perturbation_key]

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        """Run the perturbation prediction benchmark.

        Parameters
        ----------
        embeddings
            Cell embeddings of shape ``(n_cells, hidden_dim)``.
        adata
            AnnData with perturbation annotations and expression data.
        dataset_name
            Dataset identifier for the result.

        Returns
        -------
        BenchmarkResult
        """
        from scipy.stats import pearsonr
        from sklearn.linear_model import Ridge

        self.validate_adata(adata)

        # 1. Compute expression deltas per perturbation
        deltas = compute_expression_deltas(
            adata, self._perturbation_key, self._control_label,
        )

        pert_labels = sorted(deltas.keys())
        if len(pert_labels) < 2:
            msg = (
                f"Need at least 2 non-control perturbations, got {len(pert_labels)}. "
                f"Found labels: {pert_labels}"
            )
            raise ValueError(msg)

        # 2. Compute mean embedding per perturbation.
        # Pre-compute label→row indices in one O(n_cells) pass to avoid
        # repeated O(n_cells) boolean masking per perturbation.
        obs_labels = adata.obs[self._perturbation_key].values
        label_to_idx: dict[str, list[int]] = {}
        for i, lab in enumerate(obs_labels):
            label_to_idx.setdefault(str(lab), []).append(i)
        mean_embeddings: dict[str, np.ndarray] = {}
        for label in pert_labels:
            idx = label_to_idx[label]
            mean_embeddings[label] = embeddings[idx].mean(axis=0)

        # 3. Split perturbation labels into train/test
        rng = np.random.default_rng(self._seed)
        n_test = max(1, int(len(pert_labels) * self._test_fraction))
        n_test = min(n_test, len(pert_labels) - 1)  # keep at least 1 for training
        shuffled = rng.permutation(pert_labels).tolist()
        test_labels = shuffled[:n_test]
        train_labels = shuffled[n_test:]

        # 4. Prepare training data
        X_train = np.array([mean_embeddings[lab] for lab in train_labels])
        Y_train = np.array([deltas[lab] for lab in train_labels])

        # 5. Train Ridge regression
        ridge = Ridge(alpha=1.0, random_state=self._seed)
        ridge.fit(X_train, Y_train)

        # 6. Predict for test perturbations
        X_test = np.array([mean_embeddings[lab] for lab in test_labels])
        Y_pred = ridge.predict(X_test)

        # 7. Mean-shift baseline
        train_deltas = {lab: deltas[lab] for lab in train_labels}
        baseline_pred = mean_shift_baseline(train_deltas)

        # 8. Evaluate metrics per test perturbation
        n_top = min(self._n_top_genes, adata.n_vars)

        pearson_scores: list[float] = []
        mse_scores: list[float] = []
        dir_acc_scores: list[float] = []
        baseline_pearson_scores: list[float] = []
        baseline_mse_scores: list[float] = []
        baseline_dir_acc_scores: list[float] = []
        above_baseline_count = 0

        for i, label in enumerate(test_labels):
            actual = deltas[label]
            predicted = Y_pred[i]

            # Top DEGs from actual delta
            deg_idx = find_top_degs(actual, n_top)

            actual_degs = actual[deg_idx]
            pred_degs = predicted[deg_idx]
            baseline_degs = baseline_pred[deg_idx]

            # Pearson correlation
            r_model, _ = pearsonr(actual_degs, pred_degs)
            r_baseline, _ = pearsonr(actual_degs, baseline_degs)
            pearson_scores.append(float(r_model))
            baseline_pearson_scores.append(float(r_baseline))

            # MSE
            mse_model = float(np.mean((actual_degs - pred_degs) ** 2))
            mse_baseline = float(np.mean((actual_degs - baseline_degs) ** 2))
            mse_scores.append(mse_model)
            baseline_mse_scores.append(mse_baseline)

            # Direction accuracy (fraction of DEGs with correct sign)
            dir_model = float(np.mean(np.sign(actual_degs) == np.sign(pred_degs)))
            dir_baseline = float(np.mean(np.sign(actual_degs) == np.sign(baseline_degs)))
            dir_acc_scores.append(dir_model)
            baseline_dir_acc_scores.append(dir_baseline)

            # Model beats baseline?
            if r_model > r_baseline:
                above_baseline_count += 1

        metrics = {
            "pearson_mean": float(np.mean(pearson_scores)),
            "mse_mean": float(np.mean(mse_scores)),
            "direction_accuracy": float(np.mean(dir_acc_scores)),
            "baseline_pearson_mean": float(np.mean(baseline_pearson_scores)),
            "baseline_mse_mean": float(np.mean(baseline_mse_scores)),
            "baseline_direction_accuracy": float(np.mean(baseline_dir_acc_scores)),
            "fraction_above_baseline": float(above_baseline_count / len(test_labels)),
        }

        metadata = {
            "n_cells": len(embeddings),
            "n_perturbations": len(pert_labels),
            "n_train": len(train_labels),
            "n_test": len(test_labels),
            "n_top_genes": n_top,
            "perturbation_key": self._perturbation_key,
            "control_label": self._control_label,
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=dataset_name,
            metrics=metrics,
            metadata=metadata,
        )
