"""Tests for eval.perturbation — PerturbationBenchmark."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scmodelforge.eval.base import BenchmarkResult
from scmodelforge.eval.perturbation import (
    PerturbationBenchmark,
    compute_expression_deltas,
    find_top_degs,
    mean_shift_baseline,
)
from scmodelforge.eval.registry import list_benchmarks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def perturbation_adata():
    """60 cells x 30 genes with perturbation labels.

    20 control cells, 4 perturbations x 10 cells each.
    Each perturbation upregulates a different gene subset.
    """
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(42)
    n_cells, n_genes = 60, 30

    # Baseline expression
    X = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float32)

    # Perturbation-specific upregulation
    X[20:30, 0:8] += 8.0   # pert_A: genes 0-7
    X[30:40, 8:16] += 8.0  # pert_B: genes 8-15
    X[40:50, 16:23] += 8.0  # pert_C: genes 16-22
    X[50:60, 23:30] += 8.0  # pert_D: genes 23-29

    labels = (
        ["control"] * 20
        + ["pert_A"] * 10
        + ["pert_B"] * 10
        + ["pert_C"] * 10
        + ["pert_D"] * 10
    )

    obs = pd.DataFrame(
        {"perturbation": labels},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture()
def perturbation_embeddings():
    """60x32 array with perturbation-specific shifts in embedding space."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((60, 32)).astype(np.float32)

    # Make embeddings reflect perturbation structure
    emb[20:30, 0:8] += 5.0   # pert_A
    emb[30:40, 8:16] += 5.0  # pert_B
    emb[40:50, 16:24] += 5.0  # pert_C
    emb[50:60, 24:32] += 5.0  # pert_D

    return emb


# ---------------------------------------------------------------------------
# TestComputeExpressionDeltas
# ---------------------------------------------------------------------------

class TestComputeExpressionDeltas:
    """Tests for compute_expression_deltas helper."""

    def test_correct_delta_shape(self, perturbation_adata):
        deltas = compute_expression_deltas(
            perturbation_adata, "perturbation", "control",
        )
        for delta in deltas.values():
            assert delta.shape == (30,)

    def test_control_excluded(self, perturbation_adata):
        deltas = compute_expression_deltas(
            perturbation_adata, "perturbation", "control",
        )
        assert "control" not in deltas
        assert len(deltas) == 4

    def test_sparse_handled(self, perturbation_adata):
        perturbation_adata.X = sp.csr_matrix(perturbation_adata.X)
        deltas = compute_expression_deltas(
            perturbation_adata, "perturbation", "control",
        )
        assert len(deltas) == 4
        for delta in deltas.values():
            assert delta.shape == (30,)
            assert not np.any(np.isnan(delta))

    def test_control_not_found_raises(self, perturbation_adata):
        with pytest.raises(ValueError, match="not found"):
            compute_expression_deltas(
                perturbation_adata, "perturbation", "nonexistent_ctrl",
            )


# ---------------------------------------------------------------------------
# TestFindTopDegs
# ---------------------------------------------------------------------------

class TestFindTopDegs:
    """Tests for find_top_degs helper."""

    def test_returns_correct_count(self):
        delta = np.array([0.1, -5.0, 3.0, -1.0, 0.5])
        idx = find_top_degs(delta, 3)
        assert len(idx) == 3

    def test_selects_largest_changes(self):
        delta = np.array([0.1, -5.0, 3.0, -1.0, 0.5])
        idx = find_top_degs(delta, 2)
        # Top 2 by absolute value: index 1 (|-5|=5) and index 2 (|3|=3)
        assert set(idx) == {1, 2}

    def test_caps_to_available_genes(self):
        delta = np.array([1.0, 2.0, 3.0])
        idx = find_top_degs(delta, 100)
        assert len(idx) == 3


# ---------------------------------------------------------------------------
# TestMeanShiftBaseline
# ---------------------------------------------------------------------------

class TestMeanShiftBaseline:
    """Tests for mean_shift_baseline helper."""

    def test_computes_mean_of_deltas(self):
        deltas = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([3.0, 4.0, 5.0]),
        }
        result = mean_shift_baseline(deltas)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# TestPerturbationBenchmark
# ---------------------------------------------------------------------------

class TestPerturbationBenchmark:
    """Tests for PerturbationBenchmark."""

    def test_name(self):
        bench = PerturbationBenchmark()
        assert bench.name == "perturbation"

    def test_required_obs_keys(self):
        bench = PerturbationBenchmark(perturbation_key="my_pert")
        assert bench.required_obs_keys == ["my_pert"]

    def test_run_returns_benchmark_result(
        self, perturbation_adata, perturbation_embeddings,
    ):
        bench = PerturbationBenchmark(
            perturbation_key="perturbation", control_label="control", seed=42,
        )
        result = bench.run(perturbation_embeddings, perturbation_adata, "test_ds")
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "perturbation"
        assert result.dataset_name == "test_ds"

    def test_expected_metrics_present(
        self, perturbation_adata, perturbation_embeddings,
    ):
        bench = PerturbationBenchmark(
            perturbation_key="perturbation", control_label="control", seed=42,
        )
        result = bench.run(perturbation_embeddings, perturbation_adata, "test_ds")
        expected_keys = {
            "pearson_mean",
            "mse_mean",
            "direction_accuracy",
            "baseline_pearson_mean",
            "baseline_mse_mean",
            "baseline_direction_accuracy",
            "fraction_above_baseline",
        }
        assert set(result.metrics.keys()) == expected_keys

    def test_metrics_are_finite_floats(
        self, perturbation_adata, perturbation_embeddings,
    ):
        bench = PerturbationBenchmark(
            perturbation_key="perturbation", control_label="control", seed=42,
        )
        result = bench.run(perturbation_embeddings, perturbation_adata, "test_ds")
        for key, value in result.metrics.items():
            assert isinstance(value, float), f"{key} is not a float"
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_baseline_metrics_present(
        self, perturbation_adata, perturbation_embeddings,
    ):
        bench = PerturbationBenchmark(
            perturbation_key="perturbation", control_label="control", seed=42,
        )
        result = bench.run(perturbation_embeddings, perturbation_adata, "test_ds")
        baseline_keys = [k for k in result.metrics if k.startswith("baseline_")]
        assert len(baseline_keys) == 3

    def test_missing_perturbation_key_raises(
        self, perturbation_adata, perturbation_embeddings,
    ):
        bench = PerturbationBenchmark(perturbation_key="nonexistent")
        with pytest.raises(ValueError, match="nonexistent"):
            bench.run(perturbation_embeddings, perturbation_adata, "test_ds")

    def test_missing_control_label_raises(
        self, perturbation_adata, perturbation_embeddings,
    ):
        bench = PerturbationBenchmark(
            perturbation_key="perturbation", control_label="no_such_control",
        )
        with pytest.raises(ValueError, match="no_such_control"):
            bench.run(perturbation_embeddings, perturbation_adata, "test_ds")

    def test_too_few_perturbations_raises(self, perturbation_embeddings):
        """Need at least 2 non-control perturbations."""
        import anndata as ad
        import pandas as pd

        # Only 1 perturbation + control
        n_cells, n_genes = 30, 30
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        labels = ["control"] * 20 + ["pert_A"] * 10
        obs = pd.DataFrame(
            {"perturbation": labels},
            index=[f"cell_{i}" for i in range(n_cells)],
        )
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        emb = rng.standard_normal((n_cells, 32)).astype(np.float32)

        bench = PerturbationBenchmark(
            perturbation_key="perturbation", control_label="control",
        )
        with pytest.raises(ValueError, match="at least 2"):
            bench.run(emb, adata, "test_ds")

    def test_registry_registered(self):
        assert "perturbation" in list_benchmarks()

    def test_deterministic(self, perturbation_adata, perturbation_embeddings):
        bench = PerturbationBenchmark(
            perturbation_key="perturbation", control_label="control", seed=42,
        )
        r1 = bench.run(perturbation_embeddings, perturbation_adata, "test_ds")
        r2 = bench.run(perturbation_embeddings, perturbation_adata, "test_ds")
        for key in r1.metrics:
            assert r1.metrics[key] == r2.metrics[key], f"Non-deterministic: {key}"
