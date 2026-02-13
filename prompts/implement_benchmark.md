# Implement a New Benchmark

Follow this guide to add a new assessment benchmark to scModelForge. Every benchmark takes precomputed embeddings + an AnnData object and returns a `BenchmarkResult` with named metrics.

## Overview

**Files to create:**
- `src/scmodelforge/eval/<benchmark_name>.py` — Benchmark implementation
- `tests/test_eval/test_<benchmark_name>.py` — Tests

**Files to modify:**
- `src/scmodelforge/eval/__init__.py` — Import + `__all__`
- `docs/api/eval.md` — Documentation

## Background: Existing Benchmarks

| Name | Registry key | What it measures |
|------|-------------|-----------------|
| `LinearProbeBenchmark` | `"linear_probe"` | Cell-type classification via logistic regression (accuracy, F1) |
| `EmbeddingQualityBenchmark` | `"embedding_quality"` | scIB metrics: NMI, ARI, ASW, overall score |
| `PerturbationBenchmark` | `"perturbation"` | Perturbation prediction via Ridge regression (Pearson, MSE, direction accuracy) |

## Step 1: Implementation File

Create `src/scmodelforge/eval/<benchmark_name>.py`:

```python
"""<Description> benchmark for single-cell embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.registry import register_benchmark

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


@register_benchmark("<registry_name>")
class <BenchmarkClassName>(BaseBenchmark):
    """<One-line description>.

    Parameters
    ----------
    <param_name> : <type>
        <Description>.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        # Benchmark-specific parameters
        seed: int = 42,
    ) -> None:
        self._seed = seed
        # Store other parameters with underscore prefix (private)

    @property
    def name(self) -> str:
        """Return benchmark registry name."""
        return "<registry_name>"

    @property
    def required_obs_keys(self) -> list[str]:
        """Return obs columns this benchmark needs in the AnnData.

        These are validated by ``validate_adata()`` before ``run()`` executes.
        """
        return ["<required_column>"]  # e.g., ["cell_type"], ["perturbation"]

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        """Run the benchmark.

        Parameters
        ----------
        embeddings : np.ndarray
            Cell embeddings, shape ``(n_cells, embedding_dim)``.
        adata : AnnData
            Annotated data with obs columns matching ``required_obs_keys``.
        dataset_name : str
            Name of the dataset (used in result reporting).

        Returns
        -------
        BenchmarkResult
            Named metrics and metadata.

        Raises
        ------
        ValueError
            If required obs keys are missing from adata.
        ImportError
            If optional dependencies are not installed.
        """
        # 0. Import optional dependencies at runtime
        try:
            from sklearn.linear_model import Ridge  # example
        except ImportError as e:
            msg = "scikit-learn is required for <BenchmarkClassName>."
            raise ImportError(msg) from e

        # 1. Validate input
        self.validate_adata(adata)

        # 2. Extract labels/metadata from adata.obs
        labels = adata.obs["<required_column>"].values

        # 3. Run benchmark logic
        rng = np.random.default_rng(self._seed)
        # ... your computation here ...

        # 4. Compute metrics
        metrics = {
            "metric_1": float(value_1),
            "metric_2": float(value_2),
        }

        # 5. Optional metadata
        metadata = {
            "n_cells": len(embeddings),
            "seed": self._seed,
        }

        logger.info("%s on %s: %s", self.name, dataset_name, metrics)

        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=dataset_name,
            metrics=metrics,
            metadata=metadata,
        )
```

### Key design points

- **Optional dependencies**: Import at runtime inside `run()` with a clear `ImportError` message
- **`validate_adata()`**: Call this first — it checks `required_obs_keys` exist in `adata.obs`
- **Determinism**: Use `np.random.default_rng(self._seed)` for reproducibility
- **Metrics**: All values must be `float` — no numpy scalars, no NaN/Inf
- **Logging**: Use `logger.info()` to report results
- **`BenchmarkResult`**: Standard output that works with `EvalHarness` and `AssessmentCallback`

### How benchmarks integrate

```
EvalHarness.run()
  → extract_embeddings(model, dataset)  # model.encode() on all cells
  → for each benchmark:
      benchmark.run(embeddings, adata, dataset_name)
      → BenchmarkResult

AssessmentCallback (Lightning)
  → on_validation_epoch_end
  → harness.run_on_embeddings(embeddings, adata)
  → logs metrics as assessment/{bench_name}/{dataset}/{metric}
```

## Step 2: Register in `__init__.py`

Add to `src/scmodelforge/eval/__init__.py`:

```python
# Add import (alphabetical order)
from scmodelforge.eval.<benchmark_name> import <BenchmarkClassName>

# Add to __all__ (alphabetical order)
__all__ = [
    ...
    "<BenchmarkClassName>",
    ...
]
```

## Step 3: Tests

Create `tests/test_eval/test_<benchmark_name>.py`:

```python
"""Tests for <BenchmarkClassName>."""

from __future__ import annotations

import numpy as np
import pytest

from scmodelforge.eval.<benchmark_name> import <BenchmarkClassName>
from scmodelforge.eval.base import BenchmarkResult


@pytest.fixture()
def benchmark_adata():
    """Create a synthetic AnnData for testing.

    Adjust cell count, gene count, and obs columns for your benchmark.
    """
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(42)
    n_cells, n_genes = 40, 30
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "<required_column>": ["label_A"] * 20 + ["label_B"] * 20,
            # Add other columns your benchmark needs
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=X, obs=obs)


@pytest.fixture()
def benchmark_embeddings():
    """40x32 numpy array with structure matching the test adata."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((40, 32)).astype(np.float32)
    # Add structure so the benchmark can detect signal
    emb[:20, 0] += 5.0
    emb[20:, 0] -= 5.0
    return emb


class TestProperties:
    def test_name(self):
        bench = <BenchmarkClassName>()
        assert bench.name == "<registry_name>"

    def test_required_obs_keys(self):
        bench = <BenchmarkClassName>()
        assert "<required_column>" in bench.required_obs_keys


class TestRun:
    def test_returns_benchmark_result(self, benchmark_adata, benchmark_embeddings):
        bench = <BenchmarkClassName>(seed=42)
        result = bench.run(benchmark_embeddings, benchmark_adata, "test_ds")
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "<registry_name>"
        assert result.dataset_name == "test_ds"

    def test_expected_metrics_present(self, benchmark_adata, benchmark_embeddings):
        bench = <BenchmarkClassName>(seed=42)
        result = bench.run(benchmark_embeddings, benchmark_adata, "test_ds")
        expected_keys = {"metric_1", "metric_2"}  # List your metric keys
        assert expected_keys.issubset(result.metrics.keys())

    def test_metrics_are_finite_floats(self, benchmark_adata, benchmark_embeddings):
        bench = <BenchmarkClassName>(seed=42)
        result = bench.run(benchmark_embeddings, benchmark_adata, "test_ds")
        for key, val in result.metrics.items():
            assert isinstance(val, float), f"{key} is not float"
            assert np.isfinite(val), f"{key} is not finite"

    def test_deterministic(self, benchmark_adata, benchmark_embeddings):
        bench = <BenchmarkClassName>(seed=42)
        r1 = bench.run(benchmark_embeddings, benchmark_adata, "test_ds")
        r2 = bench.run(benchmark_embeddings, benchmark_adata, "test_ds")
        for key in r1.metrics:
            assert r1.metrics[key] == r2.metrics[key], f"{key} not deterministic"


class TestValidation:
    def test_missing_obs_key_raises(self, benchmark_embeddings):
        import anndata as ad
        import pandas as pd

        adata = ad.AnnData(
            X=np.zeros((40, 10)),
            obs=pd.DataFrame({"wrong_column": ["x"] * 40}),
        )
        bench = <BenchmarkClassName>()
        with pytest.raises(ValueError, match="<required_column>"):
            bench.run(benchmark_embeddings, adata, "test_ds")


class TestRegistry:
    def test_registered(self):
        from scmodelforge.eval.registry import list_benchmarks
        assert "<registry_name>" in list_benchmarks()

    def test_get_benchmark_with_kwargs(self):
        from scmodelforge.eval.registry import get_benchmark
        bench = get_benchmark("<registry_name>", seed=123)
        assert bench._seed == 123
```

## Step 4: Verification

```bash
# Lint
.venv/bin/ruff check src/scmodelforge/eval/<benchmark_name>.py tests/test_eval/test_<benchmark_name>.py

# New tests
.venv/bin/python -m pytest tests/test_eval/test_<benchmark_name>.py -v

# Assessment module regression
.venv/bin/python -m pytest tests/test_eval/ -v

# Full suite
.venv/bin/python -m pytest tests/ -v
```

## Benchmark Contract Summary

Every benchmark **must** provide:

| Member | Type | Description |
|--------|------|-------------|
| `name` | Property → `str` | Registry name |
| `required_obs_keys` | Property → `list[str]` | Obs columns needed in AnnData |
| `run(embeddings, adata, dataset_name)` | Method | Returns `BenchmarkResult` |

Inherited from `BaseBenchmark`:
- `validate_adata(adata)` — raises `ValueError` if required obs keys missing

### `BenchmarkResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `benchmark_name` | `str` | Name of the benchmark |
| `dataset_name` | `str` | Name of the dataset |
| `metrics` | `dict[str, float]` | Named metric values |
| `metadata` | `dict[str, Any]` | Optional extra info |

Methods: `to_dict()`, `summary()` (formatted string).
