# Building Custom Benchmarks

This tutorial shows how to implement custom evaluation metrics that integrate seamlessly with scModelForge's evaluation framework. Advanced users with Python and scikit-learn experience will learn to create benchmarks that work with `EvalHarness` and `AssessmentCallback`.

## Why Custom Benchmarks?

scModelForge ships with several built-in benchmarks:

- **linear_probe**: Logistic regression on embeddings for cell-type classification
- **embedding_quality**: scIB metrics for biological conservation and batch correction
- **perturbation**: Ridge regression for perturbation response prediction
- **grn_inference**: Gene regulatory network inference via cosine similarity
- **cz_\* adapters**: CZI Virtual Cells standard evaluation tasks

However, your research may require domain-specific metrics such as trajectory preservation, rare cell-type detection, or custom biological hypotheses. The benchmark registry makes adding these evaluations straightforward.

## Benchmark Architecture

The evaluation system consists of four key components:

1. **BaseBenchmark**: Abstract base class with `run(embeddings, adata, dataset_name)` method
2. **BenchmarkResult**: Dataclass holding benchmark name, dataset name, and metrics dictionary
3. **Registry**: `@register_benchmark("name")` decorator for automatic registration
4. **EvalHarness**: Orchestrator that extracts embeddings and runs multiple benchmarks

When you register a benchmark, it automatically becomes available for:

- YAML configuration files
- Command-line `scmodelforge benchmark` invocation
- `EvalHarness.from_config()` instantiation
- `AssessmentCallback` during training

## Example: Trajectory Consistency Benchmark

We will implement a benchmark that measures whether embeddings preserve pseudotime ordering. This is valuable for developmental biology datasets where cells follow differentiation trajectories.

The benchmark will:

1. Extract pseudotime annotations from AnnData
2. Build a k-nearest neighbor graph in embedding space
3. Compute average pseudotime of each cell's neighbors
4. Measure Spearman correlation between actual and neighbor-averaged pseudotime

### Step 1: Implement the Benchmark

Create `trajectory_consistency.py` in your project:

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.registry import register_benchmark

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


@register_benchmark("trajectory_consistency")
class TrajectoryConsistencyBenchmark(BaseBenchmark):
    """Evaluate whether embeddings preserve pseudotime ordering.

    Constructs a k-nearest neighbor graph in embedding space and measures
    the correlation between each cell's pseudotime and the average pseudotime
    of its neighbors. High correlation indicates that the embedding preserves
    developmental trajectories.

    Parameters
    ----------
    pseudotime_key
        Column in ``adata.obs`` containing pseudotime values.
    n_neighbors
        Number of nearest neighbors for graph construction.
    metric
        Distance metric for neighbor search (default: euclidean).
    """

    def __init__(
        self,
        pseudotime_key: str = "pseudotime",
        n_neighbors: int = 15,
        metric: str = "euclidean",
    ) -> None:
        self._pseudotime_key = pseudotime_key
        self._n_neighbors = n_neighbors
        self._metric = metric

    @property
    def name(self) -> str:
        return "trajectory_consistency"

    @property
    def required_obs_keys(self) -> list[str]:
        return [self._pseudotime_key]

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        """Run trajectory consistency evaluation.

        Parameters
        ----------
        embeddings
            Cell embeddings of shape ``(n_cells, hidden_dim)``.
        adata
            AnnData with pseudotime annotations in ``.obs``.
        dataset_name
            Dataset identifier for the result.

        Returns
        -------
        BenchmarkResult
        """
        try:
            from scipy.spatial.distance import cdist
            from scipy.stats import spearmanr
        except ImportError as e:
            msg = "scipy is required for TrajectoryConsistencyBenchmark"
            raise ImportError(msg) from e

        # Validate that pseudotime column exists
        self.validate_adata(adata)

        # Extract pseudotime values
        pseudotime = adata.obs[self._pseudotime_key].values
        if not np.issubdtype(pseudotime.dtype, np.number):
            msg = (
                f"Pseudotime column '{self._pseudotime_key}' must be numeric, "
                f"got dtype {pseudotime.dtype}"
            )
            raise ValueError(msg)

        n_cells = len(embeddings)

        # Handle edge case: too few cells
        if n_cells < self._n_neighbors + 1:
            logger.warning(
                "Dataset '%s' has only %d cells, fewer than n_neighbors=%d. "
                "Returning default metrics.",
                dataset_name, n_cells, self._n_neighbors,
            )
            return BenchmarkResult(
                benchmark_name=self.name,
                dataset_name=dataset_name,
                metrics={"spearman_r": 0.5, "p_value": 1.0},
                metadata={
                    "n_cells": n_cells,
                    "n_neighbors": self._n_neighbors,
                    "insufficient_data": True,
                },
            )

        # Compute pairwise distances
        distances = cdist(embeddings, embeddings, metric=self._metric)

        # For each cell, find k nearest neighbors (excluding self)
        neighbor_pseudotimes = []
        for i in range(n_cells):
            # Get indices of k+1 nearest neighbors (including self)
            neighbor_indices = np.argpartition(distances[i], self._n_neighbors + 1)[
                : self._n_neighbors + 1
            ]
            # Exclude self
            neighbor_indices = neighbor_indices[neighbor_indices != i][:self._n_neighbors]
            # Average pseudotime of neighbors
            neighbor_pseudotimes.append(pseudotime[neighbor_indices].mean())

        neighbor_pseudotimes = np.array(neighbor_pseudotimes)

        # Compute Spearman correlation
        corr, pval = spearmanr(pseudotime, neighbor_pseudotimes)

        logger.info(
            "Trajectory consistency on '%s': Spearman r=%.4f (p=%.2e)",
            dataset_name, corr, pval,
        )

        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=dataset_name,
            metrics={
                "spearman_r": float(corr),
                "p_value": float(pval),
            },
            metadata={
                "n_cells": n_cells,
                "n_neighbors": self._n_neighbors,
                "pseudotime_key": self._pseudotime_key,
                "metric": self._metric,
            },
        )
```

### Step 2: Register the Benchmark

The `@register_benchmark("trajectory_consistency")` decorator automatically registers your benchmark. To make it discoverable, import it in your evaluation package's `__init__.py`:

```python
# In src/scmodelforge/eval/__init__.py or your custom package
from scmodelforge.eval.trajectory_consistency import TrajectoryConsistencyBenchmark

__all__ = [
    # ... existing exports
    "TrajectoryConsistencyBenchmark",
]
```

Simply importing the module triggers registration. You can verify it worked:

```python
from scmodelforge.eval.registry import list_benchmarks

print(list_benchmarks())
# ['cz_batch_integration', 'cz_clustering', 'cz_embedding', 'cz_label_prediction',
#  'embedding_quality', 'grn_inference', 'linear_probe', 'perturbation',
#  'trajectory_consistency']
```

### Step 3: Use in Configuration

Add your benchmark to a YAML config file:

```yaml
eval:
  benchmarks:
    - name: linear_probe
      cell_type_key: cell_type
      test_size: 0.2

    - name: trajectory_consistency
      pseudotime_key: dpt_pseudotime
      n_neighbors: 20
      metric: cosine

  datasets:
    - path: /data/developmental_lineage.h5ad
      name: developmental
```

The `params` dictionary (with keys like `pseudotime_key`, `n_neighbors`) is passed as keyword arguments to the benchmark constructor.

### Step 4: Use with EvalHarness

Run benchmarks programmatically:

```python
from scmodelforge.config.schema import load_config
from scmodelforge.eval import EvalHarness
from scmodelforge.models.hub import load_pretrained_with_vocab
import anndata as ad

# Load model and tokenizer
model, tokenizer, vocab = load_pretrained_with_vocab("username/model-name")

# Load config
cfg = load_config("config.yaml")

# Create harness from config
harness = EvalHarness.from_config(cfg.eval)

# Load datasets
datasets = {
    "developmental": ad.read_h5ad("/data/developmental_lineage.h5ad"),
}

# Run all benchmarks
results = harness.run(model, datasets, tokenizer, batch_size=256, device="cuda")

# Print results
for result in results:
    print(result.summary())
# [linear_probe] developmental: accuracy=0.9234, f1_macro=0.9102, f1_weighted=0.9241
# [trajectory_consistency] developmental: p_value=0.0012, spearman_r=0.8234
```

### Step 5: Use with AssessmentCallback

Enable automatic evaluation during training:

```yaml
training:
  callbacks:
    - name: assessment
      every_n_epochs: 5
      datasets:
        - path: /data/developmental_lineage.h5ad
          name: developmental
      benchmarks:
        - name: trajectory_consistency
          pseudotime_key: dpt_pseudotime
```

The callback will:

1. Run all benchmarks every 5 epochs
2. Log metrics to TensorBoard/WandB as `assessment/trajectory_consistency/developmental/spearman_r`
3. Save results to disk for post-training analysis

### Step 6: Write Tests

Follow the testing pattern used for built-in benchmarks:

```python
# tests/test_custom/test_trajectory_consistency.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scmodelforge.eval.trajectory_consistency import TrajectoryConsistencyBenchmark


class TestTrajectoryConsistencyBenchmark:
    def _make_adata(self, n_cells=100, pseudotime_values=None):
        """Helper to create synthetic AnnData with pseudotime."""
        if pseudotime_values is None:
            # Linear trajectory from 0 to 1
            pseudotime_values = np.linspace(0, 1, n_cells)

        return ad.AnnData(
            X=np.random.randn(n_cells, 50).astype(np.float32),
            obs=pd.DataFrame(
                {"pseudotime": pseudotime_values},
                index=[f"cell_{i}" for i in range(n_cells)],
            ),
        )

    def test_name(self):
        bench = TrajectoryConsistencyBenchmark()
        assert bench.name == "trajectory_consistency"

    def test_required_obs_keys(self):
        bench = TrajectoryConsistencyBenchmark(pseudotime_key="dpt_pseudotime")
        assert bench.required_obs_keys == ["dpt_pseudotime"]

    def test_perfect_trajectory_preservation(self):
        """Embeddings perfectly preserve linear pseudotime."""
        n_cells = 100
        pseudotime = np.linspace(0, 1, n_cells)
        adata = self._make_adata(n_cells, pseudotime)

        # Embeddings are 1D and identical to pseudotime
        embeddings = pseudotime[:, np.newaxis].astype(np.float32)

        bench = TrajectoryConsistencyBenchmark(n_neighbors=10)
        result = bench.run(embeddings, adata, "test_dataset")

        assert result.benchmark_name == "trajectory_consistency"
        assert result.dataset_name == "test_dataset"
        # Perfect correlation
        assert result.metrics["spearman_r"] > 0.99
        assert result.metrics["p_value"] < 0.01

    def test_random_embeddings_low_correlation(self):
        """Random embeddings should not preserve pseudotime."""
        n_cells = 100
        adata = self._make_adata(n_cells)
        embeddings = np.random.randn(n_cells, 32).astype(np.float32)

        bench = TrajectoryConsistencyBenchmark(n_neighbors=15)
        result = bench.run(embeddings, adata, "random")

        # Low correlation expected
        assert -0.5 < result.metrics["spearman_r"] < 0.5

    def test_too_few_cells_graceful(self):
        """Handle datasets smaller than n_neighbors."""
        n_cells = 10
        adata = self._make_adata(n_cells)
        embeddings = np.random.randn(n_cells, 16).astype(np.float32)

        bench = TrajectoryConsistencyBenchmark(n_neighbors=15)
        result = bench.run(embeddings, adata, "small")

        assert result.metrics["spearman_r"] == 0.5
        assert result.metadata["insufficient_data"] is True

    def test_missing_pseudotime_raises(self):
        """Raise error if pseudotime column missing."""
        adata = ad.AnnData(
            X=np.random.randn(50, 20).astype(np.float32),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(50)]),
        )
        embeddings = np.random.randn(50, 16).astype(np.float32)

        bench = TrajectoryConsistencyBenchmark(pseudotime_key="missing_key")
        with pytest.raises(ValueError, match="missing_key"):
            bench.run(embeddings, adata, "test")

    def test_registry_integration(self):
        """Verify benchmark is registered."""
        from scmodelforge.eval.registry import get_benchmark, list_benchmarks

        assert "trajectory_consistency" in list_benchmarks()
        bench = get_benchmark("trajectory_consistency", n_neighbors=20)
        assert bench._n_neighbors == 20
```

Run tests:

```bash
pytest tests/test_custom/test_trajectory_consistency.py -v
```

## Best Practices

### 1. Return Meaningful Metrics

Always return a `BenchmarkResult` with descriptive metric names:

```python
# Good
metrics = {
    "spearman_r": 0.823,
    "p_value": 0.001,
}

# Avoid
metrics = {
    "score": 0.823,  # What kind of score?
}
```

### 2. Handle Edge Cases Gracefully

Log warnings and return default values rather than raising errors:

```python
if n_cells < self._n_neighbors:
    logger.warning("Dataset has too few cells for k-NN, returning defaults")
    return BenchmarkResult(
        benchmark_name=self.name,
        dataset_name=dataset_name,
        metrics={"spearman_r": 0.5, "p_value": 1.0},
        metadata={"insufficient_data": True},
    )
```

### 3. Keep Benchmarks Deterministic

Set random seeds when using stochastic operations:

```python
def run(self, embeddings, adata, dataset_name):
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(embeddings), size=1000, replace=False)
    # ... use indices for subsampling
```

### 4. Match YAML Parameter Names

Constructor parameter names should match what users pass in `params:`:

```python
# In YAML:
# params:
#   pseudotime_key: dpt_pseudotime
#   n_neighbors: 20

def __init__(
    self,
    pseudotime_key: str = "pseudotime",  # Matches YAML key
    n_neighbors: int = 15,               # Matches YAML key
):
```

### 5. Use validate_adata()

The base class provides validation for required obs keys:

```python
@property
def required_obs_keys(self) -> list[str]:
    return [self._pseudotime_key, self._batch_key]

def run(self, embeddings, adata, dataset_name):
    self.validate_adata(adata)  # Raises ValueError if keys missing
    # ... proceed safely
```

### 6. Add Informative Metadata

Include parameters and data statistics in metadata:

```python
return BenchmarkResult(
    benchmark_name=self.name,
    dataset_name=dataset_name,
    metrics={"auroc": 0.85},
    metadata={
        "n_cells": n_cells,
        "n_classes": n_classes,
        "model_type": "logistic_regression",
        "test_size": self._test_size,
    },
)
```

## What's Next?

Once you have a custom benchmark working, consider:

1. **Packaging as a plugin**: Distribute your benchmark as a separate Python package using scModelForge's plugin system
2. **Contributing upstream**: If your benchmark has broad applicability, consider contributing it to scModelForge
3. **Composing multiple metrics**: Create a composite benchmark that runs several related evaluations and aggregates results
4. **Domain-specific suites**: Build evaluation suites tailored to specific biological contexts (e.g., immunology, neuroscience)

For plugin development, see the plugin tutorial. For contributing to scModelForge, see the contributing guide.
