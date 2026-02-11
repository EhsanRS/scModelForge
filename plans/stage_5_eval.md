# Stage 5: scModelForge.eval

## Overview

The eval module provides integrated evaluation that runs both as callbacks during training and as standalone benchmarking scripts. It wraps existing community benchmarks (scIB, PerturBench, etc.) behind a unified API, making it easy to measure model quality with standardised metrics throughout the training process.

**Core responsibility:** Model + data → standardised biological evaluation metrics.

**Design philosophy:** Evaluation is a first-class citizen, not an afterthought. Every training run should produce comparable, reproducible metrics.

**Dependencies:** Stage 0 (scaffolding), Stage 1 (data), Stage 3 (models — needs `encode()` for embeddings)
**Integrates with:** Stage 4 (training — eval callbacks run during training)

---

## Phase 1: Foundation (Months 1–3)

### Goals
- Cell embedding quality evaluation via scIB metrics
- Linear probe baseline for cell type classification
- Standardised held-out evaluation datasets (shipped or auto-downloaded)
- Evaluation as both a training callback and a standalone CLI command
- Metrics match published scIB results on reference datasets

### Architecture

```
Trained model + eval dataset
     │
     ▼
┌─────────────────────────┐
│ EvalHarness             │  ← orchestrator: runs all configured benchmarks
│  .run(model, dataset)   │
└──────────┬──────────────┘
           │
    ┌──────┼──────────┐
    │      │          │
    ▼      ▼          ▼
  scIB   LinearProbe  (Phase 2: PerturbBench, GRN)
  metrics  baseline
```

### File Structure

```
src/scmodelforge/eval/
├── __init__.py              # Public API: EvalHarness, EvalCallback, evaluate
├── harness.py               # EvalHarness orchestrator
├── callback.py              # Lightning EvalCallback (runs during training)
├── benchmarks/
│   ├── __init__.py
│   ├── base.py              # BaseBenchmark abstract class
│   ├── embedding_quality.py # scIB metrics (NMI, ASW, ARI, batch correction)
│   ├── linear_probe.py      # Linear probe for cell type classification
│   └── registry.py          # Benchmark registry
├── datasets/
│   ├── __init__.py
│   ├── manager.py           # Download and cache eval datasets
│   └── registry.py          # Dataset registry (name → download function)
├── metrics/
│   ├── __init__.py
│   ├── scib_metrics.py      # Wrapper around scIB library
│   └── classification.py    # Accuracy, F1, confusion matrix
└── _utils.py                # Embedding extraction, UMAP computation
```

### Key Classes and Interfaces

#### `BaseBenchmark` (Abstract)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """Result from a single benchmark evaluation."""
    benchmark_name: str
    dataset_name: str
    metrics: dict[str, float]      # e.g., {"NMI": 0.72, "ASW": 0.65}
    metadata: dict[str, Any]       # e.g., {"n_cells": 10000, "n_cell_types": 15}

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark_name,
            "dataset": self.dataset_name,
            **self.metrics,
        }

    def summary(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"[{self.benchmark_name}] {self.dataset_name}: {metrics_str}"


class BaseBenchmark(ABC):
    """Abstract base class for all evaluation benchmarks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""
        ...

    @property
    @abstractmethod
    def required_metadata(self) -> list[str]:
        """Metadata fields required in the eval dataset (e.g., ['cell_type', 'batch'])."""
        ...

    @abstractmethod
    def run_evaluation(
        self,
        embeddings: np.ndarray,          # (n_cells, hidden_dim) — cell embeddings from model
        adata: AnnData,                  # Original AnnData with metadata
        **kwargs,
    ) -> BenchmarkResult:
        """Run the benchmark and return results."""
        ...
```

#### `EmbeddingQualityBenchmark` (scIB)

```python
class EmbeddingQualityBenchmark(BaseBenchmark):
    """Cell embedding quality assessment using scIB metrics.

    Measures:
    - Biology conservation: NMI, ARI, ASW (cell type)
    - Batch correction: ASW_batch, graph connectivity, kBET
    - Overall: weighted combination
    """

    name = "embedding_quality"
    required_metadata = ["cell_type"]  # "batch" optional but recommended

    def __init__(
        self,
        cell_type_key: str = "cell_type",
        batch_key: str | None = "batch",
        n_neighbors: int = 15,
        compute_kbet: bool = False,  # Expensive, off by default
    ):
        self.cell_type_key = cell_type_key
        self.batch_key = batch_key
        self.n_neighbors = n_neighbors
        self.compute_kbet = compute_kbet

    def run_evaluation(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        **kwargs,
    ) -> BenchmarkResult:
        # 1. Store embeddings in adata.obsm
        adata.obsm["X_scmodelforge"] = embeddings

        # 2. Compute neighbors on embedding space
        sc.pp.neighbors(adata, use_rep="X_scmodelforge", n_neighbors=self.n_neighbors)

        # 3. Biology conservation metrics
        nmi = scib.metrics.nmi(adata, self.cell_type_key, "leiden")
        ari = scib.metrics.ari(adata, self.cell_type_key, "leiden")
        asw = scib.metrics.silhouette(adata, self.cell_type_key, "X_scmodelforge")

        metrics = {"NMI": nmi, "ARI": ari, "ASW_cell_type": asw}

        # 4. Batch correction metrics (if batch key available)
        if self.batch_key and self.batch_key in adata.obs:
            asw_batch = scib.metrics.silhouette_batch(
                adata, self.batch_key, self.cell_type_key, "X_scmodelforge"
            )
            graph_conn = scib.metrics.graph_connectivity(adata, self.cell_type_key)
            metrics["ASW_batch"] = asw_batch
            metrics["graph_connectivity"] = graph_conn

        # 5. Overall score
        metrics["overall"] = self._compute_overall(metrics)

        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=kwargs.get("dataset_name", "unknown"),
            metrics=metrics,
            metadata={"n_cells": len(adata), "n_cell_types": adata.obs[self.cell_type_key].nunique()},
        )

    def _compute_overall(self, metrics: dict[str, float]) -> float:
        """Weighted combination of bio conservation and batch correction."""
        bio_scores = [metrics.get(k, 0) for k in ["NMI", "ARI", "ASW_cell_type"]]
        batch_scores = [metrics.get(k, 0) for k in ["ASW_batch", "graph_connectivity"] if k in metrics]
        bio_avg = np.mean(bio_scores) if bio_scores else 0
        batch_avg = np.mean(batch_scores) if batch_scores else 0
        # 60% bio, 40% batch (standard scIB weighting)
        return 0.6 * bio_avg + 0.4 * batch_avg if batch_scores else bio_avg
```

#### `LinearProbeBenchmark`

```python
class LinearProbeBenchmark(BaseBenchmark):
    """Linear probe for cell type classification.

    Trains a simple logistic regression on frozen embeddings
    to measure how linearly separable cell types are in the
    embedding space. This is a critical baseline — if a linear
    model achieves high accuracy, the embeddings are useful.
    """

    name = "linear_probe"
    required_metadata = ["cell_type"]

    def __init__(
        self,
        cell_type_key: str = "cell_type",
        test_size: float = 0.2,
        max_iter: int = 1000,
    ):
        self.cell_type_key = cell_type_key
        self.test_size = test_size
        self.max_iter = max_iter

    def run_evaluation(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        **kwargs,
    ) -> BenchmarkResult:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        labels = adata.obs[self.cell_type_key].values
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=self.test_size, stratify=labels, random_state=42
        )

        clf = LogisticRegression(max_iter=self.max_iter, multi_class="multinomial")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=kwargs.get("dataset_name", "unknown"),
            metrics={
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            },
            metadata={
                "n_cells": len(adata),
                "n_cell_types": len(set(labels)),
                "n_train": len(X_train),
                "n_test": len(X_test),
            },
        )
```

#### `EvalHarness`

```python
class EvalHarness:
    """Orchestrates running multiple benchmarks on a model."""

    def __init__(self, benchmarks: list[BaseBenchmark]):
        self.benchmarks = benchmarks

    @classmethod
    def from_config(cls, config: EvalConfig) -> "EvalHarness":
        """Build harness from YAML config."""
        benchmarks = []
        for bench_config in config.benchmarks:
            benchmark = get_benchmark(bench_config["name"], **bench_config.get("params", {}))
            benchmarks.append(benchmark)
        return cls(benchmarks)

    def run(
        self,
        model: nn.Module,
        datasets: dict[str, AnnData],
        tokenizer: BaseTokenizer,
        batch_size: int = 256,
        device: str = "cuda",
    ) -> list[BenchmarkResult]:
        """Run all benchmarks on all datasets."""
        results = []

        for dataset_name, adata in datasets.items():
            # 1. Extract embeddings
            embeddings = self._extract_embeddings(model, adata, tokenizer, batch_size, device)

            # 2. Run each benchmark
            for benchmark in self.benchmarks:
                # Check required metadata
                missing = [k for k in benchmark.required_metadata if k not in adata.obs]
                if missing:
                    logger.warning(f"Skipping {benchmark.name} on {dataset_name}: missing {missing}")
                    continue

                result = benchmark.run_evaluation(embeddings, adata, dataset_name=dataset_name)
                results.append(result)
                logger.info(result.summary())

        return results

    def _extract_embeddings(
        self,
        model: nn.Module,
        adata: AnnData,
        tokenizer: BaseTokenizer,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        """Extract cell embeddings from model in batches."""
        model_in_eval_mode = model.training
        model.train(False)
        dataset = CellDataset(adata, gene_vocab=tokenizer.gene_vocab)
        dataloader = CellDataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                embeddings = model.encode(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    values=batch.get("values"),
                )
                all_embeddings.append(embeddings.cpu().numpy())

        model.train(model_in_eval_mode)
        return np.concatenate(all_embeddings, axis=0)
```

#### `EvalCallback` (Lightning Callback)

```python
class EvalCallback(pl.Callback):
    """Lightning callback that runs evaluation during training."""

    def __init__(
        self,
        config: EvalConfig,
        datasets: dict[str, AnnData] | None = None,
    ):
        self.config = config
        self.datasets = datasets
        self.harness = EvalHarness.from_config(config)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Run evaluation at configured intervals."""
        current_epoch = trainer.current_epoch
        if current_epoch % self.config.every_n_epochs != 0:
            return

        # Load datasets if not provided
        if self.datasets is None:
            self.datasets = self._load_datasets()

        # Run evaluation
        results = self.harness.run(
            model=pl_module.model,
            datasets=self.datasets,
            tokenizer=pl_module.tokenizer,
            device=pl_module.device,
        )

        # Log to trainer's logger
        for result in results:
            for metric_name, value in result.metrics.items():
                key = f"benchmark/{result.benchmark_name}/{result.dataset_name}/{metric_name}"
                pl_module.log(key, value, sync_dist=True)

    def _load_datasets(self) -> dict[str, AnnData]:
        """Download and cache datasets for benchmarking."""
        from scmodelforge.eval.datasets import get_eval_dataset
        loaded = {}
        for bench_config in self.config.benchmarks:
            dataset_name = bench_config.get("dataset", "tabula_sapiens")
            if dataset_name not in loaded:
                loaded[dataset_name] = get_eval_dataset(dataset_name)
        return loaded
```

### Evaluation Datasets

```python
# datasets/manager.py
import pooch

EVAL_DATASETS = {
    "tabula_sapiens": {
        "url": "...",  # URL to preprocessed .h5ad
        "hash": "...",
        "description": "Tabula Sapiens: 500k+ cells, 400+ cell types across organs",
    },
    "immune_atlas": {
        "url": "...",
        "hash": "...",
        "description": "Immune cell atlas for batch integration benchmarking",
    },
    "pbmc_10k": {
        "url": "...",
        "hash": "...",
        "description": "10k PBMC cells for quick validation",
    },
}

def get_eval_dataset(name: str, cache_dir: str | None = None) -> AnnData:
    """Download (if needed) and load a benchmarking dataset."""
    if name not in EVAL_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(EVAL_DATASETS.keys())}")

    info = EVAL_DATASETS[name]
    path = pooch.retrieve(
        url=info["url"],
        known_hash=info["hash"],
        path=cache_dir or pooch.os_cache("scmodelforge"),
    )
    return ad.read_h5ad(path)
```

### CLI Integration

```python
@main.command()
@click.option("--model", required=True, help="Path to model checkpoint or hub ID")
@click.option("--dataset", required=True, help="Dataset name or path to .h5ad")
@click.option("--benchmarks", multiple=True, default=["embedding_quality", "linear_probe"])
@click.option("--output", default=None, help="Path to save results JSON")
def benchmark(model: str, dataset: str, benchmarks: tuple[str], output: str | None):
    """Run benchmarks on a trained model."""
    ...
```

### Config Integration

```yaml
eval:
  every_n_epochs: 2
  benchmarks:
    - name: embedding_quality
      dataset: tabula_sapiens
      params:
        cell_type_key: cell_type
        batch_key: batch
    - name: linear_probe
      dataset: tabula_sapiens
      params:
        cell_type_key: cell_type
        test_size: 0.2
```

### Tests (Phase 1)

- `test_benchmark_base.py`: BaseBenchmark interface contract.
- `test_embedding_quality.py`: scIB metric computation on known embeddings with known results.
- `test_linear_probe.py`: Linear probe on separable synthetic embeddings (should get ~100%).
- `test_harness.py`: Multi-benchmark orchestration, missing metadata handling.
- `test_callback.py`: EvalCallback integration with Lightning trainer (mock benchmarking).
- `test_datasets.py`: Dataset download, caching, format validation.
- `test_embedding_extraction.py`: Verify embedding extraction produces correct shapes.

---

## Phase 2: Breadth (Months 4–6)

### Perturbation Prediction Benchmark

Assess the model's ability to predict post-perturbation expression profiles.

```python
class PerturbationBenchmark(BaseBenchmark):
    """Perturbation prediction quality assessment.

    Metrics (from PerturBench):
    - Pearson correlation on differentially expressed genes
    - MSE on top DEGs
    - Energy distance between predicted and true distributions
    - Comparison against linear baseline (mean shift)
    """

    name = "perturbation_prediction"
    required_metadata = ["perturbation", "control"]

    def __init__(
        self,
        n_top_genes: int = 50,
        include_linear_baseline: bool = True,
    ): ...
```

### PerturBench Integration

- Wrap PerturBench's assessment pipeline for standardised perturbation metrics.
- Include the critical linear baseline comparison (many models fail to beat this).

### Additional Phase 2 Files

```
src/scmodelforge/eval/
├── ...existing...
├── benchmarks/
│   ├── ...existing...
│   ├── perturbation.py      # PerturbationBenchmark
│   └── baselines.py         # Linear baseline, mean-shift baseline
```

---

## Phase 3: Community & Scale (Months 7–12)

### Gene Regulatory Network Inference

```python
class GRNBenchmark(BaseBenchmark):
    """Gene regulatory network inference assessment.

    Metrics:
    - AUROC against known regulatory networks (ENCODE, ChIP-Atlas)
    - AUPRC against known regulatory networks
    """

    name = "grn_inference"
    required_metadata = []  # Uses attention weights, not metadata
```

### cz-benchmarks Compatibility

- Direct integration with CZI's cz-benchmarks package.
- Results formatted for CZI Virtual Cells Platform leaderboards.
- `scmodelforge benchmark --format cz-benchmarks` output mode.

### Comprehensive Report Generation

- Generate a full report (HTML or PDF) with:
  - Metric tables across all benchmarks
  - UMAP visualisations of embeddings
  - Comparison against published baselines
  - Training curves
  - Model card metadata

### Community Benchmark Plugin System

- Entry point for third-party benchmarks: `scmodelforge.benchmarks`
- Community can create `pip install scmodelforge-my-benchmark` packages.

---

## Checklist

### Phase 1
- [ ] Define `BenchmarkResult` dataclass
- [ ] Implement `BaseBenchmark` abstract class
- [ ] Implement `EmbeddingQualityBenchmark` (scIB metrics)
- [ ] Implement `LinearProbeBenchmark` (logistic regression on embeddings)
- [ ] Implement `EvalHarness` orchestrator
- [ ] Implement `EvalCallback` (Lightning callback)
- [ ] Implement dataset manager (download, cache, load)
- [ ] Implement embedding extraction utility
- [ ] Implement benchmark registry
- [ ] Wire up CLI `benchmark` command
- [ ] Add config parsing for benchmarking section
- [ ] Write comprehensive tests
- [ ] Validate metrics against published scIB results
- [ ] Write docstrings and API documentation

### Phase 2
- [ ] Implement `PerturbationBenchmark` with PerturBench metrics
- [ ] Implement linear baseline for perturbation prediction
- [ ] Add perturbation datasets (Norman, Adamson)

### Phase 3
- [ ] Implement `GRNBenchmark` (AUROC/AUPRC on regulatory networks)
- [ ] cz-benchmarks compatibility layer
- [ ] HTML/PDF report generator
- [ ] Benchmark plugin system via entry points
