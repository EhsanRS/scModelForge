# Evaluation Module (`scmodelforge.eval`)

Comprehensive evaluation framework for single-cell foundation models.

## Overview

The `scmodelforge.eval` module provides a structured framework for assessing single-cell foundation model quality across three key dimensions: cell-type classification via linear probe, embedding quality via scIB metrics, and perturbation response prediction. The module is built around a benchmark abstraction that separates embedding extraction from evaluation, enabling efficient multi-benchmark assessment.

The evaluation architecture follows a three-layer design. At the base, `BaseBenchmark` defines the interface for all evaluation tasks: benchmarks receive precomputed embeddings and AnnData annotations, validate required metadata, and return structured results. Concrete benchmarks implement domain-specific evaluation logic: `LinearProbeBenchmark` measures biological signal via supervised cell-type classification, `EmbeddingQualityBenchmark` quantifies clustering quality and batch correction using scIB metrics, and `PerturbationBenchmark` assesses the model's ability to predict gene expression changes in response to perturbations.

The `EvalHarness` orchestrates evaluation across multiple datasets and benchmarks, extracting embeddings once per dataset and funneling them through all benchmarks. During pretraining, `AssessmentCallback` integrates with PyTorch Lightning to trigger periodic evaluation, logging metrics to the training dashboard. This design enables reproducible, efficient evaluation at scale while maintaining flexibility for custom benchmarks via the registry system.

All benchmarks produce `BenchmarkResult` objects with standardized metrics dictionaries, enabling downstream analysis, comparison across models, and integration with experiment tracking systems. The module handles both pretraining assessment and standalone benchmark evaluation via the CLI.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| `BenchmarkResult` | Dataclass storing benchmark name, dataset, metrics, and metadata |
| `BaseBenchmark` | Abstract base class for all benchmark implementations |
| `LinearProbeBenchmark` | Logistic regression on cell embeddings for cell-type classification |
| `EmbeddingQualityBenchmark` | scIB metrics: NMI, ARI, ASW, batch correction |
| `PerturbationBenchmark` | Ridge regression for perturbation response prediction |
| `EvalHarness` | Orchestrates multiple benchmarks across multiple datasets |
| `AssessmentCallback` | PyTorch Lightning callback for in-training evaluation |
| `register_benchmark()` | Decorator to register a benchmark class in the global registry |
| `get_benchmark()` | Instantiate a registered benchmark by name |
| `list_benchmarks()` | List all registered benchmark names |
| `extract_embeddings()` | Extract cell embeddings from a model for evaluation |

## Detailed API

### `BenchmarkResult`

```python
from scmodelforge.eval import BenchmarkResult
```

Dataclass storing the result of a single benchmark run on a single dataset.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark_name` | `str` | *required* | Name of the benchmark that produced this result |
| `dataset_name` | `str` | *required* | Name of the dataset that was evaluated |
| `metrics` | `dict[str, float]` | *required* | Dictionary mapping metric names to float values |
| `metadata` | `dict[str, Any]` | `{}` | Additional metadata (e.g., number of cells, parameters) |

#### Methods

**`to_dict() -> dict[str, Any]`**

Serialize the result to a plain dictionary for JSON export or logging.

**Returns:** Dictionary with `benchmark_name`, `dataset_name`, `metrics`, and `metadata` keys.

**`summary() -> str`**

Generate a human-readable one-line summary of the result.

**Returns:** String in the format `[benchmark_name] dataset_name: metric1=0.9234, metric2=0.8567`

#### Example

```python
from scmodelforge.eval import BenchmarkResult

result = BenchmarkResult(
    benchmark_name="linear_probe",
    dataset_name="pbmc_3k",
    metrics={"accuracy": 0.92, "f1_macro": 0.89},
    metadata={"n_cells": 2638, "n_classes": 8}
)

print(result.summary())
# [linear_probe] pbmc_3k: accuracy=0.9200, f1_macro=0.8900

result_dict = result.to_dict()
# {'benchmark_name': 'linear_probe', 'dataset_name': 'pbmc_3k', ...}
```

---

### `BaseBenchmark`

```python
from scmodelforge.eval import BaseBenchmark
```

Abstract base class for all evaluation benchmarks. Subclasses must implement `run()`, `name`, and `required_obs_keys`.

#### Properties

**`name: str`** (abstract)

Short identifier for this benchmark (e.g., `"linear_probe"`).

**`required_obs_keys: list[str]`** (abstract)

List of AnnData `.obs` column names required by this benchmark.

#### Methods

**`run(embeddings: np.ndarray, adata: AnnData, dataset_name: str) -> BenchmarkResult`** (abstract)

Run the benchmark on precomputed embeddings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | `np.ndarray` | Cell embeddings of shape `(n_cells, hidden_dim)` |
| `adata` | `AnnData` | AnnData object with matching `.obs` annotations |
| `dataset_name` | `str` | Name used to identify this dataset in results |

**Returns:** `BenchmarkResult` containing metrics and metadata.

**`validate_adata(adata: AnnData) -> None`**

Check that `adata` has all required `.obs` columns.

**Raises:** `ValueError` if any required key is missing from `adata.obs`.

#### Example

```python
from scmodelforge.eval import BaseBenchmark, BenchmarkResult
from scmodelforge.eval import register_benchmark

@register_benchmark("custom_metric")
class CustomBenchmark(BaseBenchmark):
    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "custom_metric"

    @property
    def required_obs_keys(self) -> list[str]:
        return ["cell_type"]

    def run(self, embeddings, adata, dataset_name):
        self.validate_adata(adata)
        # Custom evaluation logic here
        score = compute_custom_score(embeddings, adata)
        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=dataset_name,
            metrics={"custom_score": score},
            metadata={"threshold": self._threshold}
        )
```

---

### `LinearProbeBenchmark`

```python
from scmodelforge.eval import LinearProbeBenchmark
```

Evaluate embeddings via logistic regression on cell-type labels. Splits data into train/test, fits a `sklearn.linear_model.LogisticRegression`, and reports accuracy, macro-F1, and weighted-F1.

#### Constructor

```python
LinearProbeBenchmark(
    cell_type_key: str = "cell_type",
    test_size: float = 0.2,
    max_iter: int = 1000,
    seed: int = 42
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cell_type_key` | `str` | `"cell_type"` | Column in `adata.obs` with cell-type labels |
| `test_size` | `float` | `0.2` | Fraction of data to hold out for testing |
| `max_iter` | `int` | `1000` | Maximum iterations for the logistic regression solver |
| `seed` | `int` | `42` | Random seed for train/test split and solver |

#### Properties

**`name: str`** → Returns `"linear_probe"`

**`required_obs_keys: list[str]`** → Returns `[cell_type_key]`

#### Methods

**`run(embeddings: np.ndarray, adata: AnnData, dataset_name: str) -> BenchmarkResult`**

Run the linear probe benchmark. Performs stratified train/test split, fits logistic regression, and computes metrics.

**Metrics returned:**
- `accuracy`: Overall classification accuracy
- `f1_macro`: Macro-averaged F1 score (unweighted)
- `f1_weighted`: Weighted F1 score (weighted by class support)

**Metadata returned:**
- `n_cells`: Total number of cells
- `n_classes`: Number of unique cell types
- `test_size`: Fraction used for testing
- `cell_type_key`: Column name used for labels

**Raises:** `ImportError` if scikit-learn is not installed.

#### Example

```python
from scmodelforge.eval import LinearProbeBenchmark, extract_embeddings
import scanpy as sc

# Load data and model
adata = sc.read_h5ad("pbmc_3k.h5ad")
model = load_pretrained_model("checkpoint.ckpt")
tokenizer = get_tokenizer("rank_value", gene_vocab)

# Extract embeddings
embeddings = extract_embeddings(
    model, adata, tokenizer,
    batch_size=256, device="cuda"
)

# Run benchmark
benchmark = LinearProbeBenchmark(
    cell_type_key="cell_type",
    test_size=0.2,
    seed=42
)
result = benchmark.run(embeddings, adata, "pbmc_3k")
print(result.summary())
# [linear_probe] pbmc_3k: accuracy=0.9234, f1_macro=0.8967, f1_weighted=0.9201
```

---

### `EmbeddingQualityBenchmark`

```python
from scmodelforge.eval import EmbeddingQualityBenchmark
```

Evaluate embedding quality using scIB metrics. Computes NMI, ARI, ASW for cell-type (biological conservation) and optionally ASW and graph connectivity for batch correction. Overall score is `0.6 * bio_mean + 0.4 * batch_mean`.

#### Constructor

```python
EmbeddingQualityBenchmark(
    cell_type_key: str = "cell_type",
    batch_key: str | None = "batch",
    n_neighbors: int = 15
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cell_type_key` | `str` | `"cell_type"` | Column in `adata.obs` for cell-type labels |
| `batch_key` | `str \| None` | `"batch"` | Column in `adata.obs` for batch labels (None to skip batch metrics) |
| `n_neighbors` | `int` | `15` | Number of neighbors for scanpy neighborhood graph |

#### Properties

**`name: str`** → Returns `"embedding_quality"`

**`required_obs_keys: list[str]`** → Returns `[cell_type_key]` or `[cell_type_key, batch_key]` if batch_key is set

#### Methods

**`run(embeddings: np.ndarray, adata: AnnData, dataset_name: str) -> BenchmarkResult`**

Run the embedding quality benchmark. Builds neighborhood graph, performs optimal-resolution clustering, and computes scIB metrics.

**Metrics returned:**
- `nmi`: Normalized mutual information (clustering vs cell-type)
- `ari`: Adjusted Rand index (clustering vs cell-type)
- `asw_cell_type`: Average silhouette width by cell-type
- `asw_batch`: Average silhouette width by batch (batch correction, if batch_key set)
- `graph_connectivity`: Graph connectivity score (if batch_key set)
- `overall`: Weighted overall score (0.6 * bio + 0.4 * batch)

**Metadata returned:**
- `n_cells`: Total number of cells
- `cell_type_key`: Column name used for cell types
- `batch_key`: Column name used for batches
- `n_neighbors`: Neighbor count used for graph construction

**Raises:** `ImportError` if scanpy or scib are not installed.

#### Example

```python
from scmodelforge.eval import EmbeddingQualityBenchmark, extract_embeddings

# Extract embeddings (same as LinearProbeBenchmark example)
embeddings = extract_embeddings(model, adata, tokenizer, device="cuda")

# Run benchmark with batch correction metrics
benchmark = EmbeddingQualityBenchmark(
    cell_type_key="cell_type",
    batch_key="donor",
    n_neighbors=15
)
result = benchmark.run(embeddings, adata, "multi_donor_pbmc")

print(result.metrics)
# {
#   'nmi': 0.8234, 'ari': 0.7891, 'asw_cell_type': 0.6543,
#   'asw_batch': 0.4321, 'graph_connectivity': 0.8765,
#   'overall': 0.7012
# }

# Run without batch metrics
benchmark_no_batch = EmbeddingQualityBenchmark(batch_key=None)
result_no_batch = benchmark_no_batch.run(embeddings, adata, "pbmc_3k")
# metrics will only include nmi, ari, asw_cell_type, overall
```

---

### `PerturbationBenchmark`

```python
from scmodelforge.eval import PerturbationBenchmark
```

Evaluate embeddings via perturbation response prediction. Trains a Ridge regression on `{mean_perturbed_embedding → expression_delta}` for training perturbations, then evaluates Pearson correlation, MSE, and direction accuracy on held-out perturbations. Includes a mean-shift baseline for honest comparison.

#### Constructor

```python
PerturbationBenchmark(
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    n_top_genes: int = 50,
    test_fraction: float = 0.2,
    seed: int = 42
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `perturbation_key` | `str` | `"perturbation"` | Column in `adata.obs` with perturbation labels |
| `control_label` | `str` | `"control"` | Value identifying control (unperturbed) cells |
| `n_top_genes` | `int` | `50` | Number of top DEGs to evaluate metrics on |
| `test_fraction` | `float` | `0.2` | Fraction of perturbations held out for testing |
| `seed` | `int` | `42` | Random seed for train/test split and Ridge regression |

#### Properties

**`name: str`** → Returns `"perturbation"`

**`required_obs_keys: list[str]`** → Returns `[perturbation_key]`

#### Methods

**`run(embeddings: np.ndarray, adata: AnnData, dataset_name: str) -> BenchmarkResult`**

Run the perturbation prediction benchmark. Computes expression deltas, splits perturbations into train/test, trains Ridge regression, and evaluates against mean-shift baseline.

**Metrics returned:**
- `pearson_mean`: Mean Pearson correlation across test perturbations (model)
- `mse_mean`: Mean squared error on top DEGs (model)
- `direction_accuracy`: Fraction of DEGs with correct sign (model)
- `baseline_pearson_mean`: Mean Pearson for mean-shift baseline
- `baseline_mse_mean`: MSE for mean-shift baseline
- `baseline_direction_accuracy`: Direction accuracy for baseline
- `fraction_above_baseline`: Fraction of test perturbations where model beats baseline

**Metadata returned:**
- `n_cells`: Total number of cells
- `n_perturbations`: Total number of non-control perturbations
- `n_train`: Number of perturbations used for training
- `n_test`: Number of perturbations used for testing
- `n_top_genes`: Number of top DEGs evaluated
- `perturbation_key`: Column name for perturbations
- `control_label`: Label identifying control cells

**Raises:** `ValueError` if control label not found or fewer than 2 perturbations available.

#### Example

```python
from scmodelforge.eval import PerturbationBenchmark, extract_embeddings

# Assume adata has perturbation data
# adata.obs["perturbation"] = ["control", "gene_A_ko", "gene_B_ko", ...]

embeddings = extract_embeddings(model, adata, tokenizer, device="cuda")

benchmark = PerturbationBenchmark(
    perturbation_key="perturbation",
    control_label="control",
    n_top_genes=50,
    test_fraction=0.2,
    seed=42
)
result = benchmark.run(embeddings, adata, "perturb_seq_pilot")

print(result.metrics)
# {
#   'pearson_mean': 0.6543,
#   'mse_mean': 0.0234,
#   'direction_accuracy': 0.7891,
#   'baseline_pearson_mean': 0.4321,
#   'baseline_mse_mean': 0.0456,
#   'baseline_direction_accuracy': 0.6234,
#   'fraction_above_baseline': 0.75
# }
```

---

### `EvalHarness`

```python
from scmodelforge.eval import EvalHarness
```

Orchestrates evaluation benchmarks across multiple datasets. Extracts embeddings once per dataset, then runs all benchmarks on those embeddings.

#### Constructor

```python
EvalHarness(benchmarks: list[BaseBenchmark])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `benchmarks` | `list[BaseBenchmark]` | List of benchmark instances to run |

#### Class Methods

**`from_config(config: EvalConfig) -> EvalHarness`**

Create an `EvalHarness` from an `EvalConfig`. Instantiates benchmarks from the registry using the names in `config.benchmarks`. Falls back to `["linear_probe"]` if the list is empty.

Benchmark specifications can be:
- **String:** `"linear_probe"` (uses default parameters)
- **Dict:** `{"name": "linear_probe", "test_size": 0.3}` (with custom parameters)

**Returns:** `EvalHarness` instance.

#### Methods

**`run(model: nn.Module, datasets: dict[str, AnnData], tokenizer: BaseTokenizer, batch_size: int = 256, device: str = "cpu") -> list[BenchmarkResult]`**

Run all benchmarks on all datasets. Extracts embeddings once per dataset, then runs every benchmark.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | Model with an `encode()` method |
| `datasets` | `dict[str, AnnData]` | *required* | Mapping of dataset name to AnnData |
| `tokenizer` | `BaseTokenizer` | *required* | Tokenizer for embedding extraction |
| `batch_size` | `int` | `256` | Batch size for embedding extraction |
| `device` | `str` | `"cpu"` | Device for inference (`"cpu"` or `"cuda"`) |

**Returns:** List of `BenchmarkResult` objects (one per benchmark per dataset).

**`run_on_embeddings(embeddings: np.ndarray, adata: AnnData, dataset_name: str) -> list[BenchmarkResult]`**

Run all benchmarks on precomputed embeddings (useful when embeddings are already cached).

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | `np.ndarray` | Cell embeddings of shape `(n_cells, hidden_dim)` |
| `adata` | `AnnData` | AnnData with matching annotations |
| `dataset_name` | `str` | Name for this dataset in results |

**Returns:** List of `BenchmarkResult` objects (one per benchmark).

#### Example

```python
from scmodelforge.eval import EvalHarness, LinearProbeBenchmark, EmbeddingQualityBenchmark
import scanpy as sc

# Manual construction
benchmarks = [
    LinearProbeBenchmark(test_size=0.2),
    EmbeddingQualityBenchmark(batch_key="donor")
]
harness = EvalHarness(benchmarks)

# Or from config
from scmodelforge.config import EvalConfig
config = EvalConfig(
    benchmarks=[
        "linear_probe",
        {"name": "embedding_quality", "batch_key": "batch"}
    ]
)
harness = EvalHarness.from_config(config)

# Run on multiple datasets
datasets = {
    "pbmc_3k": sc.read_h5ad("pbmc_3k.h5ad"),
    "heart_1k": sc.read_h5ad("heart_1k.h5ad")
}

results = harness.run(
    model=model,
    datasets=datasets,
    tokenizer=tokenizer,
    batch_size=256,
    device="cuda"
)

# Results has 4 entries: 2 benchmarks × 2 datasets
for result in results:
    print(result.summary())
# [linear_probe] pbmc_3k: accuracy=0.9234, ...
# [embedding_quality] pbmc_3k: nmi=0.8234, ...
# [linear_probe] heart_1k: accuracy=0.8765, ...
# [embedding_quality] heart_1k: nmi=0.7543, ...
```

---

### `AssessmentCallback`

```python
from scmodelforge.eval import AssessmentCallback
```

PyTorch Lightning callback for running benchmarks during training. Triggered at the end of validation epochs at a configurable frequency. Logs results to the Lightning logger under `assessment/{benchmark}/{dataset}/{metric}`.

#### Constructor

```python
AssessmentCallback(
    config: EvalConfig,
    datasets: dict[str, AnnData],
    tokenizer: BaseTokenizer,
    batch_size: int | None = None,
    device: str | None = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `EvalConfig` | *required* | Configuration controlling frequency and benchmark list |
| `datasets` | `dict[str, AnnData]` | *required* | Mapping of dataset name to AnnData for evaluation |
| `tokenizer` | `BaseTokenizer` | *required* | Tokenizer for embedding extraction |
| `batch_size` | `int \| None` | `None` | Batch size for embedding extraction (overrides config if set) |
| `device` | `str \| None` | `None` | Device override (if None, uses trainer's device) |

#### Methods

**`on_validation_epoch_end(trainer: pl.Trainer, pl_module: pl.LightningModule) -> None`**

Callback hook triggered at the end of each validation epoch. Runs benchmarks if the current epoch is a multiple of `config.every_n_epochs`.

#### Attributes

**`_last_results: list[BenchmarkResult]`**

Stores the results from the most recent evaluation run (useful for post-training inspection).

#### Example

```python
from scmodelforge.eval import AssessmentCallback
from scmodelforge.config import EvalConfig
import lightning.pytorch as pl
import scanpy as sc

# Prepare evaluation datasets
eval_datasets = {
    "pbmc_val": sc.read_h5ad("pbmc_val.h5ad"),
    "heart_val": sc.read_h5ad("heart_val.h5ad")
}

# Configure evaluation
eval_config = EvalConfig(
    every_n_epochs=2,  # Run every 2 epochs
    batch_size=256,
    benchmarks=["linear_probe", "embedding_quality"]
)

# Create callback
callback = AssessmentCallback(
    config=eval_config,
    datasets=eval_datasets,
    tokenizer=tokenizer,
    device="cuda"
)

# Add to trainer
trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[callback]
)

# During training, metrics will be logged as:
# assessment/linear_probe/pbmc_val/accuracy
# assessment/linear_probe/pbmc_val/f1_macro
# assessment/embedding_quality/pbmc_val/nmi
# ...

trainer.fit(model, datamodule)

# Access last results
for result in callback._last_results:
    print(result.summary())
```

---

### Registry Functions

```python
from scmodelforge.eval import register_benchmark, get_benchmark, list_benchmarks
```

#### `register_benchmark(name: str)`

Class decorator that registers a benchmark class under the given name.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Registry key (e.g., `"linear_probe"`) |

**Raises:** `ValueError` if the name is already registered.

**Example:**

```python
@register_benchmark("my_benchmark")
class MyBenchmark(BaseBenchmark):
    ...
```

#### `get_benchmark(name: str, **kwargs: Any) -> BaseBenchmark`

Instantiate a registered benchmark by name.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Registry key (e.g., `"linear_probe"`) |
| `**kwargs` | `Any` | Forwarded to the benchmark constructor |

**Returns:** Benchmark instance.

**Raises:** `ValueError` if the name is not found in the registry.

**Example:**

```python
# Default parameters
benchmark = get_benchmark("linear_probe")

# Custom parameters
benchmark = get_benchmark("linear_probe", test_size=0.3, seed=123)
```

#### `list_benchmarks() -> list[str]`

Return sorted list of all registered benchmark names.

**Example:**

```python
available = list_benchmarks()
# ['embedding_quality', 'linear_probe', 'perturbation']
```

---

### `extract_embeddings()`

```python
from scmodelforge.eval import extract_embeddings
```

Extract cell embeddings from a model for evaluation. Uses the existing data pipeline (`CellDataset` + `TokenizedCellDataset`) without masking, then calls `model.encode()` to get cell-level embeddings.

```python
extract_embeddings(
    model: torch.nn.Module,
    adata: AnnData,
    tokenizer: BaseTokenizer,
    batch_size: int = 256,
    device: str = "cpu"
) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | *required* | Model with an `encode(input_ids, attention_mask, values)` method |
| `adata` | `AnnData` | *required* | AnnData object with expression data |
| `tokenizer` | `BaseTokenizer` | *required* | Tokenizer instance for converting cells to model inputs |
| `batch_size` | `int` | `256` | Batch size for inference |
| `device` | `str` | `"cpu"` | Device to run inference on (`"cpu"` or `"cuda"`) |

**Returns:** Cell embeddings as a numpy array of shape `(n_cells, hidden_dim)`.

#### Example

```python
from scmodelforge.eval import extract_embeddings
from scmodelforge.models import get_model
from scmodelforge.tokenizers import get_tokenizer
import scanpy as sc

# Load model and data
model = get_model("transformer_encoder", hidden_dim=512, num_layers=6)
model.load_state_dict(torch.load("checkpoint.pt"))
adata = sc.read_h5ad("pbmc_3k.h5ad")
tokenizer = get_tokenizer("rank_value", gene_vocab)

# Extract embeddings
embeddings = extract_embeddings(
    model=model,
    adata=adata,
    tokenizer=tokenizer,
    batch_size=512,
    device="cuda"
)

print(embeddings.shape)  # (2638, 512)

# Use embeddings for custom analysis
from sklearn.manifold import TSNE
tsne_coords = TSNE(n_components=2).fit_transform(embeddings)
```

## See Also

- **Configuration**: `scmodelforge.config.EvalConfig` for configuring evaluation frequency and benchmark selection
- **Training Integration**: `scmodelforge.training.TrainingPipeline` for using `AssessmentCallback` during pretraining
- **Models**: `scmodelforge.models` for model architectures with `encode()` methods
- **Tokenizers**: `scmodelforge.tokenizers` for preparing data for embedding extraction
- **CLI**: `scmodelforge benchmark --config config.yaml` for standalone benchmark evaluation
