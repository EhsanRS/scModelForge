# `scmodelforge.training`

End-to-end training infrastructure for single-cell foundation model pretraining.

## Overview

The `scmodelforge.training` module provides a complete, config-driven training pipeline built on PyTorch Lightning. It orchestrates every step of the pretraining workflow: loading and preprocessing single-cell data, constructing tokenizers with masking strategies, building model architectures, and executing multi-epoch training with distributed GPU support.

The central orchestrator is `TrainingPipeline`, which reads a YAML configuration file and executes a reproducible training run. Raw AnnData files are loaded, preprocessed, and wrapped in a `CellDataModule` that handles train/validation splits. Tokenization and masking happen on-the-fly in the dataset `__getitem__` method via `TokenizedCellDataset`, ensuring efficient memory usage for large corpora. The `ScModelForgeLightningModule` wraps the model and implements the training loop, logging loss and perplexity metrics.

This module integrates seamlessly with Lightning's ecosystem: multi-GPU training with DDP/FSDP, automatic checkpointing, learning rate scheduling, gradient clipping, and logging to Weights & Biases, TensorBoard, or CSV. Custom callbacks track throughput metrics (cells/second) and gradient norms for monitoring training health. The design supports resuming from checkpoints, mixed-precision training, and flexible optimizer/scheduler configuration.

All configuration is expressed declaratively in YAML and loaded via `ScModelForgeConfig`. This makes experiments reproducible and easy to version-control. The training module is the reference implementation of the full data → tokenization → model → training pipeline.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| `TrainingPipeline` | Config-driven orchestrator for end-to-end pretraining |
| `ScModelForgeLightningModule` | Lightning module wrapping model with training/validation steps |
| `CellDataModule` | Lightning-style data module with setup, train/val dataloaders |
| `TokenizedCellDataset` | Wraps CellDataset with tokenization and masking in `__getitem__` |
| `build_optimizer` | Factory for AdamW/Adam with parameter groups and weight decay |
| `build_scheduler` | Factory for cosine warmup, cosine, and linear LR schedulers |
| `TrainingMetricsLogger` | Callback logging cells/sec, step time, epoch time |
| `GradientNormLogger` | Callback logging gradient L2 norms before optimizer step |
| `SamplerEpochCallback` | Callback advancing epoch-aware samplers (curriculum learning, shard rotation) |
| `get_environment_info` | Collect runtime environment details (Python, CUDA, platform) |
| `log_training_config` | Log key training config values at INFO level |

## API Reference

### `TrainingPipeline`

Config-driven training pipeline that orchestrates data loading, model creation, and Lightning training.

```python
from scmodelforge.config import load_config
from scmodelforge.training import TrainingPipeline

config = load_config("configs/geneformer_basic.yaml")
pipeline = TrainingPipeline(config)
trainer = pipeline.run()
```

#### Constructor

```python
TrainingPipeline(config: ScModelForgeConfig)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `ScModelForgeConfig` | *required* | Full scModelForge configuration with data, model, training sections |

#### Methods

##### `run() -> pl.Trainer`

Execute the full training pipeline from seed to checkpoint.

**Workflow:**
1. Seed RNG with `pl.seed_everything` for reproducibility
2. Log configuration and environment info
3. Build `CellDataModule` and call `setup()`
4. Infer `vocab_size` from gene vocabulary
5. Instantiate model from registry using `get_model`
6. Wrap model in `ScModelForgeLightningModule`
7. Build callbacks (checkpoint, LR monitor, custom metrics, optional `AssessmentCallback` when `eval.benchmarks` is configured)
8. Build logger (wandb/tensorboard/csv)
9. Resolve devices and distributed strategy
10. Create `pl.Trainer` with all config options
11. Call `trainer.fit()` with train/val dataloaders

**Returns:**
- `pl.Trainer` — The Lightning Trainer after fitting

**Example:**

```python
from scmodelforge.config import load_config
from scmodelforge.training import TrainingPipeline

# Load configuration from YAML
config = load_config("configs/geneformer_basic.yaml")

# Override config for quick test run
config.training.max_epochs = 1
config.training.batch_size = 32

# Run full pipeline
pipeline = TrainingPipeline(config)
trainer = pipeline.run()

# Access best checkpoint path
best_ckpt = trainer.checkpoint_callback.best_model_path
print(f"Best checkpoint: {best_ckpt}")
```

##### `_build_callbacks(data_module=None) -> list[pl.Callback]`

Build the list of Lightning callbacks (internal).

**Parameters:**
- `data_module` (`CellDataModule | None`) — If provided and `config.eval.benchmarks` is non-empty, an `AssessmentCallback` is appended for in-training evaluation using the data module's loaded AnnData and tokenizer.

**Returns:**
- `list[pl.Callback]` — ModelCheckpoint, LearningRateMonitor, TrainingMetricsLogger, GradientNormLogger, and optionally `AssessmentCallback`

##### `_build_logger() -> Logger`

Build the training logger based on `config.training.logger` (internal).

**Returns:**
- `Logger` — WandbLogger, TensorBoardLogger, or CSVLogger

##### `_resolve_devices_and_strategy() -> tuple[int, str]`

Determine devices and strategy for the Trainer (internal).

**Returns:**
- `tuple[int, str]` — (devices, strategy) suitable for `pl.Trainer`

---

### `ScModelForgeLightningModule`

PyTorch Lightning module wrapping a scModelForge model for pretraining. Implements training and validation steps with loss and perplexity logging.

```python
from scmodelforge.models import get_model
from scmodelforge.training import ScModelForgeLightningModule

model = get_model("transformer_encoder", config.model)
lightning_module = ScModelForgeLightningModule(
    model=model,
    optimizer_config=config.training.optimizer,
    scheduler_config=config.training.scheduler,
)
```

#### Constructor

```python
ScModelForgeLightningModule(
    model: nn.Module,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | Neural network model (e.g. TransformerEncoder) |
| `optimizer_config` | `OptimizerConfig` | *required* | Optimizer configuration (name, lr, weight_decay) |
| `scheduler_config` | `SchedulerConfig \| None` | `None` | Optional scheduler configuration |

#### Methods

##### `forward(batch: dict[str, torch.Tensor]) -> ModelOutput`

Forward pass through the model.

**Parameters:**
- `batch` (dict) — Dict with keys `input_ids`, `attention_mask`, optionally `values`, `labels`, plus any extra keys (e.g. `bin_ids`, `masked_positions`) forwarded to the model

**Returns:**
- `ModelOutput` — Model output with `loss`, `logits`, `embeddings`

##### `training_step(batch: dict, batch_idx: int) -> torch.Tensor`

Run a single training step.

**Logs:**
- `train/loss` (progress bar, sync_dist)
- `train/perplexity` (sync_dist)

**Returns:**
- `torch.Tensor` — Loss value for backpropagation

##### `validation_step(batch: dict, batch_idx: int) -> None`

Run a single validation step.

**Logs:**
- `val/loss` (progress bar, sync_dist)
- `val/perplexity` (sync_dist)

##### `configure_optimizers() -> dict[str, Any]`

Build optimizer and optional scheduler.

**Returns:**
- `dict` — Dictionary with keys `optimizer` and optionally `lr_scheduler`

**Example:**

```python
from scmodelforge.config.schema import OptimizerConfig, SchedulerConfig
from scmodelforge.models import get_model
from scmodelforge.training import ScModelForgeLightningModule
import lightning.pytorch as pl

# Build model
model = get_model("transformer_encoder", config.model)

# Configure optimizer and scheduler
opt_cfg = OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.01)
sched_cfg = SchedulerConfig(
    name="cosine_warmup",
    warmup_steps=2000,
    total_steps=100000,
)

# Create Lightning module
lightning_module = ScModelForgeLightningModule(
    model=model,
    optimizer_config=opt_cfg,
    scheduler_config=sched_cfg,
)

# Train with Lightning Trainer
trainer = pl.Trainer(max_epochs=10)
trainer.fit(lightning_module, train_loader, val_loader)
```

---

### `CellDataModule`

Lightning-style DataModule for single-cell pretraining. Loads data, builds gene vocabulary, tokenizer, and masking strategy, splits into train/val, and provides dataloaders.

```python
from scmodelforge.training import CellDataModule

data_module = CellDataModule(
    data_config=config.data,
    tokenizer_config=config.tokenizer,
    training_batch_size=64,
    num_workers=4,
    val_split=0.05,
    seed=42,
)
data_module.setup()
train_loader = data_module.train_dataloader()
```

#### Constructor

```python
CellDataModule(
    data_config: DataConfig,
    tokenizer_config: TokenizerConfig,
    training_batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.05,
    seed: int = 42,
    adata: object | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_config` | `DataConfig` | *required* | Data loading and preprocessing configuration |
| `tokenizer_config` | `TokenizerConfig` | *required* | Tokenizer strategy and masking configuration |
| `training_batch_size` | `int` | `64` | Batch size for training and validation |
| `num_workers` | `int` | `4` | DataLoader worker count |
| `val_split` | `float` | `0.05` | Fraction of data reserved for validation |
| `seed` | `int` | `42` | Random seed for reproducible splits |
| `adata` | `object \| None` | `None` | Optional pre-loaded AnnData for testing (skips file loading) |
| `sampling_config` | `SamplingConfig \| None` | `None` | Weighted sampling configuration |
| `gene_selection_config` | `GeneSelectionConfig \| None` | `None` | Batch-level gene selection configuration |

#### Properties

##### `gene_vocab: GeneVocab`

Gene vocabulary (available after `setup()`).

##### `tokenizer: BaseTokenizer`

Tokenizer instance (available after `setup()`).

##### `adata: AnnData`

Loaded AnnData (available after `setup()`). Used by `TrainingPipeline` to supply data for `AssessmentCallback`.

##### `masking: MaskingStrategy | None`

Masking strategy (available after `setup()`).

#### Methods

##### `setup(stage: str | None = None) -> None`

Load data, build tokenizer, split into train/val.

**Workflow:**
1. Load AnnData from file or Census
2. Build `GeneVocab` from AnnData
3. Build `PreprocessingPipeline`
4. Build `CellDataset`
5. Get tokenizer from registry
6. Build `MaskingStrategy` from config
7. Random split into train/val using `torch.random_split`
8. Wrap subsets with `TokenizedCellDataset` (with masking)

**Parameters:**
- `stage` (str | None) — Lightning stage (unused, for compatibility)

**Notes:**
- Idempotent — calling multiple times is safe
- Both train and val datasets get masking (standard BERT-style pretraining)

##### `train_dataloader() -> DataLoader`

Training DataLoader with shuffle.

**Returns:**
- `DataLoader` — DataLoader with shuffle=True, collate_fn=tokenizer._collate

**Raises:**
- `RuntimeError` — If called before `setup()`

##### `val_dataloader() -> DataLoader`

Validation DataLoader without shuffle.

**Returns:**
- `DataLoader` — DataLoader with shuffle=False, collate_fn=tokenizer._collate

**Raises:**
- `RuntimeError` — If called before `setup()`

**Example:**

```python
from scmodelforge.config import load_config
from scmodelforge.training import CellDataModule

config = load_config("configs/geneformer_basic.yaml")
data_module = CellDataModule(
    data_config=config.data,
    tokenizer_config=config.tokenizer,
    training_batch_size=64,
    num_workers=4,
    val_split=0.05,
    seed=42,
)

# Setup loads data and builds vocab/tokenizer
data_module.setup()

# Access vocab size for model config
vocab_size = len(data_module.gene_vocab)
config.model.vocab_size = vocab_size

# Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Inspect a batch
batch = next(iter(train_loader))
print(batch.keys())  # dict_keys(['input_ids', 'attention_mask', 'values', 'labels', ...])
```

---

### `TokenizedCellDataset`

Wraps a `CellDataset` (or Subset) with tokenization and masking. Tokenization and masking happen on-the-fly in `__getitem__` for memory efficiency.

```python
from scmodelforge.training import TokenizedCellDataset
from scmodelforge.tokenizers import get_tokenizer
from scmodelforge.tokenizers.masking import MaskingStrategy

tokenizer = get_tokenizer("rank_value", gene_vocab, max_len=2048)
masking = MaskingStrategy(mask_ratio=0.15, vocab_size=len(gene_vocab))

tokenized_dataset = TokenizedCellDataset(
    dataset=cell_dataset,
    tokenizer=tokenizer,
    masking=masking,
)
```

#### Constructor

```python
TokenizedCellDataset(
    dataset: Dataset,
    tokenizer: BaseTokenizer,
    masking: MaskingStrategy | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `Dataset` | *required* | Underlying dataset returning dict with `expression`, `gene_indices`, `metadata` |
| `tokenizer` | `BaseTokenizer` | *required* | Tokenizer to convert raw cell data into model inputs |
| `masking` | `MaskingStrategy \| None` | `None` | Optional masking strategy applied after tokenization |

#### Methods

##### `__len__() -> int`

Return the number of cells.

##### `__getitem__(idx: int) -> TokenizedCell | MaskedTokenizedCell`

Tokenize and optionally mask a single cell.

**Parameters:**
- `idx` (int) — Cell index

**Returns:**
- `TokenizedCell` or `MaskedTokenizedCell` — Tokenized cell with input_ids, attention_mask, values, labels

**Example:**

```python
from scmodelforge.data import CellDataset, GeneVocab, PreprocessingPipeline
from scmodelforge.tokenizers import get_tokenizer
from scmodelforge.tokenizers.masking import MaskingStrategy
from scmodelforge.training import TokenizedCellDataset
import anndata as ad

# Load data
adata = ad.read_h5ad("data/pbmc.h5ad")
gene_vocab = GeneVocab.from_adata(adata)
preprocessing = PreprocessingPipeline(normalize="library_size", log1p=True)
cell_dataset = CellDataset(adata, gene_vocab, preprocessing)

# Build tokenizer and masking
tokenizer = get_tokenizer("rank_value", gene_vocab, max_len=2048)
masking = MaskingStrategy(mask_ratio=0.15, vocab_size=len(gene_vocab))

# Wrap with tokenization
tokenized = TokenizedCellDataset(cell_dataset, tokenizer, masking)

# Get a tokenized cell
cell = tokenized[0]
print(cell.input_ids.shape)  # (seq_len,)
print(cell.labels.shape)     # (seq_len,)
```

---

### `build_optimizer`

Create an optimizer with per-parameter weight decay groups. Bias and LayerNorm parameters get `weight_decay=0.0`.

```python
from scmodelforge.training import build_optimizer
from scmodelforge.config.schema import OptimizerConfig

config = OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.01)
optimizer = build_optimizer(model, config)
```

#### Signature

```python
build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> Optimizer
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | The model whose parameters to optimize |
| `config` | `OptimizerConfig` | *required* | Optimizer configuration (name, lr, weight_decay) |

**Returns:**
- `Optimizer` — torch.optim.AdamW or torch.optim.Adam

**Raises:**
- `ValueError` — If `config.name` is not a supported optimizer ("adamw" or "adam")

**Example:**

```python
from scmodelforge.config.schema import OptimizerConfig
from scmodelforge.training import build_optimizer
import torch.nn as nn

model = nn.Linear(512, 1000)
config = OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.01)
optimizer = build_optimizer(model, config)

# Inspect parameter groups
for i, group in enumerate(optimizer.param_groups):
    print(f"Group {i}: {len(group['params'])} params, wd={group['weight_decay']}")
# Group 0: 1 params, wd=0.01  (weight)
# Group 1: 1 params, wd=0.0   (bias)
```

---

### `build_scheduler`

Create a learning-rate scheduler dict for PyTorch Lightning.

```python
from scmodelforge.training import build_scheduler
from scmodelforge.config.schema import SchedulerConfig

config = SchedulerConfig(name="cosine_warmup", warmup_steps=2000, total_steps=100000)
scheduler_dict = build_scheduler(optimizer, config)
```

#### Signature

```python
build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
) -> dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | `Optimizer` | *required* | The optimizer to schedule |
| `config` | `SchedulerConfig` | *required* | Scheduler configuration (name, warmup_steps, total_steps) |

**Returns:**
- `dict` — Lightning scheduler dict with keys `scheduler`, `interval="step"`, `frequency=1`

**Raises:**
- `ValueError` — If `config.name` is not a supported scheduler

**Supported schedulers:**
- `"cosine_warmup"` — Linear warmup followed by cosine decay to 0
- `"cosine"` — Cosine decay without warmup
- `"linear"` — Linear warmup followed by linear decay to 0

**Example:**

```python
from scmodelforge.config.schema import OptimizerConfig, SchedulerConfig
from scmodelforge.training import build_optimizer, build_scheduler
import torch.nn as nn

model = nn.Linear(512, 1000)
opt_cfg = OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.01)
optimizer = build_optimizer(model, opt_cfg)

sched_cfg = SchedulerConfig(
    name="cosine_warmup",
    warmup_steps=2000,
    total_steps=100000,
)
scheduler_dict = build_scheduler(optimizer, sched_cfg)

# Use in Lightning configure_optimizers
return {
    "optimizer": optimizer,
    "lr_scheduler": scheduler_dict,
}
```

---

### `TrainingMetricsLogger`

Lightning callback that logs throughput and timing metrics during training.

```python
from scmodelforge.training import TrainingMetricsLogger

callback = TrainingMetricsLogger(log_every_n_steps=50)
```

#### Constructor

```python
TrainingMetricsLogger(log_every_n_steps: int = 50)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_every_n_steps` | `int` | `50` | How often to log step-level metrics |

**Logged metrics:**
- `perf/cells_per_sec` — Throughput in cells per second
- `perf/step_time_ms` — Step time in milliseconds
- `perf/epoch_time_sec` — Total epoch time in seconds

**Example:**

```python
from scmodelforge.training import TrainingMetricsLogger
import lightning.pytorch as pl

callback = TrainingMetricsLogger(log_every_n_steps=100)
trainer = pl.Trainer(max_epochs=10, callbacks=[callback])
trainer.fit(model, train_loader, val_loader)
```

---

### `GradientNormLogger`

Lightning callback that logs gradient L2 norm before the optimizer step.

```python
from scmodelforge.training import GradientNormLogger

callback = GradientNormLogger(log_every_n_steps=50)
```

#### Constructor

```python
GradientNormLogger(log_every_n_steps: int = 50)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_every_n_steps` | `int` | `50` | How often to log gradient norms |

**Logged metrics:**
- `train/grad_norm` — L2 norm of all gradients

**Example:**

```python
from scmodelforge.training import GradientNormLogger
import lightning.pytorch as pl

callback = GradientNormLogger(log_every_n_steps=100)
trainer = pl.Trainer(max_epochs=10, callbacks=[callback])
trainer.fit(model, train_loader, val_loader)
```

---

### `SamplerEpochCallback`

Lightning callback that advances epoch-aware samplers at the start of each training epoch. Ensures `WeightedCellSampler` curriculum interpolation and `DistributedShardSampler` shard rotation work correctly.

```python
from scmodelforge.training import SamplerEpochCallback

callback = SamplerEpochCallback(sampler)
```

#### Constructor

```python
SamplerEpochCallback(sampler: object)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampler` | `object` | *required* | Any sampler with a `set_epoch(int)` method |

**Behavior:**
- Calls `sampler.set_epoch(trainer.current_epoch)` in `on_train_epoch_start`
- Automatically wired by `TrainingPipeline` when weighted sampling is configured

---

### `get_environment_info`

Collect runtime environment information for logging and debugging.

```python
from scmodelforge.training._utils import get_environment_info

env = get_environment_info()
print(env)
# {'python': '3.10.12', 'platform': 'Linux-5.15.0', 'torch': '2.1.0', ...}
```

#### Signature

```python
get_environment_info() -> dict[str, str]
```

**Returns:**
- `dict[str, str]` — Environment info with keys: `python`, `platform`, `torch`, `lightning`, `cuda_available`, `gpu_count`, `gpu_name`

---

### `log_training_config`

Log key training configuration values at INFO level.

```python
from scmodelforge.training._utils import log_training_config

log_training_config(config)
# INFO: === Training Configuration ===
# INFO:   Model: transformer_encoder (hidden=512, layers=12, heads=8)
# ...
```

#### Signature

```python
log_training_config(config: ScModelForgeConfig) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `ScModelForgeConfig` | *required* | Full configuration object |

## See Also

- [`scmodelforge.data`](data.md) — Data loading, preprocessing, and gene vocabulary
- [`scmodelforge.tokenizers`](tokenizers.md) — Tokenization strategies and masking
- [`scmodelforge.models`](models.md) — Model architectures (TransformerEncoder, etc.)
- [`scmodelforge.eval`](eval.md) — Evaluation benchmarks and metrics
- [`scmodelforge.finetuning`](finetuning.md) — Fine-tuning pretrained backbones
- [`scmodelforge.config`](config.md) — YAML configuration reference
