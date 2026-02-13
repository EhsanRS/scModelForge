# Configuration Module (`scmodelforge.config`)

YAML-based hierarchical configuration system for scModelForge.

## Overview

The `scmodelforge.config` module implements a structured, type-safe configuration system built on OmegaConf. All aspects of the toolkit—from data preprocessing through model architecture to training hyperparameters—are controlled via YAML configuration files that map to Python dataclasses. This design enforces validation at load time, enables easy hyperparameter sweeps, and provides a single source of truth for experiment reproducibility.

The configuration hierarchy reflects the modular architecture of scModelForge. At the top level, `ScModelForgeConfig` aggregates six major sections: `data` (loading and preprocessing), `tokenizer` (tokenization strategy and masking), `model` (architecture and dimensions), `training` (optimization and logging), `eval` (benchmark selection and frequency), and `finetune` (task-specific fine-tuning). Each section is itself a structured dataclass with typed fields, default values, and optional sub-configurations.

The module follows a YAML-first philosophy: users write concise YAML files specifying only the parameters they want to override, and the system merges these with sensible defaults from the dataclass definitions. The `load_config()` function handles this merge via OmegaConf, returning a fully-instantiated `ScModelForgeConfig` object with all fields validated and populated. This approach eliminates boilerplate, catches configuration errors early, and enables seamless integration with experiment tracking systems like Weights & Biases.

All CLI commands (`train`, `finetune`, `benchmark`) accept a `--config` flag pointing to a YAML file. The configuration system also supports programmatic construction for advanced workflows, enabling users to build configs in Python, override specific fields, and pass them directly to pipelines. The combination of YAML convenience and Python flexibility makes scModelForge configurations both human-friendly and automation-ready.

## Quick Reference

| Class | Description |
|-------|-------------|
| `ScModelForgeConfig` | Top-level configuration combining all modules |
| `DataConfig` | Data loading, preprocessing, and splitting |
| `PreprocessingConfig` | Normalization, HVG selection, log transformation |
| `CensusConfig` | CELLxGENE Census data source parameters |
| `TokenizerConfig` | Tokenization strategy, masking, and binning |
| `MaskingConfig` | BERT-style masking ratios |
| `ModelConfig` | Model architecture, dimensions, layers, heads |
| `TrainingConfig` | Training loop, optimizer, scheduler, logging |
| `OptimizerConfig` | Optimizer type, learning rate, weight decay |
| `SchedulerConfig` | Learning rate scheduler with warmup |
| `EvalConfig` | Evaluation benchmarks and frequency |
| `FinetuneConfig` | Fine-tuning task, checkpoint, freezing strategy |
| `TaskHeadConfig` | Task head architecture (classification/regression) |
| `LoRAConfig` | LoRA adapter parameters for efficient fine-tuning |
| `load_config()` | Load and validate a config from a YAML file |

## Detailed API

### `ScModelForgeConfig`

```python
from scmodelforge.config import ScModelForgeConfig
```

Top-level configuration combining all modules. This is the root object returned by `load_config()`.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataConfig` | `DataConfig()` | Data loading and preprocessing configuration |
| `tokenizer` | `TokenizerConfig` | `TokenizerConfig()` | Tokenization strategy and masking |
| `model` | `ModelConfig` | `ModelConfig()` | Model architecture and dimensions |
| `training` | `TrainingConfig` | `TrainingConfig()` | Training loop and optimization |
| `eval` | `EvalConfig` | `EvalConfig()` | Evaluation benchmarks and frequency |
| `finetune` | `FinetuneConfig \| None` | `None` | Fine-tuning configuration (None for pretraining) |

#### Example

```python
from scmodelforge.config import load_config

# Load from YAML
config = load_config("configs/geneformer_basic.yaml")

# Access nested fields
print(config.model.hidden_dim)  # 512
print(config.training.batch_size)  # 64
print(config.tokenizer.strategy)  # "rank_value"

# Programmatic construction
from scmodelforge.config import ScModelForgeConfig, ModelConfig, TrainingConfig

config = ScModelForgeConfig(
    model=ModelConfig(hidden_dim=768, num_layers=12),
    training=TrainingConfig(batch_size=128, max_epochs=20)
)
```

---

### `DataConfig`

```python
from scmodelforge.config import DataConfig
```

Configuration for data loading and preprocessing.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | `"local"` | Data source: `"local"` or `"cellxgene_census"` |
| `paths` | `list[str]` | `[]` | List of paths to local AnnData `.h5ad` files |
| `gene_vocab` | `str` | `"human_protein_coding"` | Gene vocabulary name |
| `preprocessing` | `PreprocessingConfig` | `PreprocessingConfig()` | Preprocessing pipeline configuration |
| `max_genes` | `int` | `2048` | Maximum number of genes per cell |
| `num_workers` | `int` | `4` | Number of data loader workers |
| `census` | `CensusConfig` | `CensusConfig()` | CELLxGENE Census-specific configuration |

#### Example

```yaml
data:
  source: local
  paths:
    - data/pbmc_3k.h5ad
    - data/heart_1k.h5ad
  gene_vocab: human_protein_coding
  max_genes: 2048
  num_workers: 8
  preprocessing:
    normalize: library_size
    target_sum: 10000
    hvg_selection: null
    log1p: true
```

---

### `PreprocessingConfig`

```python
from scmodelforge.config import PreprocessingConfig
```

Preprocessing options applied to raw expression data.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `str \| None` | `"library_size"` | Normalization method (`"library_size"` or `None`) |
| `target_sum` | `float \| None` | `1e4` | Target sum for library size normalization |
| `hvg_selection` | `int \| None` | `None` | Number of highly variable genes to select (None = all genes) |
| `log1p` | `bool` | `True` | Whether to apply log1p transformation |

---

### `CensusConfig`

```python
from scmodelforge.config import CensusConfig
```

CELLxGENE Census data source configuration. Used when `DataConfig.source = "cellxgene_census"`.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `organism` | `str` | `"Homo sapiens"` | Census organism name (`"Homo sapiens"` or `"Mus musculus"`) |
| `census_version` | `str` | `"latest"` | Census release version or `"latest"` |
| `obs_value_filter` | `str \| None` | `None` | Raw SOMA `obs_value_filter` string (takes precedence over `filters`) |
| `var_value_filter` | `str \| None` | `None` | Raw SOMA `var_value_filter` string for gene filtering |
| `filters` | `dict[str, Any] \| None` | `None` | Structured filter dict auto-converted to SOMA filter string |
| `obs_columns` | `list[str] \| None` | `None` | Additional `obs` metadata columns to include |

#### Example

```yaml
data:
  source: cellxgene_census
  census:
    organism: Homo sapiens
    census_version: latest
    filters:
      tissue: ["brain", "lung"]
      is_primary_data: true
    obs_columns: ["tissue", "donor_id", "assay"]
```

---

### `TokenizerConfig`

```python
from scmodelforge.config import TokenizerConfig
```

Configuration for tokenization strategy.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | `"rank_value"` | Tokenizer type: `"rank_value"`, `"binned_expression"`, `"continuous_projection"`, `"gene_embedding"` |
| `max_genes` | `int` | `2048` | Maximum sequence length (genes per cell) |
| `gene_vocab` | `str` | `"human_protein_coding"` | Gene vocabulary name |
| `prepend_cls` | `bool` | `True` | Whether to prepend a `[CLS]` token |
| `n_bins` | `int` | `51` | Number of expression bins (for `binned_expression` strategy) |
| `binning_method` | `str` | `"uniform"` | Binning method: `"uniform"` or `"quantile"` (for `binned_expression` strategy) |
| `embedding_path` | `str \| None` | `None` | Path to pretrained gene embedding file (for `gene_embedding` strategy) |
| `embedding_dim` | `int` | `200` | Expected embedding dimension (for `gene_embedding` strategy) |
| `masking` | `MaskingConfig` | `MaskingConfig()` | Masking strategy configuration |

All strategy-specific fields (`n_bins`, `binning_method`, `embedding_path`, `embedding_dim`) are automatically propagated to the tokenizer constructor via `build_tokenizer_kwargs()` when using the training pipeline, fine-tuning pipeline, or CLI.

#### Example

```yaml
# Rank-value tokenizer (Geneformer-style)
tokenizer:
  strategy: rank_value
  max_genes: 2048
  prepend_cls: true
  masking:
    mask_ratio: 0.15
    random_replace_ratio: 0.1
    keep_ratio: 0.1
```

```yaml
# Binned expression tokenizer (scGPT-style)
tokenizer:
  strategy: binned_expression
  max_genes: 2048
  n_bins: 51
  binning_method: uniform
  prepend_cls: true
```

```yaml
# Gene embedding tokenizer
tokenizer:
  strategy: gene_embedding
  max_genes: 2048
  embedding_path: ./pretrained/gene2vec.npy
  embedding_dim: 200
  prepend_cls: true
```

---

### `MaskingConfig`

```python
from scmodelforge.config import MaskingConfig
```

BERT-style masking strategy for pretraining.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask_ratio` | `float` | `0.15` | Fraction of tokens to mask |
| `random_replace_ratio` | `float` | `0.1` | Fraction of masked tokens to replace with random genes |
| `keep_ratio` | `float` | `0.1` | Fraction of masked tokens to keep unchanged |

Standard BERT masking: 80% replaced with `[MASK]`, 10% random, 10% unchanged.

---

### `ModelConfig`

```python
from scmodelforge.config import ModelConfig
```

Configuration for model architecture.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `architecture` | `str` | `"transformer_encoder"` | Model type: `"transformer_encoder"`, `"autoregressive_transformer"`, `"masked_autoencoder"` |
| `hidden_dim` | `int` | `512` | Hidden dimension / embedding size |
| `num_layers` | `int` | `12` | Number of transformer layers |
| `num_heads` | `int` | `8` | Number of attention heads |
| `ffn_dim` | `int \| None` | `None` | Feedforward dimension (default: `4 * hidden_dim`) |
| `dropout` | `float` | `0.1` | Dropout probability |
| `max_seq_len` | `int` | `2048` | Maximum sequence length |
| `pooling` | `str` | `"cls"` | Pooling strategy: `"cls"` or `"mean"` |
| `activation` | `str` | `"gelu"` | Activation function: `"gelu"` or `"relu"` |
| `use_expression_values` | `bool` | `True` | Whether to use continuous expression values |
| `pretraining_task` | `str` | `"masked_gene_prediction"` | Pretraining task type |
| `mask_ratio` | `float` | `0.15` | Masking ratio for pretraining |
| `vocab_size` | `int \| None` | `None` | Vocabulary size (inferred from gene vocab if None) |
| `n_bins` | `int` | `51` | Number of expression bins (for `autoregressive_transformer`) |
| `gene_loss_weight` | `float` | `1.0` | Weight for gene prediction loss (autoregressive) |
| `expression_loss_weight` | `float` | `1.0` | Weight for expression prediction loss (autoregressive) |
| `decoder_dim` | `int \| None` | `None` | Decoder hidden dimension (masked autoencoder, default: `hidden_dim // 2`) |
| `decoder_layers` | `int` | `4` | Number of decoder layers (masked autoencoder) |
| `decoder_heads` | `int \| None` | `None` | Number of decoder attention heads (default: `num_heads`) |

#### Example

```yaml
model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 12
  num_heads: 8
  ffn_dim: 2048
  dropout: 0.1
  max_seq_len: 2048
  pooling: cls
  activation: gelu
  use_expression_values: true
  pretraining_task: masked_gene_prediction
  mask_ratio: 0.15
```

---

### `TrainingConfig`

```python
from scmodelforge.config import TrainingConfig
```

Configuration for the training loop.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `64` | Training batch size |
| `max_epochs` | `int` | `10` | Maximum number of epochs |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `strategy` | `str` | `"ddp"` | Distributed training strategy: `"ddp"`, `"ddp_spawn"`, `"auto"` |
| `num_gpus` | `int \| None` | `None` | Number of GPUs (None = auto-detect) |
| `precision` | `str` | `"bf16-mixed"` | Training precision: `"32"`, `"16-mixed"`, `"bf16-mixed"` |
| `optimizer` | `OptimizerConfig` | `OptimizerConfig()` | Optimizer configuration |
| `scheduler` | `SchedulerConfig \| None` | `SchedulerConfig()` | Scheduler configuration (None = no scheduler) |
| `gradient_clip` | `float` | `1.0` | Gradient clipping max norm |
| `gradient_accumulation` | `int` | `1` | Number of gradient accumulation steps |
| `logger` | `str` | `"wandb"` | Logger type: `"wandb"`, `"tensorboard"`, `"csv"` |
| `wandb_project` | `str` | `"scmodelforge"` | Weights & Biases project name |
| `run_name` | `str \| None` | `None` | Run name (None = auto-generate) |
| `log_dir` | `str` | `"logs"` | Directory for logs |
| `log_every_n_steps` | `int` | `50` | Logging frequency (steps) |
| `checkpoint_dir` | `str` | `"checkpoints"` | Directory for checkpoints |
| `save_top_k` | `int` | `3` | Number of best checkpoints to keep |
| `num_workers` | `int` | `4` | Number of data loader workers |
| `val_split` | `float` | `0.05` | Validation split fraction |
| `resume_from` | `str \| None` | `None` | Path to checkpoint to resume from |

#### Example

```yaml
training:
  batch_size: 128
  max_epochs: 50
  seed: 42
  strategy: ddp
  num_gpus: 4
  precision: bf16-mixed
  optimizer:
    name: adamw
    lr: 1e-4
    weight_decay: 0.01
  scheduler:
    name: cosine_warmup
    warmup_steps: 2000
    total_steps: 100000
  gradient_clip: 1.0
  gradient_accumulation: 2
  logger: wandb
  wandb_project: scmodelforge
  run_name: geneformer_large
  log_every_n_steps: 50
  checkpoint_dir: checkpoints
  save_top_k: 3
  val_split: 0.05
```

---

### `OptimizerConfig`

```python
from scmodelforge.config import OptimizerConfig
```

Optimizer configuration.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"adamw"` | Optimizer name: `"adamw"` or `"adam"` |
| `lr` | `float` | `1e-4` | Learning rate |
| `weight_decay` | `float` | `0.01` | Weight decay (L2 regularization) |

---

### `SchedulerConfig`

```python
from scmodelforge.config import SchedulerConfig
```

Learning rate scheduler configuration.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"cosine_warmup"` | Scheduler type: `"cosine_warmup"`, `"cosine"`, `"linear"` |
| `warmup_steps` | `int` | `2000` | Number of warmup steps |
| `total_steps` | `int` | `100000` | Total training steps |

---

### `EvalConfig`

```python
from scmodelforge.config import EvalConfig
```

Configuration for evaluation benchmarks.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `every_n_epochs` | `int` | `2` | Run evaluation every N epochs |
| `batch_size` | `int` | `256` | Batch size for embedding extraction |
| `benchmarks` | `list[Any]` | `[]` | List of benchmark specs (strings or dicts) |

#### Example

```yaml
eval:
  every_n_epochs: 2
  batch_size: 256
  benchmarks:
    - linear_probe
    - name: embedding_quality
      batch_key: donor
    - name: perturbation
      n_top_genes: 100
```

---

### `FinetuneConfig`

```python
from scmodelforge.config import FinetuneConfig
```

Configuration for fine-tuning a pretrained backbone.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | `str` | `""` | Path to pretrained model checkpoint |
| `freeze_backbone` | `bool` | `False` | If True, freeze all backbone parameters throughout training |
| `freeze_backbone_epochs` | `int` | `0` | Unfreeze backbone after this many epochs (0 = no gradual unfreezing) |
| `label_key` | `str` | `"cell_type"` | Column in `adata.obs` containing task labels |
| `head` | `TaskHeadConfig` | `TaskHeadConfig()` | Task head configuration |
| `backbone_lr` | `float \| None` | `None` | Discriminative LR for backbone (None = use optimizer LR) |
| `head_lr` | `float \| None` | `None` | Discriminative LR for head (None = use optimizer LR) |
| `lora` | `LoRAConfig` | `LoRAConfig()` | LoRA adapter configuration |

#### Example

```yaml
finetune:
  checkpoint_path: checkpoints/pretrained_epoch10.ckpt
  freeze_backbone: false
  freeze_backbone_epochs: 3
  label_key: cell_type
  head:
    task: classification
    n_classes: null  # Inferred from data
    hidden_dim: 256
    dropout: 0.2
  backbone_lr: 1e-5
  head_lr: 1e-3
  lora:
    enabled: false
```

---

### `TaskHeadConfig`

```python
from scmodelforge.config import TaskHeadConfig
```

Configuration for a fine-tuning task head.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `"classification"` | Task type: `"classification"` or `"regression"` |
| `n_classes` | `int \| None` | `None` | Number of output classes (classification only, inferred if None) |
| `output_dim` | `int` | `1` | Output dimension for regression tasks |
| `hidden_dim` | `int \| None` | `None` | Optional hidden layer dimension (None = direct projection) |
| `dropout` | `float` | `0.1` | Dropout probability in the head |

---

### `LoRAConfig`

```python
from scmodelforge.config import LoRAConfig
```

LoRA (Low-Rank Adaptation) adapter configuration for parameter-efficient fine-tuning.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `False` | Whether to apply LoRA adapters |
| `rank` | `int` | `8` | LoRA rank (r). Typical values: 4, 8, 16 |
| `alpha` | `int` | `16` | LoRA scaling factor (alpha). Usually alpha = rank or 2*rank |
| `dropout` | `float` | `0.05` | Dropout applied to LoRA layers |
| `target_modules` | `list[str] \| None` | `None` | Module name patterns to apply LoRA to (None = defaults) |
| `bias` | `str` | `"none"` | Bias handling: `"none"`, `"all"`, or `"lora_only"` |

Default target modules (when `target_modules=None`): `["out_proj", "linear1", "linear2"]`

#### Example

```yaml
finetune:
  checkpoint_path: checkpoints/pretrained.ckpt
  label_key: cell_type
  lora:
    enabled: true
    rank: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["out_proj", "linear1", "linear2"]
    bias: none
```

---

### `load_config()`

```python
from scmodelforge.config import load_config
```

Load a `ScModelForgeConfig` from a YAML file. Merges user-provided config with dataclass defaults and validates all fields.

```python
load_config(path: str | Path) -> ScModelForgeConfig
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to a YAML configuration file |

**Returns:** Validated `ScModelForgeConfig` object.

**Raises:** `ValidationError` if the YAML contains invalid fields or types.

#### Example

```python
from scmodelforge.config import load_config

# Load from YAML
config = load_config("configs/geneformer_basic.yaml")

# Access nested fields
print(f"Model: {config.model.architecture}")
print(f"Hidden dim: {config.model.hidden_dim}")
print(f"Batch size: {config.training.batch_size}")
print(f"Learning rate: {config.training.optimizer.lr}")

# Pass to pipeline
from scmodelforge.training import TrainingPipeline
pipeline = TrainingPipeline(config)
pipeline.run()
```

## Complete YAML Example

Below is a complete configuration file demonstrating all major sections:

```yaml
# Complete scModelForge configuration example

data:
  source: local
  paths:
    - data/pbmc_10k.h5ad
    - data/heart_5k.h5ad
  gene_vocab: human_protein_coding
  max_genes: 2048
  num_workers: 8
  preprocessing:
    normalize: library_size
    target_sum: 10000
    hvg_selection: null
    log1p: true
  census:
    organism: Homo sapiens
    census_version: latest
    filters: null
    obs_columns: null

tokenizer:
  strategy: rank_value
  max_genes: 2048
  gene_vocab: human_protein_coding
  prepend_cls: true
  # Binned expression options (only used if strategy: binned_expression)
  n_bins: 51
  binning_method: uniform
  # Gene embedding options (only used if strategy: gene_embedding)
  embedding_path: null
  embedding_dim: 200
  masking:
    mask_ratio: 0.15
    random_replace_ratio: 0.1
    keep_ratio: 0.1

model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 12
  num_heads: 8
  ffn_dim: 2048
  dropout: 0.1
  max_seq_len: 2048
  pooling: cls
  activation: gelu
  use_expression_values: true
  pretraining_task: masked_gene_prediction
  mask_ratio: 0.15
  vocab_size: null
  # Autoregressive model options (only used if architecture: autoregressive_transformer)
  n_bins: 51
  gene_loss_weight: 1.0
  expression_loss_weight: 1.0
  # Masked autoencoder options (only used if architecture: masked_autoencoder)
  decoder_dim: null  # Defaults to hidden_dim // 2
  decoder_layers: 4
  decoder_heads: null  # Defaults to num_heads

training:
  batch_size: 128
  max_epochs: 50
  seed: 42
  strategy: ddp
  num_gpus: 4
  precision: bf16-mixed
  optimizer:
    name: adamw
    lr: 1e-4
    weight_decay: 0.01
  scheduler:
    name: cosine_warmup
    warmup_steps: 2000
    total_steps: 100000
  gradient_clip: 1.0
  gradient_accumulation: 2
  logger: wandb
  wandb_project: scmodelforge
  run_name: geneformer_base_v1
  log_dir: logs
  log_every_n_steps: 50
  checkpoint_dir: checkpoints
  save_top_k: 3
  num_workers: 8
  val_split: 0.05
  resume_from: null

eval:
  every_n_epochs: 2
  batch_size: 256
  benchmarks:
    - linear_probe
    - name: embedding_quality
      cell_type_key: cell_type
      batch_key: donor
      n_neighbors: 15
    - name: perturbation
      perturbation_key: perturbation
      control_label: control
      n_top_genes: 50
      test_fraction: 0.2

# Fine-tuning config (optional, for finetune command)
finetune:
  checkpoint_path: checkpoints/best_model.ckpt
  freeze_backbone: false
  freeze_backbone_epochs: 3
  label_key: cell_type
  head:
    task: classification
    n_classes: null  # Inferred from data
    output_dim: 1
    hidden_dim: 256
    dropout: 0.2
  backbone_lr: 1e-5
  head_lr: 1e-3
  lora:
    enabled: true
    rank: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["out_proj", "linear1", "linear2"]
    bias: none
```

## Configuration Workflow

### 1. Minimal Config (Pretraining)

For simple pretraining with defaults, you only need to specify data paths:

```yaml
data:
  paths:
    - data/my_dataset.h5ad

training:
  max_epochs: 20
  wandb_project: my_project
```

All other parameters will use sensible defaults.

### 2. Custom Model Architecture

Override specific model parameters while keeping other defaults:

```yaml
data:
  paths:
    - data/large_dataset.h5ad

model:
  hidden_dim: 768
  num_layers: 16
  num_heads: 12

training:
  batch_size: 256
  max_epochs: 100
```

### 3. Fine-tuning Config

Add a `finetune` section for fine-tuning workflows:

```yaml
data:
  paths:
    - data/labeled_cells.h5ad

finetune:
  checkpoint_path: checkpoints/pretrained.ckpt
  label_key: disease_status
  head:
    task: classification
    hidden_dim: 128
  backbone_lr: 1e-5
  head_lr: 1e-3

training:
  batch_size: 64
  max_epochs: 10
```

### 4. CELLxGENE Census Config

Load data directly from CELLxGENE Census:

```yaml
data:
  source: cellxgene_census
  census:
    organism: Homo sapiens
    census_version: latest
    filters:
      tissue: ["brain", "heart"]
      disease: ["normal"]
      is_primary_data: true
    obs_columns: ["tissue", "donor_id", "cell_type"]

training:
  max_epochs: 50
  wandb_project: census_pretraining
```

## CLI Integration

All scModelForge commands accept `--config`:

```bash
# Pretraining
scmodelforge train --config configs/pretraining.yaml

# Fine-tuning
scmodelforge finetune --config configs/finetune.yaml

# Standalone evaluation
scmodelforge benchmark --config configs/eval.yaml \
  --model checkpoints/best.ckpt \
  --data data/test_set.h5ad \
  --output results/
```

## Programmatic Usage

Configs can also be built and modified programmatically:

```python
from scmodelforge.config import (
    ScModelForgeConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_config
)

# Load base config
config = load_config("configs/base.yaml")

# Override specific fields
config.training.batch_size = 256
config.training.max_epochs = 100
config.model.num_layers = 24

# Or build from scratch
config = ScModelForgeConfig(
    data=DataConfig(paths=["data/my_data.h5ad"]),
    model=ModelConfig(hidden_dim=768, num_layers=16),
    training=TrainingConfig(batch_size=128, max_epochs=50)
)

# Pass to pipeline
from scmodelforge.training import TrainingPipeline
pipeline = TrainingPipeline(config)
pipeline.run()
```

## See Also

- **Loading Data**: `scmodelforge.data` for data loading and preprocessing
- **Tokenization**: `scmodelforge.tokenizers` for tokenization strategies
- **Models**: `scmodelforge.models` for model architectures
- **Training**: `scmodelforge.training.TrainingPipeline` for pretraining
- **Fine-tuning**: `scmodelforge.finetuning.FineTunePipeline` for task-specific fine-tuning
- **Evaluation**: `scmodelforge.eval` for benchmarks and evaluation
- **CLI Reference**: `scmodelforge --help` for command-line usage
