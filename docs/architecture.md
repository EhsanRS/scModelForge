# Architecture Overview

scModelForge provides a modular, config-driven pipeline for pretraining single-cell foundation models. The toolkit follows PyTorch Lightning conventions and integrates seamlessly with the scverse ecosystem (AnnData, scanpy).

## Pipeline Flow

The training pipeline follows this sequential data flow:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CONFIGURATION                                                │
│    YAML → OmegaConf → ScModelForgeConfig                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. DATA LOADING                                                 │
│    AnnData (local or CELLxGENE Census) → PreprocessingPipeline  │
│    → GeneVocab → CellDataset → CellDataLoader                  │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. TOKENIZATION                                                 │
│    Raw expression → Tokenizer (rank-value, binned, continuous) │
│    → TokenizedCell → MaskingStrategy → MaskedTokenizedCell     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. MODEL FORWARD PASS                                           │
│    Batched tokens → Model (TransformerEncoder, Autoregressive,  │
│    MaskedAutoencoder) → Embeddings + Predictions                │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. TRAINING                                                     │
│    Lightning: loss computation → optimizer step → logging       │
│    Callbacks: metrics, gradient norms, evaluation harness       │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. EVALUATION                                                   │
│    Checkpoint → extract embeddings → benchmarks (linear probe,  │
│    embedding quality) → metrics logged to wandb/tensorboard     │
└─────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### Configuration (`scmodelforge.config`)

The config module provides strongly-typed dataclasses for all pipeline parameters. YAML files are parsed via OmegaConf into validated configuration objects. This ensures type safety and enables IDE autocomplete for configuration editing.

**Key components:** `ScModelForgeConfig`, `DataConfig`, `TokenizerConfig`, `ModelConfig`, `TrainingConfig`, `EvalConfig`, `FinetuneConfig`

### Data (`scmodelforge.data`)

The data module handles loading, preprocessing, and batching of single-cell RNA-seq data. It supports both local AnnData files and remote access to CELLxGENE Census. The `GeneVocab` class manages gene identifier mapping and vocabulary construction.

**Key components:** `GeneVocab`, `PreprocessingPipeline`, `AnnDataStore`, `CellDataset`, `CellDataLoader`, `load_census_adata()`

### Tokenizers (`scmodelforge.tokenizers`)

Tokenizers convert raw gene expression profiles into discrete or continuous token sequences suitable for transformer models. Three strategies are provided: rank-value (Geneformer), binned expression (scGPT), and continuous projection (TranscriptFormer). The masking module implements BERT-style masking for pretraining.

**Key components:** `BaseTokenizer`, `RankValueTokenizer`, `BinnedExpressionTokenizer`, `ContinuousProjectionTokenizer`, `MaskingStrategy`, tokenizer registry

### Models (`scmodelforge.models`)

The models module provides transformer architectures for single-cell pretraining. All models inherit from `nn.Module` and implement a standard `forward()` and `encode()` interface. The registry pattern enables config-driven model construction.

**Key components:** `TransformerEncoder`, `AutoregressiveTransformer`, `MaskedAutoencoder`, `GeneExpressionEmbedding`, prediction heads, model registry

### Training (`scmodelforge.training`)

The training module orchestrates the end-to-end training loop using PyTorch Lightning. It includes optimizer/scheduler builders, data modules, Lightning modules, and custom callbacks for logging and monitoring. The `TrainingPipeline` class provides a high-level interface for launching training from config.

**Key components:** `TrainingPipeline`, `ScModelForgeLightningModule`, `CellDataModule`, `build_optimizer()`, `build_scheduler()`, callbacks

### Evaluation (`scmodelforge.eval`)

The evaluation module provides benchmarks for assessing pretrained representations. Benchmarks include linear probing for cell type classification and scIB metrics for embedding quality. The `EvalHarness` orchestrates multiple benchmarks, and the `AssessmentCallback` enables periodic evaluation during training.

**Key components:** `EvalHarness`, `LinearProbeBenchmark`, `EmbeddingQualityBenchmark`, `AssessmentCallback`, benchmark registry

### Fine-tuning (`scmodelforge.finetuning`)

The fine-tuning module enables transfer learning from pretrained backbones to downstream tasks. It supports discriminative learning rates, gradual unfreezing, and LoRA adapters for parameter-efficient fine-tuning. Task heads include classification and regression.

**Key components:** `FineTunePipeline`, `FineTuneModel`, `FineTuneLightningModule`, `ClassificationHead`, `RegressionHead`, LoRA utilities

## Design Patterns

### Registry Pattern

Tokenizers, models, and benchmarks use a registry pattern for config-driven construction. Each module provides `register_*()`, `get_*()`, and `list_*()` functions. This enables:

- Adding new implementations without modifying core code
- Config-based selection: `tokenizer.strategy = "rank_value"`
- Runtime introspection of available options

```python
from scmodelforge.tokenizers import get_tokenizer, list_tokenizers

print(list_tokenizers())  # ['rank_value', 'binned_expression', 'continuous_projection']
tokenizer = get_tokenizer('rank_value', gene_vocab=vocab, max_len=2048)
```

### Lightning Integration

PyTorch Lightning handles all training infrastructure: distributed training, mixed precision, gradient accumulation, checkpointing, and logging. User code only implements `training_step()` and `validation_step()`. This eliminates boilerplate and ensures best practices.

The `ScModelForgeLightningModule` wraps any model from the registry and handles loss computation, optimizer configuration, and metric logging.

### Config-Driven Philosophy

Every pipeline parameter is specified in a single YAML config file. No command-line argument proliferation. This ensures:

- **Reproducibility:** Config files are self-documenting and version-controllable
- **Composability:** Configs can be merged and overridden via OmegaConf
- **Type safety:** Dataclasses catch misconfigurations at load time

Example workflow:

```bash
# Create config
vim my_experiment.yaml

# Run training
scmodelforge train --config my_experiment.yaml

# Resume from checkpoint
scmodelforge train --config my_experiment.yaml --resume checkpoints/last.ckpt

# Fine-tune
scmodelforge finetune --config finetune_config.yaml --checkpoint checkpoints/best.ckpt

# Evaluate
scmodelforge benchmark --config my_experiment.yaml --model checkpoints/best.ckpt --data test_data.h5ad
```

## Extension Points

To add a new tokenization strategy:

1. Subclass `BaseTokenizer` and implement `tokenize()`
2. Register it: `register_tokenizer("my_tokenizer", MyTokenizer)`
3. Use it: set `tokenizer.strategy = "my_tokenizer"` in config

The same pattern applies to models (subclass `nn.Module`, register, configure) and benchmarks (subclass `BaseBenchmark`, register, configure).
