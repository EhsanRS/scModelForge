# Getting Started

## Installation

Install scModelForge from PyPI:

```bash
pip install scModelForge
```

For development:

```bash
git clone https://github.com/EhsanRS/scModelForge.git
cd scModelForge
pip install -e ".[dev]"
```

### Optional dependencies

```bash
# CELLxGENE Census data source
pip install "scModelForge[census]"

# Evaluation benchmarks (scIB, pertpy)
pip install "scModelForge[eval]"

# LoRA parameter-efficient fine-tuning
pip install "scModelForge[peft]"

# Weights & Biases logging
pip install "scModelForge[wandb]"

# Everything
pip install "scModelForge[all]"
```

## Features

scModelForge is a complete toolkit for pretraining and fine-tuning single-cell foundation models:

### Data Loading & Preprocessing
- Local AnnData files or CELLxGENE Census remote access
- Flexible preprocessing pipeline (normalization, HVG selection, log transformation)
- Gene vocabulary management with Ensembl/symbol mapping
- PyTorch Lightning data modules with automatic train/val splitting

### Tokenization Strategies
- **Rank-value tokenization** (Geneformer): Rank genes by expression, use ranks as tokens
- **Binned expression tokenization** (scGPT): Discretize expression into bins
- **Continuous projection tokenization** (TranscriptFormer): Continuous expression values
- BERT-style masking for pretraining (80/10/10 split)

### Model Architectures
- **TransformerEncoder**: BERT-style bidirectional encoder with masked gene prediction
- **AutoregressiveTransformer**: GPT-style causal decoder with gene + expression bin prediction
- **MaskedAutoencoder**: Asymmetric encoder-decoder with masked reconstruction
- Modular components: gene embeddings, expression embeddings, pooling, prediction heads

### Training Pipeline
- PyTorch Lightning integration with distributed training, mixed precision, and gradient accumulation
- AdamW and Adam optimizers with weight decay and parameter groups
- Cosine warmup, cosine, and linear learning rate schedules
- Automatic checkpointing (save top-k) and Weights & Biases logging
- Custom callbacks for training metrics (cells/sec, step time) and gradient norms

### Evaluation & Benchmarking
- **Linear probe**: Cell type classification on frozen embeddings (accuracy, F1)
- **Embedding quality**: scIB metrics (NMI, ARI, ASW) for biological conservation and batch correction
- Evaluation harness for running multiple benchmarks
- Lightning callback for periodic evaluation during training

### Fine-tuning
- Transfer learning from pretrained backbones to downstream tasks
- Classification and regression task heads
- Discriminative learning rates (separate LR for backbone vs. head)
- Gradual unfreezing (freeze backbone for N epochs, then unfreeze)
- LoRA adapters for parameter-efficient fine-tuning

## Quick Start

### 1. Pretraining

Create a YAML config file (e.g., `train_config.yaml`):

```yaml
data:
  source: local
  paths:
    - ./data/pbmc_10k.h5ad
  gene_vocab: human_protein_coding
  preprocessing:
    normalize: library_size
    target_sum: 10000
    log1p: true

tokenizer:
  strategy: rank_value
  max_genes: 2048
  prepend_cls: true
  masking:
    mask_ratio: 0.15

model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 12
  num_heads: 8
  dropout: 0.1
  pretraining_task: masked_gene_prediction

training:
  batch_size: 64
  max_epochs: 10
  seed: 42
  num_gpus: 4
  precision: bf16-mixed
  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 0.01
  scheduler:
    name: cosine_warmup
    warmup_steps: 2000
  logger: wandb
  wandb_project: scmodelforge
  checkpoint_dir: ./checkpoints
```

Run training:

```bash
scmodelforge train --config train_config.yaml
```

Resume from checkpoint:

```bash
scmodelforge train --config train_config.yaml --resume checkpoints/last.ckpt
```

### 2. Evaluation

Create an evaluation config (e.g., `eval_config.yaml`):

```yaml
tokenizer:
  strategy: rank_value
  max_genes: 2048
  prepend_cls: true

model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 12
  num_heads: 8

eval:
  batch_size: 256
  benchmarks:
    - name: linear_probe
      dataset: test
      params:
        cell_type_key: cell_type
        test_size: 0.2
    - name: embedding_quality
      dataset: test
      params:
        cell_type_key: cell_type
        batch_key: batch
```

Run evaluation:

```bash
scmodelforge benchmark \
  --config eval_config.yaml \
  --model checkpoints/best.ckpt \
  --data data/test_dataset.h5ad \
  --output results/metrics.json
```

### 3. Fine-tuning

Create a fine-tuning config (e.g., `finetune_config.yaml`):

```yaml
data:
  source: local
  paths:
    - ./data/labeled_pbmc.h5ad

tokenizer:
  strategy: rank_value
  max_genes: 2048
  prepend_cls: true

model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 12
  num_heads: 8

training:
  batch_size: 32
  max_epochs: 5
  optimizer:
    name: adamw
    lr: 1.0e-5
  logger: wandb
  checkpoint_dir: ./checkpoints_finetuned

finetune:
  label_key: cell_type
  freeze_backbone: false
  freeze_backbone_epochs: 2
  head:
    task: classification
    n_classes: null  # Inferred from data
  backbone_lr: 1.0e-5
  head_lr: 1.0e-4
  lora:
    enabled: false  # Set to true for LoRA fine-tuning
    rank: 8
    alpha: 16
```

Run fine-tuning:

```bash
scmodelforge finetune \
  --config finetune_config.yaml \
  --checkpoint checkpoints/best.ckpt
```

## Concepts

### Gene Vocabulary

A gene vocabulary maps between gene identifiers (Ensembl IDs, gene symbols) and integer token IDs. The vocabulary is built from the input data and determines the model's gene embedding layer size. scModelForge supports predefined vocabularies (e.g., `human_protein_coding`) and custom vocabularies from data.

### Tokenization Strategies

Single-cell expression data must be converted to discrete token sequences for transformer models. Three strategies are provided:

- **Rank-value tokenization** (Geneformer): Sort genes by expression level and use their rank as position. Token IDs are gene IDs from the vocabulary. Simple and effective.
- **Binned expression tokenization** (scGPT): Discretize expression values into bins (e.g., 51 bins). Model predicts both gene ID and expression bin. Captures expression magnitude.
- **Continuous projection tokenization** (TranscriptFormer): Use continuous expression values projected through a learned embedding. No discretization loss.

### Model Architectures

Three pretraining architectures are available:

- **TransformerEncoder**: BERT-style bidirectional transformer with masked gene prediction. Best for learning contextual gene relationships.
- **AutoregressiveTransformer**: GPT-style causal decoder that predicts next gene and expression level. Natural for generation tasks.
- **MaskedAutoencoder**: Encoder processes only unmasked tokens, decoder reconstructs masked positions. Efficient for large-scale pretraining.

All models provide an `encode()` method for extracting cell embeddings for downstream tasks.

### Pretraining Tasks

Models are pretrained using self-supervised objectives:

- **Masked gene prediction**: Randomly mask 15% of genes, predict their identity (classification loss).
- **Expression reconstruction**: Predict expression values at masked positions (regression loss, MSE).
- **Dual prediction**: Predict both gene ID and binned expression (combined classification losses).

The pretraining task is configured via `model.pretraining_task` in the config.

### Fine-tuning Strategies

Transfer learning from pretrained models to downstream tasks:

- **Full fine-tuning**: Update all model parameters. Best accuracy but highest memory.
- **Frozen backbone**: Freeze encoder, train only task head. Fast but limited adaptation.
- **Gradual unfreezing**: Freeze backbone for N epochs, then unfreeze. Balances speed and quality.
- **Discriminative learning rates**: Lower LR for backbone, higher for head. Prevents catastrophic forgetting.
- **LoRA fine-tuning**: Low-rank adapters for parameter-efficient fine-tuning. Only 0.1-1% of parameters trained.

## Next Steps

- Read the [Architecture Overview](architecture.md) to understand the pipeline design
- Check the [CLI Reference](cli.md) for command-line details
- Browse the [API Reference](api/config.md) for configuration options
- See [configs/examples/geneformer_basic.yaml](https://github.com/EhsanRS/scModelForge/blob/main/configs/examples/geneformer_basic.yaml) for a complete example
