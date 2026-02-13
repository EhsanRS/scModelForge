# Pretraining a Foundation Model

This tutorial guides you through pretraining a single-cell foundation model from scratch using scModelForge. You will learn how to configure datasets, choose tokenization strategies and model architectures, run training, and evaluate your model.

## What is Pretraining?

In computational biology, we often work with labeled datasets where each cell has annotations like cell type, disease state, or treatment condition. However, generating these labels requires expert knowledge and experimental work. Pretraining takes a different approach: the model learns directly from unlabeled gene expression data.

The key insight is similar to how word embeddings like word2vec learned language patterns from raw text. By training on millions of cells, the model learns:

- Which genes are co-expressed together
- How gene expression patterns relate to biological states
- General representations of cellular identity that transfer across tasks

scModelForge implements this through **masked gene prediction**: the model randomly masks out 15% of genes in each cell, then tries to predict which genes were masked based on the remaining expression context. This forces the model to learn gene-gene relationships and expression patterns without requiring any labels.

After pretraining, you can fine-tune the model on smaller labeled datasets for specific tasks like cell type annotation, perturbation response prediction, or batch integration.

## Choosing Your Dataset

The first decision is what data to pretrain on. Some key considerations:

**Dataset size vs model size tradeoffs:**

- Small models (256-512 dim, 6 layers): 10,000-100,000 cells
- Medium models (512-768 dim, 12 layers): 100,000-1,000,000 cells
- Large models (768-1024 dim, 24+ layers): 1,000,000+ cells

For this tutorial, we will use a moderately-sized dataset from **CELLxGENE Census** containing approximately 50,000 human lung cells from Tabula Sapiens. This is large enough to demonstrate the workflow but small enough to train on a single GPU.

### Option 1: Fetch from CELLxGENE Census

The CELLxGENE Census provides access to millions of quality-controlled single-cell profiles across tissues and studies. To use Census data, configure your data source with a structured query:

```yaml
data:
  source: census
  census:
    organism: Homo sapiens
    census_version: latest
    filters:
      tissue:
        - lung
      dataset_id: tabula-sapiens
    obs_columns:
      - cell_type
      - tissue
      - donor_id
  gene_vocab: human_protein_coding
  preprocessing:
    normalize: library_size
    target_sum: 10000
    log1p: true
  max_genes: 2048
  num_workers: 4
```

The `filters` section lets you subset by tissue, cell type, disease status, or any other `obs` field in Census. scModelForge will automatically download and cache the data.

### Option 2: Use a Local H5AD File

If you have your own AnnData file:

```yaml
data:
  source: local
  paths:
    - /path/to/your/dataset.h5ad
  gene_vocab: human_protein_coding
  preprocessing:
    normalize: library_size
    target_sum: 10000
    hvg_selection: 2000  # optional: select top 2000 highly variable genes
    log1p: true
  max_genes: 2048
  num_workers: 4
```

**Key parameters:**

- `gene_vocab`: Controls which genes are included. Options: `human_protein_coding`, `mouse_protein_coding`, or path to a custom gene list.
- `preprocessing.normalize`: Standard scanpy normalization (`library_size` or `median`).
- `preprocessing.target_sum`: Target total counts per cell after normalization (typically 10,000).
- `preprocessing.hvg_selection`: Optional. Select top N highly variable genes to reduce dimensionality.
- `preprocessing.log1p`: Apply log1p transformation (recommended).
- `max_genes`: Maximum genes per cell after tokenization (sets sequence length).
- `num_workers`: Number of CPU workers for data loading.

## Choosing a Tokenization Strategy

Tokenization converts continuous gene expression values into discrete tokens that transformers can process. scModelForge implements four strategies inspired by published single-cell foundation models:

| Strategy | Approach | Best for | Config value |
|----------|----------|----------|--------------|
| **Rank-value** | Geneformer-style: rank genes by expression within each cell, assign rank-based token IDs | General purpose; robust to technical variation | `rank_value` |
| **Binned expression** | scGPT-style: discretize expression into fixed bins (e.g., 0-51), predict both gene identity and expression bin | Expression magnitude matters for your task | `binned_expression` |
| **Continuous projection** | TranscriptFormer-style: project continuous expression values through learned embeddings | Maximum information retention; assumes normalized data | `continuous_projection` |
| **Gene embedding** | Use pretrained gene embeddings (e.g., from gene2vec or pathway databases) | Transfer learning from external gene knowledge | `gene_embedding` |

**Recommendation for starting:** Use `rank_value`. It is the most robust to batch effects and technical noise, and has been validated on millions of cells in the Geneformer paper.

Example tokenizer configuration:

```yaml
tokenizer:
  strategy: rank_value
  max_genes: 2048
  gene_vocab: human_protein_coding
  prepend_cls: true  # Add a [CLS] token for pooling
  masking:
    mask_ratio: 0.15  # Mask 15% of genes (BERT standard)
    random_replace_ratio: 0.1  # 10% of masked tokens replaced with random gene
    keep_ratio: 0.1  # 10% of masked tokens kept unchanged
```

The masking parameters implement BERT-style masking:

- 80% of masked positions: replace with `[MASK]` token
- 10% of masked positions: replace with a random gene
- 10% of masked positions: keep the original gene

This prevents the model from simply learning to recognize `[MASK]` tokens and encourages learning robust gene representations.

## Choosing a Model Architecture

scModelForge provides three transformer architectures, each with different strengths:

### transformer_encoder (BERT-style)

**Best for:** General-purpose pretraining, cell embeddings, most downstream tasks.

- Bidirectional attention: each gene attends to all other genes
- Predicts masked genes using full context
- Produces high-quality cell-level embeddings via `[CLS]` token

```yaml
model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 12
  num_heads: 8
  dropout: 0.1
  max_seq_len: 2048
  pooling: cls  # Use [CLS] token for cell embedding
  activation: gelu
```

### autoregressive_transformer (GPT-style)

**Best for:** Generative modeling, in silico perturbations, expression forecasting.

- Causal attention: each gene only attends to previous genes
- Predicts both next gene identity and expression bin
- Requires `binned_expression` tokenizer

```yaml
model:
  architecture: autoregressive_transformer
  hidden_dim: 512
  num_layers: 12
  num_heads: 8
  dropout: 0.1
  max_seq_len: 2048
  n_bins: 51  # Number of expression bins
  gene_loss_weight: 1.0
  expression_loss_weight: 1.0
```

### masked_autoencoder (MAE-style)

**Best for:** Large-scale pretraining with computational efficiency.

- Encoder processes only unmasked tokens (more efficient)
- Decoder reconstructs full sequence including masked positions
- Asymmetric design: small encoder, tiny decoder

```yaml
model:
  architecture: masked_autoencoder
  hidden_dim: 768
  num_layers: 12
  num_heads: 12
  dropout: 0.1
  max_seq_len: 2048
  decoder_dim: 256
  decoder_layers: 4
  decoder_heads: 4
```

**Recommendation for starting:** Use `transformer_encoder`. It is the most versatile and produces the best embeddings for downstream tasks.

## Writing the Configuration File

Here is a complete, production-ready configuration for pretraining a medium-sized model:

```yaml
# pretrain_lung_model.yaml
# Pretrains a 512-dim transformer on 50k lung cells from Tabula Sapiens
# Expected training time: ~2 hours on a single A100 GPU

# ============================================================================
# DATA: Source and preprocessing
# ============================================================================
data:
  source: census
  census:
    organism: Homo sapiens
    census_version: latest
    filters:
      tissue:
        - lung
      dataset_id: tabula-sapiens
    obs_columns:
      - cell_type
      - tissue
      - donor_id

  gene_vocab: human_protein_coding  # ~19k protein-coding genes

  preprocessing:
    normalize: library_size  # Normalize to total counts
    target_sum: 10000  # Target 10k counts per cell
    log1p: true  # Apply log1p transformation

  max_genes: 2048  # Maximum sequence length (genes per cell)
  num_workers: 4  # Parallel data loading workers

# ============================================================================
# TOKENIZER: How to convert expression to tokens
# ============================================================================
tokenizer:
  strategy: rank_value  # Geneformer-style rank-based tokenization
  max_genes: 2048
  gene_vocab: human_protein_coding
  prepend_cls: true  # Add [CLS] token for pooled embeddings

  masking:
    mask_ratio: 0.15  # Mask 15% of genes (BERT standard)
    random_replace_ratio: 0.1  # 10% replaced with random gene
    keep_ratio: 0.1  # 10% kept unchanged

# ============================================================================
# MODEL: Architecture and hyperparameters
# ============================================================================
model:
  architecture: transformer_encoder  # BERT-style bidirectional model
  hidden_dim: 512  # Embedding dimension
  num_layers: 12  # Number of transformer layers
  num_heads: 8  # Number of attention heads
  dropout: 0.1  # Dropout rate for regularization
  max_seq_len: 2048  # Must match tokenizer.max_genes
  pooling: cls  # Use [CLS] token for cell-level embedding
  activation: gelu  # GELU activation function

# ============================================================================
# TRAINING: Optimization and execution
# ============================================================================
training:
  # Batch and device settings
  batch_size: 64  # Cells per batch (adjust for your GPU memory)
  max_epochs: 10  # Number of training epochs
  seed: 42  # Random seed for reproducibility

  # Distributed training
  strategy: auto  # auto-select: ddp for multi-GPU, single device otherwise
  num_gpus: 1  # Number of GPUs (0 for CPU)
  precision: bf16-mixed  # Mixed precision training (bf16 recommended for A100)

  # Optimization
  optimizer:
    name: adamw  # AdamW optimizer
    lr: 1.0e-4  # Learning rate
    weight_decay: 0.01  # L2 regularization

  scheduler:
    name: cosine_warmup  # Cosine decay with linear warmup
    warmup_steps: 2000  # Warmup for first 2k steps
    total_steps: 100000  # Total training steps (auto-computed if null)

  gradient_clip: 1.0  # Gradient clipping threshold
  gradient_accumulation: 1  # Accumulate gradients over N batches

  # Logging and checkpointing
  logger: wandb  # Use Weights & Biases for logging (or "tensorboard")
  wandb_project: scmodelforge  # W&B project name
  run_name: lung_pretrain_512d  # Descriptive run name
  log_every_n_steps: 50  # Log metrics every 50 steps

  checkpoint_dir: ./checkpoints/lung_pretrain  # Save checkpoints here
  save_top_k: 3  # Keep top 3 checkpoints by validation loss

  # Data splitting
  val_split: 0.05  # Hold out 5% for validation
  num_workers: 4  # Data loader workers

# ============================================================================
# EVAL: Benchmark during training
# ============================================================================
eval:
  every_n_epochs: 2  # Run benchmarks every 2 epochs

  benchmarks:
    # Embedding quality: biological vs batch mixing
    - name: embedding_quality
      dataset: self  # Use the training dataset
      params:
        cell_type_key: cell_type  # Biological label for clustering
        batch_key: donor_id  # Batch label for integration quality

    # Linear probe: cell type classification accuracy
    - name: linear_probe
      dataset: self
      params:
        label_key: cell_type  # What to predict
        test_size: 0.2  # 20% held-out test set
```

**Key configuration sections:**

- **data**: Defines where to get cells and how to preprocess them
- **tokenizer**: Converts expression values to token sequences
- **model**: Neural network architecture and size
- **training**: Optimization hyperparameters, GPU settings, logging
- **eval**: Benchmarks to run during training for quality monitoring

## Running Pretraining

Save the configuration to a file (e.g., `pretrain_lung_model.yaml`), then start training:

```bash
scmodelforge train --config pretrain_lung_model.yaml
```

**What to expect:**

1. Data loading: scModelForge downloads Census data and caches it locally (~1-2 minutes first run)
2. Preprocessing: Normalization and gene vocabulary filtering
3. Model initialization: Transformer weights randomly initialized
4. Training loop begins:
   ```
   Epoch 1/10: 100%|██████████| 782/782 [02:15<00:00,  5.77it/s, loss=3.45]
   Validation: 100%|██████████| 42/42 [00:08<00:00,  5.12it/s]
   Epoch 1: train_loss=3.45, val_loss=3.21, accuracy=0.32
   ```
5. Checkpoints saved to `./checkpoints/lung_pretrain/`
6. Evaluation benchmarks run every 2 epochs
7. Metrics logged to Weights & Biases dashboard

**Expected metrics during training:**

- Initial masked gene prediction accuracy: 30-40% (random is ~0.05%)
- After 10 epochs on 50k cells: 60-70% accuracy
- Validation loss should decrease steadily and plateau

**Resource requirements:**

- Single A100 GPU: ~2 hours for 10 epochs on 50k cells
- Single V100 GPU: ~4 hours
- CPU only: not recommended (20+ hours)

## Monitoring with Weights & Biases

If you set `logger: wandb`, scModelForge automatically logs:

- Loss curves (train and validation)
- Masked gene prediction accuracy
- Learning rate schedule
- Gradient norms
- System metrics (GPU utilization, memory)

View your dashboard at [wandb.ai](https://wandb.ai). You will see charts updating in real-time as training progresses.

To use TensorBoard instead:

```yaml
training:
  logger: tensorboard
  tensorboard_dir: ./tensorboard_logs
```

Then run: `tensorboard --logdir ./tensorboard_logs`

## Resuming from Checkpoint

Training was interrupted? Resume from the last checkpoint:

```bash
scmodelforge train --config pretrain_lung_model.yaml --resume ./checkpoints/lung_pretrain/last.ckpt
```

scModelForge will:

- Restore model weights and optimizer state
- Resume from the last completed epoch
- Continue logging to the same W&B run

You can also resume from a specific checkpoint:

```bash
scmodelforge train --config pretrain_lung_model.yaml --resume ./checkpoints/lung_pretrain/epoch=5-val_loss=2.87.ckpt
```

## Evaluating During Training

The `eval` section in your config controls **automated assessment** during training. scModelForge uses the `AssessmentCallback` to periodically run benchmarks and log metrics.

**Available benchmarks:**

embedding_quality
: Measures how well embeddings capture biological variation vs batch effects. Uses scIB metrics (NMI, ARI, silhouette scores). Higher is better.

linear_probe
: Trains a logistic regression classifier on frozen embeddings to predict cell types. Reports accuracy and F1 score. Indicates how well embeddings separate cell types.

perturbation
: Evaluates perturbation response prediction using ridge regression. Requires perturbation metadata in your dataset.

grn_inference
: Gene regulatory network inference. Computes gene-gene similarity and compares to known networks (e.g., from SCENIC).

Example benchmark output in logs:

```
Epoch 2: running 2 benchmarks on self...
  embedding_quality/self/bio_conservation: 0.68
  embedding_quality/self/batch_correction: 0.82
  embedding_quality/self/overall: 0.73
  linear_probe/self/accuracy: 0.71
  linear_probe/self/f1_macro: 0.69
```

These metrics appear as `assessment/{benchmark_name}/{dataset}/{metric}` in your W&B dashboard.

## When is My Model Done?

Unlike supervised learning where you have a clear target metric, pretraining requires more subjective judgment. Watch for these signals:

**Primary signals:**

1. **Validation loss plateau**: Loss stops decreasing for several epochs
2. **Masked accuracy plateau**: Prediction accuracy stops improving
3. **Benchmark metrics stabilize**: Embedding quality and linear probe scores level off

**Typical training schedules:**

- Small model (10-100k cells): 10-20 epochs, ~10k-50k steps
- Medium model (100k-1M cells): 20-50 epochs, ~50k-200k steps
- Large model (1M+ cells): 50-100 epochs, ~200k-500k steps

**Rule of thumb:** Train until validation metrics plateau for 10-20% of total training time. For the 50k cell example, this happens around epoch 8-10.

**When to stop early:**

- Validation loss increases (overfitting)
- Benchmark metrics degrade
- Gradient norms explode (numerical instability)

**When to continue:**

- Metrics still improving slowly
- You have more compute budget
- You plan to scale to a larger dataset

After pretraining, you can always fine-tune on downstream tasks to improve task-specific performance.

## Using the Python API Directly

For programmatic control or integration into workflows:

```python
from scmodelforge.training.pipeline import TrainingPipeline
from scmodelforge.config.schema import load_config

# Load configuration from YAML
cfg = load_config("pretrain_lung_model.yaml")

# Create and run pipeline
pipeline = TrainingPipeline(cfg)
trainer = pipeline.run()

# trainer is a Lightning Trainer instance
# Access the trained model:
model = trainer.model

# Save to checkpoint manually:
trainer.save_checkpoint("my_final_model.ckpt")
```

You can also construct configs programmatically without YAML:

```python
from scmodelforge.config.schema import (
    ScModelForgeConfig,
    DataConfig,
    TokenizerConfig,
    ModelConfig,
    TrainingConfig,
)

cfg = ScModelForgeConfig(
    data=DataConfig(
        source="local",
        paths=["./data/my_data.h5ad"],
        gene_vocab="human_protein_coding",
        max_genes=2048,
    ),
    tokenizer=TokenizerConfig(
        strategy="rank_value",
        max_genes=2048,
    ),
    model=ModelConfig(
        architecture="transformer_encoder",
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
    ),
    training=TrainingConfig(
        batch_size=64,
        max_epochs=10,
    ),
)

pipeline = TrainingPipeline(cfg)
pipeline.run()
```

This is useful for:

- Hyperparameter sweeps
- Automated experiments
- Integration with workflow engines (Nextflow, Snakemake)

## What's Next

After pretraining, your model is ready for downstream applications:

**Fine-tuning**: Adapt your pretrained model to specific tasks like cell type annotation or perturbation prediction. See the [Fine-tuning Tutorial](finetuning_cell_type.md).

**Evaluation**: Run comprehensive benchmarks on held-out datasets. Use the `scmodelforge benchmark` CLI or the `EvalHarness` API. See the [Evaluation Guide](evaluation.md).

**Sharing**: Push your model to the HuggingFace Hub for the community to use:

```bash
# Export to HuggingFace format
scmodelforge export --checkpoint ./checkpoints/lung_pretrain/best.ckpt --output ./hf_model

# Push to Hub
scmodelforge push --model-dir ./hf_model --repo-id your-username/lung-foundation-model
```

**Advanced pretraining**:

- Multi-GPU training: Set `training.num_gpus: 4` and `strategy: ddp` or `fsdp`
- Multi-species models: Enable `data.multi_species.enabled: true`
- Cloud data: Use `s3://`, `gs://`, or `az://` paths with Census or local data
- Sharded datasets: For 10M+ cells, use `scmodelforge shard` then train on sharded data

See the [Distributed Training](distributed_training.md) and [Large-scale Data Handling](large_scale_data.md) tutorials for details.
