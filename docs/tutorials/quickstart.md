# Quick Start

Welcome to scModelForge. In this 10-minute tutorial, you will pretrain a small transformer model on single-cell RNA-seq data, then use it to extract cell embeddings. By the end, you will have a working foundation model and understand the basic workflow for training and using models in scModelForge.

## What is scModelForge?

scModelForge is a toolkit for pretraining transformer-based foundation models on single-cell RNA-seq data. Instead of manually engineering features or using traditional dimensionality reduction (PCA, UMAP), you train a neural network to learn rich, task-agnostic representations of cells. These representations can then be fine-tuned for downstream tasks like cell type annotation, perturbation response prediction, or trajectory inference.

## Prerequisites

Before you begin, you should have:

- Python 3.10 or later
- A GPU (recommended but not required for this small demo)
- Basic familiarity with scRNA-seq data and AnnData objects
- Comfort with the command line and editing YAML files

You do NOT need experience with PyTorch, transformer models, or pretraining objectives. This tutorial will guide you through the process step by step.

## Installation

Install scModelForge using pip:

```bash
pip install scmodelforge
```

For this tutorial, we also recommend installing optional dependencies for data preprocessing and visualization:

```bash
pip install scmodelforge[scanpy]
```

If you plan to use HuggingFace Hub models or push your own models later, install the hub extras:

```bash
pip install scmodelforge[hub]
```

For development or to run tests, use:

```bash
pip install scmodelforge[dev]
```

## Prepare Your Data

We will use the classic PBMC 3k dataset: approximately 2,700 peripheral blood mononuclear cells profiled with 10x Genomics. This dataset is small enough to train quickly on a laptop but large enough to demonstrate the workflow.

Create a data directory and download the preprocessed dataset using scanpy:

```python
import scanpy as sc

# Create data directory
import os
os.makedirs('./data', exist_ok=True)

# Download PBMC 3k dataset (preprocessed)
adata = sc.datasets.pbmc3k_processed()

# scModelForge expects raw counts, but this demo dataset is already normalized.
# For a real workflow, start with raw counts and configure preprocessing in YAML.
# Save to H5AD format
adata.write('./data/pbmc3k.h5ad')

print(f"Saved dataset with {adata.n_obs} cells and {adata.n_vars} genes")
```

Expected output:

```
Saved dataset with 2638 cells and 1838 genes
```

This H5AD file is now ready for scModelForge.

## Create a Configuration File

scModelForge uses YAML configuration files to specify all training parameters. This makes experiments reproducible and easy to share.

Create a file named `quickstart.yaml` with the following content:

```yaml
data:
  source: local
  paths:
    - ./data/pbmc3k.h5ad
  gene_vocab: human_protein_coding
  preprocessing:
    normalize: library_size
    target_sum: 10000
    log1p: true

tokenizer:
  strategy: rank_value
  max_genes: 512
  prepend_cls: true
  masking:
    mask_ratio: 0.15

model:
  architecture: transformer_encoder
  hidden_dim: 256
  num_layers: 4
  num_heads: 4
  dropout: 0.1
  max_seq_len: 512
  pooling: cls
  pretraining_task: masked_gene_prediction

training:
  batch_size: 32
  max_epochs: 5
  seed: 42
  precision: "32-true"
  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 0.01
  scheduler:
    name: cosine_warmup
    warmup_steps: 100
  checkpoint_dir: ./checkpoints
  val_split: 0.1
```

Here is what each section does:

- **data**: Points to your H5AD file, uses the built-in human protein-coding gene vocabulary, and applies library-size normalization with log1p transformation
- **tokenizer**: Converts cells to token sequences using rank-value encoding (genes ranked by expression), keeps the top 512 genes per cell, and randomly masks 15% of genes for pretraining
- **model**: Defines a small transformer encoder with 4 layers, 256-dimensional embeddings, and 4 attention heads. Uses a CLS token for cell-level pooling
- **training**: Runs for 5 epochs with batch size 32, AdamW optimizer with learning rate 1e-4, and cosine warmup learning rate schedule. Reserves 10% of data for validation

This configuration is intentionally small so training completes quickly on a CPU or single GPU.

## Run Training

With your configuration file ready, start training with a single command:

```bash
scmodelforge train --config quickstart.yaml
```

You will see output like this:

```
Global seed set to 42
Using device: cuda:0
Loading data from ./data/pbmc3k.h5ad...
Loaded 2638 cells, 1838 genes
Building gene vocabulary: human_protein_coding
Vocabulary size: 19430 genes
Initializing rank_value tokenizer...
Initializing model: transformer_encoder
Model parameters: 8.3M
Training dataloader: 2374 cells, 75 batches
Validation dataloader: 264 cells, 9 batches

Epoch 1/5: 100%|███████████████████| 75/75 [00:23<00:00,  3.15batch/s, loss=5.234]
Validation: loss=4.987, accuracy=0.123
Epoch 2/5: 100%|███████████████████| 75/75 [00:22<00:00,  3.38batch/s, loss=4.512]
Validation: loss=4.456, accuracy=0.189
Epoch 3/5: 100%|███████████████████| 75/75 [00:22<00:00,  3.41batch/s, loss=4.123]
Validation: loss=4.098, accuracy=0.234
Epoch 4/5: 100%|███████████████████| 75/75 [00:21<00:00,  3.45batch/s, loss=3.867]
Validation: loss=3.845, accuracy=0.267
Epoch 5/5: 100%|███████████████████| 75/75 [00:21<00:00,  3.48batch/s, loss=3.678]
Validation: loss=3.701, accuracy=0.289

Training complete. Best checkpoint saved to ./checkpoints/epoch=4-step=374.ckpt
```

Key things to notice:

- The loss decreases over epochs, indicating the model is learning
- Accuracy gradually improves as the model gets better at predicting masked genes
- A checkpoint is saved after each epoch in the `./checkpoints/` directory

Training on a GPU takes about 2-3 minutes. On a CPU, expect 10-15 minutes.

## Inspect Results

After training completes, check the checkpoint directory:

```bash
ls -lh ./checkpoints/
```

You should see checkpoint files like:

```
epoch=0-step=74.ckpt
epoch=1-step=149.ckpt
epoch=2-step=224.ckpt
epoch=3-step=299.ckpt
epoch=4-step=374.ckpt
last.ckpt
```

Each checkpoint contains the full model weights and optimizer state, allowing you to resume training or use the model for inference.

If you configured Weights & Biases logging in your config, you can also view training curves and metrics in the W&B dashboard.

## Extract Cell Embeddings

Now that you have a trained model, you can use it to extract learned representations of cells. These embeddings can be used for clustering, visualization, or as input to downstream classifiers.

Create a Python script named `extract_embeddings.py`:

```python
from __future__ import annotations

import scanpy as sc
from scmodelforge.data import GeneVocab
from scmodelforge.tokenizers import get_tokenizer
from scmodelforge.eval._utils import extract_embeddings
import torch

# Load the dataset (needed for both vocabulary and embedding extraction)
adata = sc.read_h5ad("./data/pbmc3k.h5ad")

# Build gene vocabulary from the dataset
gene_vocab = GeneVocab.from_adata(adata)

# Initialize tokenizer
tokenizer = get_tokenizer(
    "rank_value",
    gene_vocab=gene_vocab,
    max_len=512,
    prepend_cls=True,
)

# Load the trained model from checkpoint
from scmodelforge.models.hub import load_pretrained
model = load_pretrained("./checkpoints/epoch=4-step=374.ckpt")

# Extract embeddings for all cells
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = extract_embeddings(
    model=model,
    adata=adata,
    tokenizer=tokenizer,
    batch_size=64,
    device=device,
)

print(f"Extracted embeddings with shape: {embeddings.shape}")
# Expected: (2638, 256) - one 256-dim vector per cell

# Add embeddings to AnnData object for downstream analysis
adata.obsm['X_scmodelforge'] = embeddings

# Now you can use scanpy for visualization
sc.pp.neighbors(adata, use_rep='X_scmodelforge')
sc.tl.umap(adata)
sc.pl.umap(adata, color='louvain', title='scModelForge embeddings')

# Save annotated object
adata.write('./data/pbmc3k_with_embeddings.h5ad')
print("Saved embeddings to AnnData object")
```

Run the script:

```bash
python extract_embeddings.py
```

Expected output:

```
Extracted embeddings with shape: (2638, 256)
Saved embeddings to AnnData object
```

You now have cell embeddings that capture learned representations from your foundation model. These can be used as input for any downstream analysis, replacing or complementing traditional PCA embeddings.

## What Makes a Good Embedding?

Unlike supervised learning where you have a clear accuracy metric, pretraining is unsupervised. How do you know if your model learned useful representations?

Good embeddings should:

- Cluster cells of the same type together
- Separate cells of different types
- Preserve biological variation while removing batch effects
- Transfer well to downstream tasks with minimal fine-tuning

scModelForge includes benchmarking tools to assess embedding quality using standardized metrics. See the Model Assessment and Benchmarking tutorial for details.

## What You Just Built

Congratulations! You just:

1. Prepared a single-cell dataset in H5AD format
2. Configured a transformer encoder with masked gene prediction pretraining
3. Trained a foundation model from scratch in under 15 minutes
4. Extracted learned cell embeddings for downstream analysis

This is the core workflow for scModelForge. Everything else builds on these fundamentals.

## Next Steps

Now that you have completed the quickstart, here are logical next steps:

**Understand the data pipeline better:**
Read Data Loading and Preprocessing to learn about loading from CELLxGENE Census, handling multiple species, and working with perturbation data.

**Train a production-quality model:**
The Pretraining a Foundation Model tutorial covers larger models, better hyperparameters, distributed training, and assessment during training.

**Fine-tune for a specific task:**
The Fine-tuning for Cell Type Annotation tutorial shows how to take a pretrained model and adapt it for classification or regression tasks.

**Choose the right tokenization strategy:**
The Tokenization Strategies tutorial explains rank-value, binned expression, continuous projection, and gene embedding tokenizers, and when to use each.

**Assess your model rigorously:**
The Model Assessment and Benchmarking tutorial covers linear probes, embedding quality metrics, perturbation benchmarks, and gene regulatory network inference.

**Scale to large datasets:**
If you are working with millions of cells, read Large-scale Data Handling to learn about sharding, memory-mapped storage, distributed sampling, and streaming datasets.

**Share your model:**
Once you have trained a good model, learn how to share it on HuggingFace Hub in the HuggingFace Hub Integration tutorial.

## Common Issues

**Out of memory during training:**
Reduce `batch_size` in your config, or reduce `model.hidden_dim` and `model.num_layers` for a smaller model.

**Training loss not decreasing:**
Try increasing `training.optimizer.lr` to 5e-4, or increasing `training.scheduler.warmup_steps` to 500. Make sure your data is properly normalized.

**Validation loss much higher than training loss:**
Your model may be overfitting. Increase `model.dropout` to 0.2, add data augmentation, or get more training data.

**Embeddings look random:**
5 epochs is very short. Try 20-50 epochs for a real model. Also ensure your preprocessing is correct and your masking ratio is reasonable (0.15 is standard).

**Import errors:**
Make sure you installed scmodelforge correctly: `pip install scmodelforge`. If using optional features, install the relevant extras like `[scanpy]` or `[hub]`.

## Getting Help

If you run into issues not covered here:

- Check the API Reference for detailed documentation of all modules
- Read the FAQ for answers to common questions
- Open an issue on the GitHub repository
- Consult the other tutorials for more advanced topics

Happy training!
