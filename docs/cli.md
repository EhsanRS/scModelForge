# CLI Reference

scModelForge provides a command-line interface for training, fine-tuning, and evaluating single-cell foundation models. All commands are config-driven and support resuming from checkpoints.

## Installation

The CLI is installed automatically with the package:

```bash
pip install scModelForge
scmodelforge --version
```

## Commands

### `scmodelforge train`

Train a model from scratch using a YAML configuration file.

**Usage:**

```bash
scmodelforge train --config CONFIG [--resume CHECKPOINT]
```

**Arguments:**

- `--config` (required): Path to YAML configuration file defining data, tokenizer, model, and training parameters.
- `--resume` (optional): Path to checkpoint file to resume training from. Overrides `training.resume_from` in config.

**Example:**

```bash
# Start fresh training
scmodelforge train --config configs/examples/geneformer_basic.yaml

# Resume from checkpoint
scmodelforge train --config configs/examples/geneformer_basic.yaml --resume checkpoints/last.ckpt
```

**What it does:**

1. Loads and validates the config file
2. Sets random seeds for reproducibility
3. Loads and preprocesses training data (local or CELLxGENE Census)
4. Constructs the tokenizer and model from config
5. Initializes PyTorch Lightning trainer with specified strategy, precision, and devices
6. Runs training loop with validation splits, checkpointing, and logging
7. Optionally runs evaluation benchmarks every N epochs

**Output:**

- Checkpoints saved to `training.checkpoint_dir` (default: `./checkpoints/`)
- Logs written to `training.log_dir` (default: `./logs/`)
- Metrics logged to Weights & Biases (if `training.logger = "wandb"`)

---

### `scmodelforge finetune`

Fine-tune a pretrained model on a downstream task (classification or regression).

**Usage:**

```bash
scmodelforge finetune --config CONFIG --checkpoint CHECKPOINT
```

**Arguments:**

- `--config` (required): Path to YAML configuration file. Must include a `finetune` section with task head and training parameters.
- `--checkpoint` (required): Path to pretrained model checkpoint. Can be a Lightning checkpoint (`.ckpt`) or raw state dict (`.pt`, `.pth`).

**Example:**

```bash
scmodelforge finetune \
  --config configs/examples/finetune_celltype.yaml \
  --checkpoint checkpoints/pretrained_best.ckpt
```

**What it does:**

1. Loads the config and validates the `finetune` section
2. Loads the pretrained backbone from checkpoint
3. Optionally applies LoRA adapters (`finetune.lora.enabled = true`)
4. Constructs a task-specific head (classification or regression)
5. Loads labeled data from `data.paths` with labels from `adata.obs[finetune.label_key]`
6. Trains with discriminative learning rates (optional) and gradual unfreezing (optional)
7. Evaluates on validation split and saves the best checkpoint

**Output:**

- Fine-tuned checkpoints saved to `training.checkpoint_dir`
- Metrics logged to configured logger (wandb, tensorboard, or csv)

**Fine-tuning options:**

- `finetune.freeze_backbone`: Freeze all backbone parameters (default: `false`)
- `finetune.freeze_backbone_epochs`: Unfreeze after N epochs (default: `0`)
- `finetune.backbone_lr`: Discriminative LR for backbone (default: same as global LR)
- `finetune.head_lr`: Discriminative LR for task head (default: same as global LR)
- `finetune.lora.enabled`: Use LoRA adapters for parameter-efficient fine-tuning (default: `false`)

---

### `scmodelforge benchmark`

Run evaluation benchmarks on a trained model.

**Usage:**

```bash
scmodelforge benchmark --config CONFIG --model MODEL --data DATA [--output OUTPUT]
```

**Arguments:**

- `--config` (required): Path to YAML configuration file. The `eval.benchmarks` section specifies which benchmarks to run.
- `--model` (required): Path to model checkpoint file (`.ckpt`, `.pt`, or `.pth`).
- `--data` (required): Path to evaluation dataset in AnnData format (`.h5ad`).
- `--output` (optional): Path to save results as JSON. If not provided, results are printed to stdout only.

**Example:**

```bash
scmodelforge benchmark \
  --config configs/examples/geneformer_basic.yaml \
  --model checkpoints/best.ckpt \
  --data data/test_dataset.h5ad \
  --output results/eval_results.json
```

**What it does:**

1. Loads the config, model, and evaluation data
2. Constructs the tokenizer from config
3. Loads model weights from checkpoint (strips Lightning module prefix if present)
4. Runs each benchmark specified in `eval.benchmarks`
5. Prints results to stdout (benchmark name, dataset, metrics)
6. Optionally saves results as JSON to `--output`

**Available benchmarks:**

Configure benchmarks in the `eval` section of your YAML config:

```yaml
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

**Benchmark types:**

- `linear_probe`: Train a logistic regression classifier on frozen embeddings to predict cell type. Reports accuracy and macro F1 score.
- `embedding_quality`: Compute scIB metrics for biological conservation (NMI, ARI, cell-type ASW) and batch correction (batch ASW). Reports overall score (0.6 × bio + 0.4 × batch).

**Output format:**

Results are printed as:

```
Benchmark: linear_probe | Dataset: test
  accuracy: 0.92
  f1_macro: 0.89

Benchmark: embedding_quality | Dataset: test
  overall: 0.85
  bio_conservation: 0.88
  batch_correction: 0.80
```

JSON output format:

```json
[
  {
    "benchmark_name": "linear_probe",
    "dataset_name": "test",
    "metrics": {
      "accuracy": 0.92,
      "f1_macro": 0.89
    }
  },
  {
    "benchmark_name": "embedding_quality",
    "dataset_name": "test",
    "metrics": {
      "overall": 0.85,
      "bio_conservation": 0.88,
      "batch_correction": 0.80
    }
  }
]
```

---

## Configuration Files

All CLI commands require a YAML configuration file. See [Configuration Reference](api/config.md) for the full schema and [configs/examples/geneformer_basic.yaml](https://github.com/EhsanRS/scModelForge/blob/main/configs/examples/geneformer_basic.yaml) for a complete example.

**Minimal training config:**

```yaml
data:
  source: local
  paths:
    - ./data/my_dataset.h5ad
  gene_vocab: human_protein_coding

tokenizer:
  strategy: rank_value
  max_genes: 2048

model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 12
  num_heads: 8

training:
  batch_size: 64
  max_epochs: 10
  optimizer:
    name: adamw
    lr: 1.0e-4
  scheduler:
    name: cosine_warmup
    warmup_steps: 2000
```

**Minimal fine-tuning config:**

```yaml
data:
  source: local
  paths:
    - ./data/labeled_dataset.h5ad

tokenizer:
  strategy: rank_value
  max_genes: 2048

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

finetune:
  checkpoint_path: ""  # Overridden by --checkpoint
  label_key: cell_type
  freeze_backbone: false
  freeze_backbone_epochs: 2
  head:
    task: classification
    n_classes: null  # Inferred from data
  backbone_lr: 1.0e-5
  head_lr: 1.0e-4
```

## Exit Codes

- `0`: Success
- `1`: Configuration error or runtime error

## Environment Variables

None required. Optional environment variables for underlying frameworks (PyTorch, CUDA, etc.) are respected.
