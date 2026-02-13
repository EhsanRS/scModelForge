# Fine-tuning for Cell Type Annotation

This tutorial shows you how to adapt a pretrained scModelForge model to classify cell types in your own dataset.

## What is Fine-tuning?

When you train a foundation model from scratch on millions of single-cell profiles, it learns the fundamental "grammar" of gene expression: which genes co-express, which patterns indicate specific biological states, and how cells organize in high-dimensional space. This is like learning the general rules of a language.

Fine-tuning takes that pretrained model and teaches it a specific task, such as recognizing cell types. Instead of starting from random weights, you start with a model that already understands gene expression patterns. This means you need far less labeled data and far less compute to achieve good performance.

The analogy: a pretrained model is like a biology PhD student who knows cell biology inside-out but hasn't seen your specific tissue yet. Fine-tuning is their one-week crash course on your dataset. You wouldn't hire a random undergraduate and ask them to learn all of biology plus your tissue in one week (training from scratch), and you wouldn't waste years teaching your PhD student introductory biology they already know (ignoring the pretrained weights).

## Prerequisites

Before starting, you need:

1. **A pretrained checkpoint**: Either train your own using the pretraining tutorial, download a publicly available checkpoint, or load one from the HuggingFace Hub.

2. **A labeled dataset**: An `.h5ad` file with cell type labels in `adata.obs`. The label column can have any name (e.g., `cell_type`, `louvain`, `celltype_major`), you'll tell scModelForge which column to use in the config.

3. **Enough labeled cells**: For most cell type classification tasks, 1000-5000 labeled cells is sufficient. You need at least a few dozen examples per class for robust learning.

## Preparing Your Labeled Data

We'll use a publicly available PBMC dataset with known cell types as an example. In practice, you'll replace this with your own annotated dataset.

```python
import scanpy as sc

# Download a labeled dataset
adata = sc.datasets.pbmc3k_processed()

# Inspect the available labels
print(adata.obs.columns)  # See what metadata columns exist
print(adata.obs['louvain'].value_counts())  # The 'louvain' column has cell type labels

# Save to disk for scModelForge
adata.write_h5ad('./data/pbmc3k.h5ad')
```

The key requirement: your `adata.obs` must have a column containing categorical cell type labels. Each cell gets exactly one label. The column name doesn't matter, as long as you specify it correctly in the config.

If your dataset needs preprocessing (normalization, log-transformation, highly variable gene selection), you can either do it in scanpy before saving, or use the `scmodelforge preprocess` CLI (see preprocessing tutorial).

## Fine-tuning Strategies: Which Approach to Use?

There are four main ways to fine-tune a model, each with different tradeoffs:

| Strategy | What Updates | Speed | Accuracy | Best For |
|----------|-------------|-------|----------|----------|
| **Full fine-tuning** | All parameters | Slow | Highest | Large datasets (10k+ cells), high-end GPUs |
| **Frozen backbone + head** | Only classification head | Very fast | Good | Quick validation, small datasets |
| **Gradual unfreezing** | Head first, then backbone | Medium | Very good | Balanced approach, most use cases |
| **LoRA (adapters)** | Small adapter layers | Fast | Very good | Limited compute, easy deployment |

### Strategy Details

**Full fine-tuning** updates every parameter in the model. This gives maximum flexibility and usually the best accuracy, but requires more compute, more training time, and more data to avoid overfitting. Use this when you have abundant labeled data and GPU resources.

**Frozen backbone + head** keeps the pretrained encoder frozen and only trains a small classification head on top. This is the fastest approach and works surprisingly well when the pretrained model already captures relevant features. Use this as a first baseline, or when you have very limited labeled data.

**Gradual unfreezing** (recommended for most users) starts with a frozen backbone for a few epochs, allowing the classification head to learn good initial weights, then unfreezes the backbone for full fine-tuning. This combines the stability of frozen training with the flexibility of full fine-tuning. It's the default strategy in many transfer learning libraries.

**LoRA (Low-Rank Adaptation)** adds small trainable adapter matrices to the frozen model. Instead of updating all 50 million parameters in a large model, you train only 500k additional parameters. This is parameter-efficient fine-tuning (PEFT). LoRA models train faster, use less memory, and are easier to share (you only need to save the tiny adapter weights, not the whole model). Accuracy is typically within 1-2% of full fine-tuning. Use this when you have limited GPU memory or want to fine-tune many models in parallel.

## Configuration for Full Fine-tuning

Create a YAML configuration file. Here's a complete example for full fine-tuning with gradual unfreezing:

```yaml
# finetune_config.yaml

data:
  source: local
  paths:
    - ./data/pbmc3k.h5ad
  gene_vocab: human_protein_coding
  preprocessing:
    normalize: library_size
    target_sum: 10000
    log1p: true
  max_genes: 2048
  num_workers: 4

tokenizer:
  strategy: rank_value          # Geneformer-style tokenization
  max_genes: 2048
  prepend_cls: true             # CLS token for classification pooling
  masking:
    # Masking config required but NOT applied during fine-tuning
    mask_ratio: 0.15
    random_replace_ratio: 0.1
    keep_ratio: 0.1

model:
  architecture: transformer_encoder
  hidden_dim: 512               # Must match your pretrained checkpoint
  num_layers: 6                 # Must match your pretrained checkpoint
  num_heads: 8                  # Must match your pretrained checkpoint
  dropout: 0.1
  max_seq_len: 2048
  pooling: cls                  # Use [CLS] token for classification
  activation: gelu

training:
  batch_size: 32
  max_epochs: 20
  seed: 42
  precision: bf16-mixed         # Mixed precision for faster training
  optimizer:
    name: adamw
    lr: 5.0e-5                  # Lower learning rate than pretraining
    weight_decay: 0.01
  scheduler:
    name: cosine_warmup
    warmup_steps: 500
  gradient_clip: 1.0
  checkpoint_dir: ./checkpoints/cell_type
  val_split: 0.1                # 10% held out for validation

finetune:
  label_key: louvain            # Column in adata.obs with cell type labels
  freeze_backbone: true         # Start frozen
  freeze_backbone_epochs: 5     # Unfreeze after 5 epochs (gradual unfreezing)
  head:
    task: classification
    n_classes: null             # Auto-detected from data
    hidden_dim: 256             # Size of hidden layer in classification head
    dropout: 0.1
  backbone_lr: 1.0e-5           # Low LR for pretrained backbone
  head_lr: 1.0e-3               # High LR for randomly initialized head
```

Key points:

- **Model architecture must match the checkpoint**: If your pretrained model has `hidden_dim: 512, num_layers: 6`, use those exact values here. Mismatches will cause errors when loading the checkpoint.

- **Learning rates are lower than pretraining**: Pretrained weights are already good. A typical pretraining LR is `1e-4` to `5e-4`, while fine-tuning uses `1e-5` to `5e-5`.

- **Discriminative learning rates**: The `backbone_lr` is much lower than `head_lr` because the backbone is already well-trained. The randomly initialized classification head needs a higher learning rate to learn quickly.

- **n_classes is auto-detected**: scModelForge reads your label column and counts unique values. You can set it explicitly if you want, but `null` is easier.

- **No masking during fine-tuning**: The masking config is required for tokenizer initialization but is ignored during fine-tuning. All genes are visible for classification.

## Running Fine-tuning

Once you have a config file and a pretrained checkpoint, run:

```bash
scmodelforge finetune \
  --config finetune_config.yaml \
  --checkpoint checkpoints/pretrain/best.ckpt
```

The checkpoint path should point to a Lightning checkpoint (`.ckpt` file) from pretraining, or a directory in HuggingFace format (containing `model.safetensors` and `config.json`).

During training, you'll see:

```
Epoch 0:  100%|██████████| 250/250 [00:45<00:00,  5.50it/s, loss=2.134, val_loss=1.987, val_acc=0.456]
Epoch 1:  100%|██████████| 250/250 [00:44<00:00,  5.62it/s, loss=1.876, val_loss=1.723, val_acc=0.612]
Epoch 5:  100%|██████████| 250/250 [00:46<00:00,  5.43it/s, loss=1.245, val_loss=1.198, val_acc=0.784]
Unfreezing backbone at epoch 5
Epoch 6:  100%|██████████| 250/250 [01:12<00:00,  3.47it/s, loss=0.987, val_loss=0.912, val_acc=0.823]
```

Notice the "Unfreezing backbone" message at epoch 5, and the training slowing down slightly afterward (because now you're updating the full model). Validation accuracy steadily improves.

The best checkpoint (by validation loss) is saved to `./checkpoints/cell_type/best.ckpt`.

## Configuration for LoRA Fine-tuning

LoRA is a parameter-efficient alternative to full fine-tuning. Instead of updating all model parameters, LoRA adds small trainable adapter layers while keeping the pretrained weights frozen.

To enable LoRA, add a `lora` section to your `finetune` config:

```yaml
finetune:
  label_key: louvain
  freeze_backbone: true         # With LoRA, backbone stays frozen
  freeze_backbone_epochs: 0     # No gradual unfreezing needed
  head:
    task: classification
    hidden_dim: 256
    dropout: 0.1
  lora:
    enabled: true
    rank: 8                     # LoRA rank (higher = more capacity, more params)
    alpha: 16                   # LoRA scaling factor (typically 2x rank)
    dropout: 0.05               # Dropout in LoRA layers
    # target_modules defaults to ["out_proj", "linear1", "linear2"]
  backbone_lr: 1.0e-4           # Can use higher LR with LoRA
  head_lr: 1.0e-3
```

Understanding the LoRA parameters:

- **rank**: Controls adapter capacity. Typical values: 4 (very small), 8 (balanced), 16 (large). Higher rank means more trainable parameters but better adaptation. Start with 8.

- **alpha**: Scaling factor for the adapter updates. Rule of thumb: set `alpha = 2 * rank`. This controls how much the adapters influence the model.

- **dropout**: Regularization in the adapter layers. 0.05 is a safe default.

- **target_modules**: Which transformer layers get adapters. The default `["out_proj", "linear1", "linear2"]` targets the attention output projection and the two feedforward layers in each transformer block. This covers the most important parameters.

With LoRA enabled:
- The pretrained backbone stays frozen (LoRA layers are added on top).
- `freeze_backbone_epochs` is ignored (backbone never unfreezes).
- Only the classification head and LoRA adapters are trained.
- Training is faster and uses less memory than full fine-tuning.
- Final accuracy is typically within 1-2% of full fine-tuning.

Run the same command as before. The CLI detects LoRA in the config and handles everything automatically:

```bash
scmodelforge finetune \
  --config finetune_lora_config.yaml \
  --checkpoint checkpoints/pretrain/best.ckpt
```

## Evaluating the Fine-tuned Model

After fine-tuning, you want to evaluate performance. There are two approaches:

### Option 1: Validation metrics during training

scModelForge automatically logs validation accuracy and loss to your logger (TensorBoard or WandB). Check these metrics to see how well the model is learning.

### Option 2: Explicit benchmark evaluation

Use the `linear_probe` benchmark to evaluate on a held-out test set:

```yaml
# eval_config.yaml

data:
  source: local
  paths:
    - ./data/test_set.h5ad      # A separate test set, not seen during training

tokenizer:
  strategy: rank_value
  max_genes: 2048
  prepend_cls: true

model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  max_seq_len: 2048
  pooling: cls

benchmarks:
  - name: linear_probe
    label_key: louvain
    train_split: 0.0            # Use all data for testing (no training)
```

Then run:

```bash
scmodelforge benchmark \
  --config eval_config.yaml \
  --model checkpoints/cell_type/best.ckpt \
  --output results.json
```

This extracts embeddings from your fine-tuned model and evaluates classification accuracy. Since the model is already fine-tuned for classification, you should see high accuracy.

Alternatively, if you want to test how well the fine-tuned embeddings transfer to a related task, set `train_split: 0.8` to train a fresh linear classifier on the embeddings.

## Tips for Better Fine-tuning Results

### Start simple, then add complexity

1. First, try frozen backbone + head (set `freeze_backbone_epochs: 999999` to never unfreeze). This is fast and gives you a baseline.
2. If accuracy is already good enough, you're done. If not, move to gradual unfreezing or LoRA.
3. Full fine-tuning is the last resort for when you have abundant data and compute.

### Use discriminative learning rates

Always set `backbone_lr` much lower than `head_lr`. The pretrained backbone has useful weights; you want to nudge them gently, not destroy them. A good rule: `backbone_lr = head_lr / 10` or even `head_lr / 100`.

### More data beats more epochs

If you can label 5000 cells instead of 1000, do that before increasing training epochs from 10 to 50. Overfitting is the main risk in fine-tuning. More diverse labeled data is the best defense.

### Match preprocessing to pretraining

If your pretrained model was trained on log-normalized counts with highly variable gene selection, fine-tune on the same preprocessing. Mismatches hurt performance. Check the pretraining config or model card for details.

### Use early stopping

Set `training.patience` in the config to stop training if validation loss stops improving. This prevents overfitting and saves compute.

### When to use LoRA

Use LoRA if:
- You have limited GPU memory (it uses 30-50% less memory than full fine-tuning).
- You want to fine-tune multiple models in parallel for different tasks.
- You plan to share the model (LoRA weights are tiny, ~1-10 MB vs. 200+ MB for full models).
- You have a smaller labeled dataset (LoRA's built-in regularization helps prevent overfitting).

Full fine-tuning is better if:
- You have abundant labeled data (10k+ cells).
- The task is very different from pretraining (e.g., the pretrained model saw healthy tissue, you're fine-tuning on disease states).
- You have the compute and don't mind longer training.

## Common Issues and Solutions

**Error: "Checkpoint dimension mismatch"**

Your config's `hidden_dim`, `num_layers`, or `num_heads` doesn't match the checkpoint. Check the pretrained model's config and use the exact same architecture.

**Validation accuracy stuck at random guessing**

Learning rate might be too low, or the backbone is frozen for too long. Try reducing `freeze_backbone_epochs` to 2-3, or increase `head_lr` to `5e-3`.

**Training loss decreases but validation loss increases**

Overfitting. Solutions: reduce `max_epochs`, add more labeled data, increase `dropout`, or use LoRA for built-in regularization.

**Out of memory error**

Reduce `batch_size`, enable `precision: bf16-mixed`, or switch to LoRA fine-tuning (uses less memory).

## What's Next

Congratulations! You've fine-tuned a foundation model for cell type annotation. Next steps:

- **Evaluation tutorial**: Deep dive into benchmarking and evaluation metrics.
- **Perturbation prediction**: Fine-tune for predicting gene perturbation effects.
- **Sharing models**: Push your fine-tuned model to HuggingFace Hub for the community.
- **Multi-task fine-tuning**: Fine-tune for multiple tasks simultaneously (cell type + batch + disease state).

The same fine-tuning pipeline works for any classification or regression task. Just change `label_key` to point at a different `obs` column (e.g., `disease_state`, `timepoint`, `dose`), and scModelForge handles the rest.
