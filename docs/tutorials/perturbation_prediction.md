# Perturbation Response Prediction

This tutorial shows you how to use foundation models to predict how cells respond to genetic perturbations, such as CRISPR knockouts or gene activation.

## What is Perturbation Prediction?

Perturb-seq experiments measure genome-wide transcriptional responses to genetic perturbations. A typical experiment might knock out 100 different genes and sequence thousands of cells per perturbation. The goal is to understand how each perturbation changes gene expression compared to control cells.

Foundation models can learn to predict these expression changes from cell embeddings. This is valuable because:

1. **Reduced experimental cost**: If the model can predict responses accurately, you can prioritize which perturbations to test experimentally.

2. **In silico screening**: Test hundreds or thousands of perturbations computationally before running expensive experiments.

3. **Understanding mechanisms**: The model learns which perturbations have similar effects, revealing functional relationships between genes.

scModelForge supports perturbation-aware data loading, specialized fine-tuning for regression tasks, and rigorous evaluation through the perturbation benchmark.

## Prerequisites

Before starting, you need:

1. **A Perturb-seq dataset**: An `.h5ad` file with perturbation labels in `adata.obs` and control (unperturbed) cells. The dataset should have multiple cells per perturbation for robust training.

2. **Perturbation metadata**: At minimum, a column indicating which perturbation was applied to each cell, and a label identifying control cells (e.g., "control", "non-targeting").

3. **Optional dose information**: If your experiment includes dose-response data (e.g., different drug concentrations), you can include a dose column.

4. **Install perturbation support**:
   ```bash
   pip install "scModelForge[perturbation]"
   ```
   This installs the optional `pertpy` dependency for perturbation data utilities.

## Understanding Perturbation Data

A typical Perturb-seq dataset has this structure:

```python
import scanpy as sc
adata = sc.read_h5ad('./data/perturb_seq.h5ad')

# Perturbation column: which gene was knocked out or activated
print(adata.obs['perturbation'].value_counts())
# control: 5234, BRCA1_KO: 412, TP53_KO: 389, MYC_activation: 401, ...
```

Key requirements:

- **Perturbation labels**: Each cell has a label indicating the perturbation applied. Control cells use a consistent label like "control", "non-targeting", or "scrambled".
- **Control cells**: Essential for computing expression deltas. The model learns to predict `expression(perturbed) - expression(control)`.
- **Multiple cells per perturbation**: Aim for at least 20-50 cells per perturbation. More is better.

## Auto-detecting Perturbation Columns

scModelForge can automatically detect common perturbation column names:

```python
from scmodelforge.data.perturbation import detect_perturbation_columns

columns = detect_perturbation_columns(adata)
print(columns)
# Output: {'perturbation': 'gene_target', 'dose': 'concentration'}
```

Common patterns it recognizes:

- Perturbation: `perturbation`, `condition`, `guide_identity`, `gene_target`, `treatment`, `compound`, `drug`, `sgRNA`
- Dose: `dose`, `concentration`, `dose_value`, `dose_amount`

If your columns use standard names, you can omit `perturbation_key` and `dose_key` from the config and let scModelForge auto-detect them.

## Parsing Perturbation Metadata

The `parse_perturbation_metadata` function extracts structured perturbation information for each cell:

```python
from scmodelforge.data.perturbation import parse_perturbation_metadata

metadata = parse_perturbation_metadata(
    adata,
    perturbation_key='perturbation',
    control_label='control',
    dose_key='dose'
)

# Each cell gets a PerturbationMetadata object
print(metadata[0])
# PerturbationMetadata(
#     perturbation_type='crispr',
#     perturbation_name='BRCA1_KO',
#     dose=1.0,
#     dose_unit=None,
#     is_control=False
# )
```

The `perturbation_type` is inferred from the column name:
- `crispr`: if the column name contains "guide", "sgRNA", "gene_target", or "crispr"
- `chemical`: if it contains "drug", "compound", "treatment", or "dose"
- `unknown`: otherwise

## Loading Perturbation Data in Config

Configure perturbation data loading in your YAML config:

```yaml
data:
  source: local
  paths:
    - ./data/perturb_seq.h5ad
  gene_vocab: human_protein_coding
  preprocessing:
    normalize: library_size
    target_sum: 10000
    log1p: true
  max_genes: 2048
  num_workers: 4
  perturbation:
    enabled: true                  # Enable perturbation-aware data handling
    perturbation_key: perturbation # Column name in adata.obs
    control_label: control         # Label for control cells
    dose_key: dose                 # Optional: column with dose values
    # dose_unit: "uM"              # Optional: unit label for doses
```

When `perturbation.enabled: true`, scModelForge uses `PerturbationDataset` instead of the standard `CellDataset`. This adds perturbation metadata to each batch during training.

If `perturbation_key` is not specified or set to `null`, auto-detection is used.

## Fine-tuning for Perturbation Prediction

Perturbation prediction is a regression task: the model learns to predict expression changes (delta values) for the most differentially expressed genes. This differs from cell type classification, which is a discrete classification task.

### Using the Recipe

scModelForge includes a ready-to-use recipe for perturbation prediction. Start by copying and customizing it:

```bash
cp configs/recipes/perturbation_prediction.yaml my_perturbation_config.yaml
```

Edit the key sections:

```yaml
data:
  paths:
    - ./data/my_perturb_seq.h5ad  # Your dataset path
  perturbation:
    enabled: true
    perturbation_key: gene_target # Your perturbation column name
    control_label: non-targeting   # Your control label

finetune:
  label_key: gene_target          # Must match perturbation_key
  head:
    task: regression              # Regression, not classification
    output_dim: 50                # Number of top DEGs to predict
    hidden_dim: 256
    dropout: 0.1
```

Key differences from cell type classification:

- **task: regression** instead of `task: classification`
- **output_dim**: Number of top differentially expressed genes to predict (typically 20-100)
- **label_key**: Set to the same value as `perturbation.perturbation_key`

### Running Fine-tuning

Fine-tune using a pretrained checkpoint:

```bash
scmodelforge finetune \
  --config my_perturbation_config.yaml \
  --checkpoint path/to/pretrained.ckpt
```

The training loop predicts expression deltas for the top DEGs per perturbation. Training takes longer than classification because regression is a harder task.

### Training Tips

- **More perturbations beat more cells per perturbation**. 50 perturbations with 100 cells each beats 10 perturbations with 500 cells each. The model learns from perturbation diversity.
- **Include control cells**. Verify control labels are consistent (no mixed "control", "Control", "non-targeting").
- **Start with frozen backbone**. Use `freeze_backbone: true` and `freeze_backbone_epochs: 3` to stabilize the regression head before full fine-tuning.
- **Adjust output_dim to your data**. Strong effects (many DEGs): `output_dim: 100`. Subtle perturbations: `output_dim: 20-50`.

## Evaluating Perturbation Prediction

The perturbation benchmark evaluates how well the model predicts expression changes for held-out perturbations.

### Benchmark Configuration

Add the benchmark to your config or create a separate evaluation config:

```yaml
eval:
  every_n_epochs: 5
  benchmarks:
    - name: perturbation
      dataset: perturb_data
      params:
        perturbation_key: perturbation
        control_label: control
        n_top_genes: 50             # Number of DEGs to evaluate (matches output_dim)
        test_fraction: 0.2          # 20% of perturbations held out for testing
        seed: 42
```

The benchmark works as follows:

1. **Split by perturbation**: Randomly hold out 20% of perturbations (not 20% of cells). The model is tested on perturbations it has never seen.

2. **Compute expression deltas**: For each perturbation, compute `mean(perturbed_cells) - mean(control_cells)` across all genes.

3. **Train Ridge regression**: Train a Ridge regression model that maps `mean_embedding(perturbed_cells) -> expression_delta`. Training uses the 80% of perturbations not held out.

4. **Predict for held-out perturbations**: Extract embeddings for held-out perturbations and predict their expression deltas.

5. **Evaluate on top DEGs**: For each held-out perturbation, identify the top N most differentially expressed genes (by absolute delta) and evaluate metrics only on those genes.

### Metrics Explained

The benchmark reports three main metrics plus baseline comparisons:

**pearson_mean**: Average Pearson correlation between predicted and actual expression deltas across held-out perturbations. Higher is better. Values above 0.6 indicate strong predictive power.

**mse_mean**: Mean squared error of predictions. Lower is better. This measures absolute prediction accuracy.

**direction_accuracy**: Fraction of DEGs where the direction of change (up or down) is correctly predicted. A random model would get 0.5. Values above 0.7-0.8 indicate the model captures biological trends.

**Mean-shift baseline**: For honest comparison, the benchmark includes a simple baseline that predicts the average expression change across all training perturbations. This baseline ignores perturbation identity and just predicts the mean response. The model should beat this baseline.

- `baseline_pearson_mean`: Baseline Pearson correlation
- `baseline_mse_mean`: Baseline MSE
- `baseline_direction_accuracy`: Baseline direction accuracy
- `fraction_above_baseline`: Fraction of held-out perturbations where the model's Pearson is better than the baseline's

A good model should have `fraction_above_baseline > 0.6-0.7`, meaning it outperforms the mean-shift baseline on most perturbations.

### Running the Benchmark

Run the benchmark standalone:

```bash
scmodelforge benchmark \
  --config eval_config.yaml \
  --model checkpoints/perturbation/best.ckpt \
  --data ./data/perturb_seq.h5ad \
  --output perturbation_results.json
```

Or include it in your fine-tuning config with `eval.every_n_epochs: 5` to run it periodically during training.

### Interpreting Results

Example output:

```json
{
  "benchmark_name": "perturbation",
  "dataset_name": "perturb_data",
  "metrics": {
    "pearson_mean": 0.68,
    "mse_mean": 1.23,
    "direction_accuracy": 0.74,
    "baseline_pearson_mean": 0.42,
    "baseline_mse_mean": 2.01,
    "baseline_direction_accuracy": 0.58,
    "fraction_above_baseline": 0.72
  },
  "metadata": {
    "n_cells": 8234,
    "n_perturbations": 87,
    "n_train": 69,
    "n_test": 18,
    "n_top_genes": 50
  }
}
```

Interpretation:

- **Pearson 0.68**: Strong correlation. The model captures most of the perturbation effects.
- **Direction accuracy 0.74**: Correctly predicts up/down regulation for 74% of DEGs.
- **Baseline comparisons**: The model substantially outperforms the mean-shift baseline (0.68 vs. 0.42 Pearson).
- **Fraction above baseline 0.72**: The model beats the baseline on 72% of held-out perturbations.

This is a successful model. It generalizes well to unseen perturbations.

### Why Perturbation-level Splitting?

The benchmark splits by perturbation, not by cell. This is critical for realistic evaluation.

If you split by cell (80% cells for train, 20% for test), the model sees some cells from every perturbation during training. It can memorize perturbation effects and just needs to recognize which perturbation each test cell belongs to. This is too easy and doesn't test generalization.

Splitting by perturbation means the model must predict responses for perturbations it has never seen. This tests whether the model has learned general principles of how perturbations affect gene expression, not just memorized specific perturbations.

Real-world use case: you train on 80 perturbations, then predict responses for 20 novel perturbations you haven't tested experimentally. This is what the benchmark evaluates.

## Tips for Better Perturbation Prediction

**More perturbations is more important than more cells**. 100 perturbations with 50 cells each beats 10 perturbations with 500 cells each. Aim for at least 50-100 different perturbations.

**Include control cells**. Essential for computing expression deltas. Verify control cells are abundant (at least 10-20% of total cells).

**Start from a pretrained backbone**. Do not train from scratch. Pretrained models have learned general gene expression patterns. Fine-tuning adapts this knowledge to perturbation prediction with far less data and compute.

**Match preprocessing to pretraining**. If the pretrained model used log-normalized counts, use the same for fine-tuning. Check the pretraining config.

**Use LoRA for limited data**. If you have fewer than 30-50 perturbations, LoRA helps prevent overfitting:

```yaml
finetune:
  lora:
    enabled: true
    rank: 8
    alpha: 16
    dropout: 0.05
```

## Common Issues and Solutions

**Low Pearson correlation (<0.4)**

The model is not capturing perturbation effects. Possible causes:

- Too few perturbations (need 30+ for robust learning)
- Weak perturbation effects (if most perturbations barely change expression, prediction is hard)
- Preprocessing mismatch between pretraining and fine-tuning
- Learning rate too low (try `backbone_lr: 1e-4` and `head_lr: 1e-3`)

**High MSE but reasonable Pearson**

The model captures relative changes (which genes go up/down) but not absolute magnitudes. This is common and often acceptable. If you need accurate magnitudes, increase `output_dim` to predict more genes and use a longer training schedule.

**Fraction above baseline is low (<0.5)**

The model is no better than the mean-shift baseline. Possible causes:

- Not enough perturbation diversity in training data
- Model is underfitting (try unfreezing the backbone earlier or using a higher learning rate)
- Perturbations have very similar effects (the mean-shift baseline is actually quite strong)

**Direction accuracy stuck at 0.5**

The model is guessing randomly. Check that:

- Control cells are correctly labeled
- Perturbation labels are consistent (no typos or mixed case)
- The dataset has actual perturbation effects (compute expression deltas manually to verify)

## What's Next

You've learned how to fine-tune a foundation model for perturbation prediction and evaluate it rigorously using the perturbation benchmark. Next steps:

- **Evaluation tutorial**: Deep dive into all evaluation benchmarks and how to interpret them.
- **Custom benchmarks**: Write your own benchmark for specialized evaluation tasks.
- **Multi-species perturbation**: Combine human and mouse perturbation data using ortholog mapping.
- **GRN inference**: Use the GRN benchmark to evaluate gene regulatory network predictions alongside perturbation effects.

The same pipeline works for chemical perturbations (drug screens), environmental perturbations (hypoxia, heat shock), and other experimental conditions. Just ensure your metadata follows the perturbation data structure, and scModelForge handles the rest.
