# Model Evaluation & Benchmarking

How to assess whether your foundation model is actually learning useful biology.

## Why evaluate?

A low training loss doesn't guarantee that your model produces useful embeddings. Your model might memorize training data without capturing the underlying biological structure that makes it generalizable to new datasets or tasks.

Benchmarks test whether the model captures biologically meaningful structure by evaluating on held-out data across different tasks. Good embeddings should:

1. Separate different cell types clearly (even ones not seen during pretraining)
2. Preserve biological variation while removing technical batch effects
3. Predict responses to perturbations or genetic programs
4. Generalize to new experimental contexts and datasets

## Available benchmarks

scModelForge provides multiple benchmark types, each testing different aspects of your model:

| Benchmark | What it measures | Key metrics | When to use |
|---|---|---|---|
| `linear_probe` | Cell type separability | accuracy, F1 | Always - your primary metric for most tasks |
| `embedding_quality` | Bio conservation + batch correction | NMI, ARI, ASW, overall | Multi-batch datasets with known cell types |
| `perturbation` | Perturbation response prediction | pearson, MSE, direction_accuracy | Perturb-seq or CRISPR screen data |
| `grn_inference` | Gene regulatory network recovery | AUROC, AUPRC | When you have known GRN ground truth |
| `cz_clustering` | CZI standardized clustering | ARI, NMI | Comparing with other models in ecosystem |
| `cz_embedding` | CZI standardized embedding | silhouette_score | Comparing with other models in ecosystem |
| `cz_label_prediction` | CZI standardized classification | accuracy, F1, AUROC | Comparing with other models in ecosystem |
| `cz_batch_integration` | CZI standardized batch correction | entropy, batch_silhouette | Comparing with other models in ecosystem |

## Linear probe benchmark: Your primary metric

The linear probe is the most important benchmark for most users. It tests whether your embeddings capture cell type identity in a biologically meaningful way.

### How it works

The benchmark freezes your model weights and trains a simple logistic regression classifier on top of the embeddings to predict cell type labels. This tests:

1. Whether cell types form separable clusters in embedding space
2. Whether the model generalizes to new cells of known types
3. Whether embeddings encode enough information for downstream classification

If the linear probe achieves high accuracy, it means your embeddings have learned to separate cell types in a way that transfers to new data. This is analogous to how biologists visually inspect UMAP plots - are cell types clustered together?

### Configuration

```yaml
eval:
  batch_size: 256
  benchmarks:
    - name: linear_probe
      params:
        cell_type_key: cell_type  # Column in adata.obs with cell type labels
        test_size: 0.2            # Hold out 20% for testing
        max_iter: 1000            # Logistic regression iterations
        seed: 42                  # For reproducibility
```

The `cell_type_key` parameter specifies which column in your AnnData `.obs` contains the cell type annotations. Common names include `cell_type`, `celltype`, `cell_ontology_class`, or dataset-specific labels.

### Running the benchmark

You can run benchmarks in two ways:

**Via CLI:**

```bash
scmodelforge benchmark \
  --config eval_config.yaml \
  --model checkpoints/best.ckpt \
  --data eval_dataset.h5ad \
  --output results.json
```

**Via Python API:**

```python
from scmodelforge.eval import get_benchmark
from scmodelforge.eval._utils import extract_embeddings
from scmodelforge.models.hub import load_pretrained_with_vocab
from scmodelforge.tokenizers import get_tokenizer
import anndata as ad

# Load model and gene vocabulary
model, gene_vocab = load_pretrained_with_vocab("checkpoints/my_model/")
tokenizer = get_tokenizer("rank_value", gene_vocab=gene_vocab)
adata = ad.read_h5ad("eval_dataset.h5ad")

# Run benchmark
benchmark = get_benchmark("linear_probe", cell_type_key="cell_type")
embeddings = extract_embeddings(model, adata, tokenizer, device="cuda")
result = benchmark.run(embeddings, adata, "my_dataset")

print(result.summary())
# Output: [linear_probe] my_dataset: accuracy=0.8523, f1_macro=0.8301, f1_weighted=0.8489
```

### Understanding the output

The linear probe reports three metrics:

- **accuracy**: Overall fraction of cells correctly classified
- **f1_macro**: F1 score averaged across cell types (treats rare types equally)
- **f1_weighted**: F1 score weighted by cell type frequency (emphasizes common types)

**Interpreting scores:**

- **> 0.8**: Excellent - embeddings clearly separate cell types
- **0.6 - 0.8**: Good - useful separation, may need fine-tuning for production
- **0.4 - 0.6**: Moderate - embeddings capture some structure but need improvement
- **< 0.4**: Poor - model may not be learning useful biology

Compare against baselines:

- Random baseline (1 / n_cell_types)
- PCA baseline (run linear probe on top PCs)
- Published models on the same dataset

### Output format

Results are saved as JSON:

```json
{
  "benchmark_name": "linear_probe",
  "dataset_name": "my_dataset",
  "metrics": {
    "accuracy": 0.8523,
    "f1_macro": 0.8301,
    "f1_weighted": 0.8489
  },
  "metadata": {
    "n_cells": 15234,
    "n_classes": 24,
    "test_size": 0.2,
    "cell_type_key": "cell_type"
  }
}
```

## Embedding quality benchmark: Bio + batch metrics

The embedding quality benchmark uses scIB (single-cell integration benchmarking) metrics to evaluate both biological conservation and batch correction. This is essential when your data contains multiple batches, donors, or experimental conditions.

### How it works

The benchmark computes:

**Biological conservation (60% of overall score):**

- **NMI (Normalized Mutual Information)**: Agreement between clusters and known cell types
- **ARI (Adjusted Rand Index)**: Similarity between cluster assignments and cell type labels
- **ASW cell type (Adjusted Silhouette Width)**: How well-separated are cell types in embedding space

**Batch correction (40% of overall score):**

- **ASW batch**: Batch mixing within each cell type (higher = better mixing)
- **Graph connectivity**: Whether cells of the same type from different batches form connected components

The overall score balances these: `0.6 * bio_mean + 0.4 * batch_mean`

### Configuration

```yaml
eval:
  batch_size: 256
  benchmarks:
    - name: embedding_quality
      params:
        cell_type_key: cell_type  # Column with cell type labels
        batch_key: batch          # Column with batch labels (or null to skip batch metrics)
        n_neighbors: 15           # For neighborhood graph construction
```

Set `batch_key: null` if your data has no batch effects to evaluate.

### Interpreting metrics

**NMI and ARI** both range from 0 to 1:

- **> 0.7**: Strong agreement between embeddings and cell type structure
- **0.4 - 0.7**: Moderate agreement, clusters partially align with biology
- **< 0.4**: Weak agreement, embeddings may not reflect cell types

**ASW cell type** ranges from -1 to 1:

- **> 0.5**: Cell types are well-separated
- **0.25 - 0.5**: Moderate separation
- **< 0.25**: Poor separation or overlapping cell types

**ASW batch** (higher is better for batch mixing):

- **> 0.7**: Excellent batch mixing
- **0.5 - 0.7**: Good mixing
- **< 0.5**: Batches remain separated (poor batch correction)

**Overall score:**

- **> 0.7**: Competitive with published integration methods
- **0.5 - 0.7**: Reasonable performance
- **< 0.5**: Model may need architectural changes or more training

### Example output

```python
result = benchmark.run(embeddings, adata, "pbmc_batched")
print(result.summary())
# [embedding_quality] pbmc_batched: ari=0.7234, asw_batch=0.6891, asw_cell_type=0.5123,
#   graph_connectivity=0.8456, nmi=0.7012, overall=0.6789
```

## Running multiple benchmarks

You can configure multiple benchmarks in a single config file and run them together:

```yaml
eval:
  batch_size: 256
  benchmarks:
    # First benchmark: cell type classification
    - name: linear_probe
      params:
        cell_type_key: cell_type
        test_size: 0.2

    # Second benchmark: integration quality
    - name: embedding_quality
      params:
        cell_type_key: cell_type
        batch_key: donor_id
        n_neighbors: 15

    # Third benchmark: clustering (CZI standardized)
    - name: cz_clustering
      params:
        cell_type_key: cell_type
```

The evaluation harness extracts embeddings once per dataset, then runs all benchmarks sequentially:

```python
from scmodelforge.eval import EvalHarness
from scmodelforge.config.schema import load_config

# Load configuration
config = load_config("eval_config.yaml")

# Create harness and run all benchmarks
harness = EvalHarness.from_config(config.eval)
results = harness.run(
    model=model,
    datasets={"my_data": adata},
    tokenizer=tokenizer,
    batch_size=256,
    device="cuda"
)

# Print all results
for result in results:
    print(result.summary())
```

## Evaluation during training: AssessmentCallback

You can run benchmarks periodically during pretraining to monitor whether your model is learning useful representations. This helps catch issues early and choose the best checkpoint.

### Configuration

Add evaluation settings to your training config:

```yaml
training:
  batch_size: 64
  max_epochs: 20
  # ... other training settings ...

eval:
  every_n_epochs: 2  # Run benchmarks every 2 epochs
  batch_size: 256
  benchmarks:
    - name: linear_probe
      params:
        cell_type_key: cell_type
    - name: embedding_quality
      params:
        cell_type_key: cell_type
        batch_key: batch
```

### How it works

The `AssessmentCallback` runs during validation:

1. At the end of every N epochs, it extracts embeddings from your validation set
2. Runs all configured benchmarks on those embeddings
3. Logs metrics to your experiment tracker (WandB, TensorBoard, etc.)

Metrics appear in logs with the format:

```
assessment/{benchmark_name}/{dataset_name}/{metric_name}
```

For example:

- `assessment/linear_probe/val_data/accuracy`
- `assessment/embedding_quality/val_data/overall`
- `assessment/embedding_quality/val_data/nmi`

This lets you track how embeddings quality evolves during training, similar to tracking training loss.

### Best practices

- Start with `every_n_epochs: 5` for large datasets to avoid slowing down training
- Use smaller validation sets (1-5K cells) for faster benchmarking
- Monitor linear probe accuracy as your primary signal for early stopping
- If linear probe accuracy plateaus or decreases, you may be overfitting

## Python API for evaluation

For programmatic use in notebooks or custom scripts:

### Extract embeddings only

```python
from scmodelforge.eval._utils import extract_embeddings
from scmodelforge.models.hub import load_pretrained_with_vocab
from scmodelforge.tokenizers import get_tokenizer
import anndata as ad

# Load model and tokenizer (see Hub Integration tutorial for details)
model, gene_vocab = load_pretrained_with_vocab("checkpoints/my_model/")
tokenizer = get_tokenizer("rank_value", gene_vocab=gene_vocab)

adata = ad.read_h5ad("my_data.h5ad")
embeddings = extract_embeddings(
    model=model,
    adata=adata,
    tokenizer=tokenizer,
    batch_size=256,
    device="cuda"
)

# embeddings is a numpy array of shape (n_cells, hidden_dim)
# Use it with any downstream tool
import scanpy as sc
adata.obsm["X_scmf"] = embeddings
sc.pp.neighbors(adata, use_rep="X_scmf")
sc.tl.umap(adata)
sc.pl.umap(adata, color="cell_type")
```

### Run specific benchmarks

```python
from scmodelforge.eval import get_benchmark

# Instantiate a benchmark
benchmark = get_benchmark(
    "linear_probe",
    cell_type_key="cell_type",
    test_size=0.2,
    seed=42
)

# Run on precomputed embeddings
result = benchmark.run(embeddings, adata, dataset_name="my_data")

# Access results
print(f"Accuracy: {result.metrics['accuracy']:.4f}")
print(f"F1 (macro): {result.metrics['f1_macro']:.4f}")
print(f"Number of cells: {result.metadata['n_cells']}")
print(f"Number of classes: {result.metadata['n_classes']}")
```

### Run full harness

```python
from scmodelforge.eval import EvalHarness

# Create harness with multiple benchmarks
benchmarks = [
    get_benchmark("linear_probe", cell_type_key="cell_type"),
    get_benchmark("embedding_quality", cell_type_key="cell_type", batch_key="batch"),
]
harness = EvalHarness(benchmarks)

# Run on multiple datasets
results = harness.run(
    model=model,
    datasets={
        "train": adata_train,
        "test": adata_test,
        "external": adata_external
    },
    tokenizer=tokenizer,
    batch_size=256,
    device="cuda"
)

# Save results to JSON
import json
with open("benchmark_results.json", "w") as f:
    json.dump([r.to_dict() for r in results], f, indent=2)
```

## Perturbation benchmark

For users with Perturb-seq, CRISPR screen, or drug treatment data, the perturbation benchmark evaluates whether embeddings can predict cellular responses to interventions.

### How it works

The benchmark:

1. Computes expression deltas for each perturbation vs. control
2. Trains a Ridge regression: `mean_perturbed_embedding → expression_delta`
3. Evaluates on held-out perturbations (not held-out cells)
4. Reports Pearson correlation, MSE, and direction accuracy
5. Compares against a mean-shift baseline

This tests whether embeddings encode perturbation state in a way that generalizes to unseen interventions.

### Quick example

```yaml
eval:
  benchmarks:
    - name: perturbation
      params:
        perturbation_key: perturbation  # Column with perturbation labels
        control_label: control           # Value for control cells
        test_size: 0.3                   # Hold out 30% of perturbations
```

See the [Perturbation Response Prediction](perturbation_prediction.md) tutorial for a complete guide.

## Gene regulatory network (GRN) benchmark

For users with known gene regulatory networks (e.g., from ChIP-seq, TF binding databases, or validated literature), this benchmark tests whether gene relationships are preserved in embedding space.

### How it works

The benchmark:

1. Computes gene-level representations (mean embedding of cells expressing each gene)
2. Calculates cosine similarity between gene pairs
3. Compares against ground-truth network edges
4. Reports AUROC and AUPRC

High scores indicate that related genes (e.g., TF and target) have similar embeddings.

### Configuration

```yaml
eval:
  benchmarks:
    - name: grn_inference
      params:
        network_path: grn_ground_truth.tsv  # TSV with gene1, gene2, weight
        min_cells: 10                        # Min cells expressing each gene
```

The ground truth file should be a TSV with columns:

```
gene1    gene2    weight
SOX2     NANOG    0.85
POU5F1   SOX2     0.72
...
```

Edges with `weight > 0` are treated as positives.

## CZI standardized benchmarks

The Chan Zuckerberg Initiative's Virtual Cells ecosystem defines standardized benchmarks for comparing models across the community. scModelForge provides adapters for four CZI benchmark types.

### Installation

CZI benchmarks are optional:

```bash
pip install "scModelForge[cz-benchmarks]"
```

### Available benchmarks

| Benchmark | CZI task | Measures |
|---|---|---|
| `cz_clustering` | ClusteringTask | ARI, NMI vs. ground truth |
| `cz_embedding` | EmbeddingTask | Silhouette score |
| `cz_label_prediction` | MetadataLabelPredictionTask | Accuracy, F1, precision, recall, AUROC |
| `cz_batch_integration` | BatchIntegrationTask | Entropy per cell, batch silhouette |

### Configuration

```yaml
eval:
  benchmarks:
    - name: cz_label_prediction
      params:
        label_key: cell_type
```

All CZI benchmarks work with the existing EvalHarness, AssessmentCallback, and CLI.

### When to use

Use CZI benchmarks when:

- Comparing your model against other models in the Virtual Cells ecosystem
- Publishing results that need standardized metrics
- Contributing to community benchmarking efforts

For most internal evaluation, the built-in `linear_probe` and `embedding_quality` benchmarks are simpler and more interpretable.

## Interpreting results: Practical guidance

### What makes a good score?

Performance depends heavily on:

- **Dataset difficulty**: Rare cell types, noisy labels, or subtle differences make classification harder
- **Data quality**: Well-annotated, high-quality data yields higher scores
- **Model size**: Larger models generally perform better (but require more data to train)
- **Training data**: More diverse pretraining data improves generalization

**Linear probe accuracy benchmarks:**

- **> 0.85**: Excellent for well-annotated datasets like PBMC
- **0.75 - 0.85**: Good for complex tissues or noisy labels
- **0.65 - 0.75**: Moderate - consider more pretraining or fine-tuning
- **< 0.65**: May need architectural changes or more data

**Embedding quality overall score:**

- **> 0.7**: Competitive with state-of-the-art integration methods
- **0.5 - 0.7**: Reasonable performance
- **< 0.5**: Needs improvement

### Compare against baselines

Always compare your model against:

1. **Random baseline**: Random embeddings or chance-level classification
2. **PCA baseline**: Classical dimensionality reduction (often surprisingly strong)
3. **Your previous checkpoints**: Is training improving over time?
4. **Published models**: How does your model compare to Geneformer, scGPT, etc.?

### Common issues and solutions

**Linear probe accuracy is low (<0.5):**

- Check that cell type labels are correct and match between train/test
- Verify your tokenizer is working correctly
- Ensure sufficient pretraining (at least 5-10 epochs)
- Try a larger model or more diverse pretraining data

**Embedding quality overall score is low (<0.4):**

- Check if batch effects are too strong in your data
- Ensure your data has sufficient cells per batch and cell type
- Consider multi-batch pretraining or batch correction preprocessing

**Metrics improve during training then degrade:**

- You may be overfitting to the pretraining task
- Reduce learning rate, increase dropout, or use more data augmentation
- Add more diverse data to the training set

**Batch metrics are good but bio metrics are poor:**

- Model is removing biology along with batch effects
- Reduce batch correction strength or check preprocessing
- Ensure cell type labels are accurate

### More data helps more than more parameters

In general:

- 2x more training cells → 10-20% accuracy improvement
- 2x larger model → 5-10% accuracy improvement

If you're deciding between:

- Training a larger model on the same data, or
- Training the same model on more diverse data

Choose more data. Foundation models benefit most from diversity of cell types, tissues, and experimental conditions.

## What's next?

After evaluating your model:

1. **Fine-tune for specific tasks**: See [Fine-tuning for Cell Type Annotation](finetuning_cell_type.md)
2. **Share your model**: See [HuggingFace Hub Integration](hub_models.md)
3. **Implement custom benchmarks**: See [Building Custom Benchmarks](custom_benchmark.md)
4. **Integrate with existing workflows**: See [scverse Ecosystem Integration](scverse_integration.md)

For questions or issues with evaluation, please open an issue on GitHub or consult the [API documentation](../api/eval.rst).
