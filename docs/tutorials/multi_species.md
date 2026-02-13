# Multi-species Analysis with scModelForge

This tutorial demonstrates how to train cross-species single-cell foundation models using scModelForge. You will learn how to combine human and mouse data in a unified training pipeline using ortholog mapping.

## Introduction

Many computational biology research questions require integrating data across species. For example:

- Validating mouse model findings in human tissue samples
- Comparing developmental processes across mammals
- Leveraging abundant mouse data to improve human models
- Building transferable cell type classifiers

scModelForge enables cross-species pretraining by mapping mouse gene names to their human orthologs. The model learns a unified representation space where both human and mouse cells are tokenized using a common human gene vocabulary.

## How It Works

scModelForge's multi-species support is built on three key components:

1. **Ortholog mapping**: A bundled Ensembl ortholog table containing 453 human-mouse gene pairs (419 one-to-one, 34 one-to-many mappings). Mouse gene names are automatically translated to their human orthologs before tokenization.

2. **Unified vocabulary**: `GeneVocab.multi_species()` creates a vocabulary in the canonical human namespace. All cells, regardless of source organism, are tokenized using this shared vocabulary.

3. **Transparent translation**: `OrthologMapper` handles gene name translation in `AnnDataStore`. From the model's perspective, all input is human genes.

The architecture looks like this:

```
Mouse data (Actb, Gapdh, ...) → OrthologMapper → Human genes (ACTB, GAPDH, ...)
                                                        ↓
Human data (ACTB, GAPDH, ...) ────────────────────→ GeneVocab → Tokenization
                                                        ↓
                                                    Model training
```

## Prerequisites

Install scModelForge with standard dependencies:

```bash
pip install scmodelforge
```

No additional packages are required for multi-species support. The ortholog table is bundled with the package.

## Preparing Multi-species Data

For this tutorial, we will combine human and mouse lung tissue data. Start by loading your datasets:

```python
import scanpy as sc

# Human lung tissue from CELLxGENE Census or local file
human_adata = sc.read_h5ad("data/human_lung.h5ad")
print(f"Human data: {human_adata.n_obs} cells, {human_adata.n_vars} genes")

# Mouse lung tissue from Tabula Muris or similar
mouse_adata = sc.read_h5ad("data/mouse_lung.h5ad")
print(f"Mouse data: {mouse_adata.n_obs} cells, {mouse_adata.n_vars} genes")
```

Ensure both datasets are in standard AnnData format with gene names as `var_names`. The human dataset should use human gene symbols (e.g. ACTB, TP53), and the mouse dataset should use mouse gene symbols (e.g. Actb, Trp53).

## Configuration for Multi-species Training

Create a YAML configuration file that enables multi-species support:

```yaml
# configs/multi_species_pretraining.yaml

data:
  source: local
  paths:
    - ./data/human_lung.h5ad
    - ./data/mouse_lung.h5ad
  gene_vocab: human_protein_coding

  # Enable multi-species support
  multi_species:
    enabled: true
    organisms: ["human", "mouse"]
    canonical_organism: human
    include_one2many: false

  preprocessing:
    normalize: library_size
    target_sum: 10000
    log1p: true

  max_genes: 2048
  num_workers: 4

tokenizer:
  strategy: rank_value
  max_genes: 2048
  gene_vocab: human_protein_coding
  prepend_cls: true
  masking:
    mask_ratio: 0.15
    random_replace_ratio: 0.1
    keep_ratio: 0.1

model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1
  max_seq_len: 2048
  pooling: cls
  activation: gelu
  use_expression_values: true
  pretraining_task: masked_gene_prediction

training:
  batch_size: 64
  max_epochs: 50
  seed: 42
  precision: bf16-mixed
  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 0.01
  scheduler:
    name: cosine_warmup
    warmup_steps: 1000
    total_steps: 100000
  gradient_clip: 1.0
  logger: wandb
  wandb_project: scmodelforge-multi-species
  run_name: human-mouse-lung
  checkpoint_dir: ./checkpoints/multi_species
```

### Configuration Explanation

The key section is `data.multi_species`:

- `enabled: true`: Activates ortholog mapping
- `organisms: ["human", "mouse"]`: Species to include in training
- `canonical_organism: human`: Target namespace for all gene names
- `include_one2many: false`: Exclude ambiguous one-to-many mappings (recommended)

With this configuration, mouse genes will be automatically translated to human orthologs during data loading. The model sees only human gene names.

## Training the Model

Run training using the CLI:

```bash
scmodelforge train --config configs/multi_species_pretraining.yaml
```

During training, you will see log messages confirming ortholog mapping:

```
INFO - Loaded ortholog table: 453 entries from human_mouse_orthologs.tsv
INFO - Filtered to one2one orthologs: 419 entries
INFO - Translated 15234 genes from mouse to human: 12891 mapped, 2343 unmapped
```

Unmapped genes (those without orthologs) will be treated as unknown tokens.

## Using the Python API

For programmatic control, use the Python API directly:

```python
from scmodelforge.data import GeneVocab, AnnDataStore, CellDataset
from scmodelforge.data.ortholog_mapper import OrthologMapper
from scmodelforge.tokenizers import get_tokenizer
import scanpy as sc

# Load datasets
human_adata = sc.read_h5ad("data/human_lung.h5ad")
mouse_adata = sc.read_h5ad("data/mouse_lung.h5ad")

# Create ortholog mapper (uses bundled table by default)
mapper = OrthologMapper(
    organisms=["human", "mouse"],
    canonical_organism="human",
    include_one2many=False,
)

print(f"Loaded {mapper.n_mapped} ortholog mappings")

# Build multi-species vocabulary
vocab = GeneVocab.multi_species(
    organisms=["human", "mouse"],
    include_one2many=False,
)

print(f"Vocabulary size: {len(vocab)} human genes")

# Create stores with transparent ortholog mapping
human_store = AnnDataStore(human_adata, vocab)

# For mouse data, provide the mapper and source organism
mouse_store = AnnDataStore(
    mouse_adata,
    vocab,
    ortholog_mapper=mapper,
    source_organism="mouse",
)

# Create datasets
human_dataset = CellDataset(human_adata, vocab)
mouse_dataset = CellDataset(mouse_adata, vocab)

# Combine datasets
combined_dataset = CellDataset(
    [human_adata, mouse_adata],
    vocab,
)

print(f"Combined dataset: {len(combined_dataset)} cells")
```

The `AnnDataStore` automatically translates mouse genes when `ortholog_mapper` and `source_organism` are provided. When using `CellDataset` with a list of AnnData objects, scModelForge handles the mapping internally.

## Custom Ortholog Tables

By default, scModelForge uses a bundled Ensembl ortholog table. You can provide a custom table if needed:

```yaml
data:
  multi_species:
    enabled: true
    organisms: ["human", "mouse"]
    canonical_organism: human
    include_one2many: false
    ortholog_table: ./data/custom_orthologs.tsv
```

The custom table must be a TSV file with the following columns:

```
human_gene_symbol	mouse_gene_symbol	human_ensembl_id	mouse_ensembl_id	orthology_type
TP53	Trp53	ENSG00000141510	ENSMUSG00000059552	one2one
BRCA1	Brca1	ENSG00000012048	ENSMUSG00000017146	one2one
```

Required columns:
- `human_gene_symbol`: Human gene symbol (canonical namespace)
- `mouse_gene_symbol`: Mouse gene symbol (source namespace)
- `orthology_type`: `"one2one"` or `"one2many"`

The Ensembl ID columns are optional but recommended for disambiguation.

## One-to-Many Orthologs

Some mouse genes have multiple human orthologs (one-to-many mappings). By default, these are excluded to avoid ambiguity. You can include them if needed:

```yaml
data:
  multi_species:
    include_one2many: true
```

When `include_one2many: true`, if a mouse gene maps to multiple human genes, the first mapping in the ortholog table is used. This increases gene coverage but introduces potential ambiguity.

**Recommendation**: Start with `include_one2many: false` (default). Only enable one-to-many mappings if you have many unmapped genes and understand the tradeoffs.

## Evaluating Cross-species Models

After pretraining, evaluate the model on held-out data from each species:

```yaml
# configs/multi_species_evaluation.yaml

data:
  source: local
  paths:
    - ./data/human_lung_test.h5ad
  gene_vocab: human_protein_coding
  multi_species:
    enabled: true
    organisms: ["human", "mouse"]
    canonical_organism: human

eval:
  benchmarks:
    - name: embedding_quality
      datasets:
        - ./data/human_lung_test.h5ad
        - ./data/mouse_lung_test.h5ad
      bio_label: cell_type
      batch_label: organism
```

Run evaluation:

```bash
scmodelforge benchmark \
  --config configs/multi_species_evaluation.yaml \
  --model checkpoints/multi_species/best_model.ckpt \
  --output results/multi_species_eval.json
```

The embedding quality benchmark will compute biological conservation (NMI, ARI) and batch mixing scores. Ideally, the model should cluster cells by cell type (high biological score) regardless of species (high batch mixing).

## Troubleshooting

### Low Ortholog Mapping Rate

If you see many unmapped genes:

```
INFO - Translated 20000 genes from mouse to human: 5000 mapped, 15000 unmapped
```

Possible causes:
- Gene names are not in standard symbol format (e.g. Ensembl IDs instead of symbols)
- Gene names use outdated nomenclature
- Genes are not protein-coding (the bundled table focuses on protein-coding genes)

Solutions:
- Verify gene name format: `mouse_adata.var_names[:10]`
- Convert Ensembl IDs to gene symbols using biomaRt or mygene
- Use a custom ortholog table with broader coverage

### Model Performance Issues

If the cross-species model underperforms compared to single-species models:

- Check that both datasets use consistent preprocessing (normalization, log transformation)
- Ensure sufficient representation from both species in training data
- Consider species-specific batch correction before training
- Evaluate separately on human and mouse test sets to identify species-specific issues

## What's Next

Now that you understand multi-species training, explore these related tutorials:

- **Pretraining Tutorial**: Learn about different pretraining strategies and model architectures
- **Large-scale Data Tutorial**: Scale to millions of cells using sharding and distributed training
- **Fine-tuning Tutorial**: Adapt your multi-species pretrained model to downstream tasks

## Complete Example

Here is a complete script for multi-species pretraining:

```python
from pathlib import Path
import scanpy as sc
from scmodelforge.config.schema import load_config
from scmodelforge.training.pipeline import TrainingPipeline

# Load and preprocess data
human_adata = sc.read_h5ad("data/human_lung.h5ad")
mouse_adata = sc.read_h5ad("data/mouse_lung.h5ad")

# Basic QC
sc.pp.filter_cells(human_adata, min_genes=200)
sc.pp.filter_genes(human_adata, min_cells=3)
sc.pp.filter_cells(mouse_adata, min_genes=200)
sc.pp.filter_genes(mouse_adata, min_cells=3)

# Save processed data
human_adata.write_h5ad("data/human_lung_processed.h5ad")
mouse_adata.write_h5ad("data/mouse_lung_processed.h5ad")

# Load training configuration
config = load_config("configs/multi_species_pretraining.yaml")

# Run training pipeline
pipeline = TrainingPipeline(config)
trainer = pipeline.run()

print(f"Training complete. Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
```

## Summary

Multi-species analysis in scModelForge requires three steps:

1. Enable `multi_species` in your data configuration
2. Provide datasets with standard gene symbol nomenclature
3. Train as usual - ortholog mapping happens automatically

The model learns a unified representation space where cells from different species are directly comparable. This enables powerful cross-species transfer learning and comparative biology analyses.
