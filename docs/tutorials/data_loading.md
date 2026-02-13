# Data Loading and Preprocessing

This tutorial covers the complete data pipeline in scModelForge, from raw single-cell data to training-ready tensors. If you are familiar with `scanpy` and `AnnData` but new to transformer models, this guide will help you understand how scModelForge loads and processes your data.

## Overview

scModelForge is designed to work seamlessly with the scverse ecosystem. The toolkit accepts AnnData objects (`.h5ad` files) and supports multiple data sources:

- **Local files**: Standard `.h5ad` files on your filesystem
- **CELLxGENE Census**: Query and load data directly from the Census
- **Cloud storage**: S3, Google Cloud Storage (GCS), and Azure Blob Storage

The data pipeline transforms raw expression data into tokenized representations suitable for transformer models, handling gene vocabulary alignment, preprocessing, and batching automatically.

## Loading Local H5AD Files

The simplest way to load data is from local `.h5ad` files. You can specify one or more files in your configuration:

```yaml
data:
  source: local
  paths:
    - ./data/pbmc_dataset.h5ad
    - ./data/lung_dataset.h5ad
  gene_vocab: human_protein_coding
  max_genes: 2048
  num_workers: 4
```

Multiple files are concatenated automatically, providing a unified view across datasets. This is useful when you have data from different experiments or batches.

### Python API

For programmatic control, you can use the Python API directly:

```python
import anndata as ad
from scmodelforge.data import GeneVocab, AnnDataStore, CellDataset

# Load your AnnData object
adata = ad.read_h5ad("./data/pbmc_dataset.h5ad")

# Create a gene vocabulary (see next section)
gene_vocab = GeneVocab.from_adata(adata)

# Create a data store (handles gene alignment)
store = AnnDataStore(
    adatas=[adata],
    gene_vocab=gene_vocab,
    obs_keys=["cell_type", "batch"]  # metadata to preserve
)

# Create a dataset for PyTorch
dataset = CellDataset(store, max_genes=2048)

# Access a single cell
cell = dataset[0]
print(f"Cell shape: {cell.expression.shape}")
print(f"Cell type: {cell.metadata.get('cell_type', 'unknown')}")
```

The `AnnDataStore` handles gene alignment automatically, ensuring that all cells are mapped to the same gene vocabulary regardless of which dataset they came from.

## Gene Vocabulary

The **gene vocabulary** is a crucial concept in scModelForge. It defines the mapping between gene names and integer indices that the model uses. Think of it as the model's "dictionary" of genes.

### Why Gene Vocabulary Matters

Transformer models use embedding layers that require fixed vocabulary sizes. The gene vocabulary determines:

1. The size of the model's gene embedding layer
2. Which genes are included in the training data
3. How genes from different datasets are aligned

### Creating a Gene Vocabulary

There are several ways to create a gene vocabulary:

**From an AnnData object:**

```python
from scmodelforge.data import GeneVocab

# Use the genes in your dataset
gene_vocab = GeneVocab.from_adata(adata)
print(f"Vocabulary size: {len(gene_vocab)} genes")
```

**From a saved vocabulary file:**

```python
# Load a previously saved gene vocabulary
gene_vocab = GeneVocab.from_file("gene_vocab.json")
```

**From a custom gene list:**

```python
# Define your own gene set
my_genes = ["GAPDH", "ACTB", "TP53", "BRCA1", ...]
gene_vocab = GeneVocab.from_genes(my_genes)
```

In YAML configuration, you specify the vocabulary as a string:

```yaml
data:
  gene_vocab: human_protein_coding  # uses the preset
  # or
  # gene_vocab: /path/to/gene_list.txt  # load from file
```

### Multi-Species Vocabularies

If you work with multiple species (e.g., human and mouse), scModelForge can map orthologous genes to a canonical namespace:

```python
# Create a multi-species vocabulary
gene_vocab = GeneVocab.multi_species(
    organisms=["human", "mouse"],
    canonical_organism="human",
    include_one2many=False
)
```

This uses Ensembl ortholog mappings to translate mouse gene names to their human equivalents, enabling cross-species training.

## Preprocessing Pipeline

scModelForge applies standard single-cell preprocessing steps. These can be configured in your YAML file or applied offline using the CLI.

### Preprocessing Configuration

The preprocessing pipeline consists of three optional steps:

1. **Library-size normalization**: Normalize each cell to a target total count
2. **Log transformation**: Apply log(x + 1) transformation
3. **Highly variable gene selection**: Keep only the top N most variable genes

Example configuration:

```yaml
data:
  preprocessing:
    normalize: library_size
    target_sum: 10000
    log1p: true
    hvg_selection: 2000  # keep top 2000 HVGs
```

This configuration:
- Normalizes each cell to 10,000 total counts
- Applies log1p transformation
- Selects the 2,000 most highly variable genes

### Offline Preprocessing with the CLI

For large datasets, it is more efficient to preprocess once and save the result rather than preprocessing on-the-fly during training. Use the `preprocess` command:

```bash
scmodelforge preprocess \
  --input ./data/raw_pbmc.h5ad \
  --output ./data/processed_pbmc.h5ad \
  --hvg 2000
```

This creates a preprocessed `.h5ad` file that can be used directly for training. The preprocessing parameters can also be specified via a config file:

```bash
scmodelforge preprocess \
  --config ./configs/preprocess_config.yaml \
  --output ./data/processed_pbmc.h5ad
```

Example `preprocess_config.yaml`:

```yaml
data:
  preprocessing:
    normalize: library_size
    target_sum: 10000
    log1p: true
    hvg_selection: 2000
```

The CLI supports cloud storage paths, so you can preprocess data stored on S3 or GCS:

```bash
scmodelforge preprocess \
  --input s3://my-bucket/raw_data.h5ad \
  --output ./data/processed.h5ad \
  --hvg 2000
```

## CELLxGENE Census Integration

The [CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) provides access to millions of standardized single-cell profiles. scModelForge can query and load Census data directly.

### Querying Census Data

To load data from Census, set `source: cellxgene_census` and specify filters:

```yaml
data:
  source: cellxgene_census
  census:
    organism: "Homo sapiens"
    filters:
      tissue: ["lung", "heart"]
      cell_type: ["T cell", "B cell"]
      is_primary_data: true
    obs_columns: ["cell_type", "tissue", "donor_id", "assay"]
  gene_vocab: human_protein_coding
  max_genes: 2048
```

This configuration:
- Queries human cells from lung and heart tissues
- Filters to T cells and B cells
- Includes only primary data (not reprocessed)
- Preserves cell type, tissue, donor, and assay metadata

### Advanced Census Queries

For complex queries, you can provide raw SOMA filter expressions:

```yaml
data:
  source: cellxgene_census
  census:
    organism: "Homo sapiens"
    obs_value_filter: "tissue in ['lung', 'heart'] and is_primary_data == True"
    obs_columns: ["cell_type", "tissue", "donor_id"]
```

### Python API for Census

```python
from scmodelforge.data import load_census_adata, GeneVocab, AnnDataStore

# Load Census data
adata = load_census_adata(
    organism="Homo sapiens",
    filters={"tissue": ["lung"], "is_primary_data": True},
    obs_columns=["cell_type", "tissue"]
)

# Continue with standard workflow
gene_vocab = GeneVocab.from_adata(adata)
store = AnnDataStore([adata], gene_vocab)
```

**Note:** Census integration requires the optional `census` dependency:

```bash
pip install "scmodelforge[census]"
```

## Cloud Storage Support

scModelForge can read `.h5ad` files directly from cloud storage using S3, Google Cloud Storage, or Azure Blob Storage URLs.

### Supported Schemes

- **S3**: `s3://bucket-name/path/to/data.h5ad`
- **Google Cloud Storage**: `gs://bucket-name/path/to/data.h5ad` or `gcs://...`
- **Azure Blob Storage**: `az://container/path/to/data.h5ad` or `abfs://...`

### Configuration

Simply use cloud URLs in your paths:

```yaml
data:
  source: local
  paths:
    - s3://my-bucket/datasets/pbmc.h5ad
    - gs://another-bucket/lung_data.h5ad
  cloud:
    storage_options:
      anon: true  # for public buckets
  gene_vocab: human_protein_coding
```

### Authentication

For private buckets, configure authentication via `storage_options`:

**S3 (AWS):**

```yaml
cloud:
  storage_options:
    key: YOUR_ACCESS_KEY_ID
    secret: YOUR_SECRET_ACCESS_KEY
    endpoint_url: https://s3.us-west-2.amazonaws.com  # optional
```

Alternatively, use environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

**Google Cloud Storage:**

```yaml
cloud:
  storage_options:
    token: /path/to/service-account-key.json
```

Or use Application Default Credentials (`gcloud auth application-default login`).

### Caching

Cloud files are downloaded to a local cache directory before reading. Configure the cache location:

```yaml
cloud:
  cache_dir: ./cache/cloud_data
```

**Note:** Cloud support requires the optional `cloud` dependency:

```bash
pip install "scmodelforge[cloud]"
```

## CellDataset and DataLoader

Once your data is loaded and preprocessed, scModelForge wraps it in PyTorch-compatible dataset and dataloader objects.

### Creating a Dataset

```python
from scmodelforge.data import CellDataset, CellDataLoader

# Create a dataset
dataset = CellDataset(
    store,  # AnnDataStore instance
    max_genes=2048  # maximum genes per cell
)

# Check dataset size
print(f"Total cells: {len(dataset)}")

# Access a single cell
cell = dataset[0]
print(f"Gene IDs: {cell.gene_ids[:10]}")  # first 10 genes
print(f"Expression: {cell.expression[:10]}")  # first 10 values
```

### Creating a DataLoader

For training, use `CellDataLoader` to batch cells efficiently:

```python
loader = CellDataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # speeds up GPU transfer
)

# Iterate over batches
for batch in loader:
    print(f"Batch gene IDs shape: {batch['gene_ids'].shape}")
    print(f"Batch expression shape: {batch['expression'].shape}")
    break
```

The dataloader automatically handles:
- Batching cells together
- Padding sequences to the same length
- Converting to PyTorch tensors
- Collating metadata

## Inspecting Your Data

Before training, verify that your data is loaded correctly.

### Check Gene Vocabulary Coverage

```python
# Count how many genes overlap between data and vocabulary
adata = ad.read_h5ad("./data/pbmc.h5ad")
vocab_genes = set(gene_vocab.genes)
data_genes = set(adata.var_names)

overlap = vocab_genes & data_genes
print(f"Overlap: {len(overlap)} / {len(data_genes)} genes ({100*len(overlap)/len(data_genes):.1f}%)")
print(f"Missing from vocab: {len(data_genes - overlap)} genes")
```

### Inspect Dataset Statistics

```python
# Sample cells from the dataset
import numpy as np

n_samples = 100
sample_indices = np.random.choice(len(dataset), n_samples, replace=False)

gene_counts = []
for idx in sample_indices:
    cell = dataset[idx]
    gene_counts.append(len(cell.gene_ids))

print(f"Average genes per cell: {np.mean(gene_counts):.1f}")
print(f"Min genes: {np.min(gene_counts)}, Max genes: {np.max(gene_counts)}")
```

### Peek at a Sample Cell

```python
# Examine the first cell in detail
cell = dataset[0]

print(f"Cell ID: {cell.cell_id}")
print(f"Number of genes: {len(cell.gene_ids)}")
print(f"Metadata: {cell.metadata}")

# Look at top 5 expressed genes
top5_idx = np.argsort(cell.expression)[-5:]
for idx in top5_idx:
    gene_id = cell.gene_ids[idx]
    gene_name = gene_vocab.idx_to_gene[gene_id]
    expr = cell.expression[idx]
    print(f"  {gene_name}: {expr:.2f}")
```

## Streaming Large Datasets

For datasets too large to fit in memory, scModelForge provides streaming mode:

```yaml
data:
  source: local
  paths:
    - ./data/large_dataset.h5ad
  streaming: true
  streaming_chunk_size: 10000
  streaming_shuffle_buffer: 10000
  gene_vocab: human_protein_coding
```

Streaming mode reads the `.h5ad` file in chunks, keeping only a small buffer in memory. This enables training on arbitrarily large datasets.

## What's Next

Now that you understand how to load and preprocess data, explore these topics:

- **[Tokenization Strategies](tokenization_guide.md)**: Learn how expression data is converted to tokens for transformer models
- **[Pretraining](pretraining.md)**: Train a foundation model from scratch
- **[Fine-tuning](finetuning_cell_type.md)**: Adapt a pretrained model to downstream tasks

For advanced data features, see:

- **[Multi-species Training](multi_species.md)**: Using ortholog mapping for cross-species models
- **[Perturbation Data](perturbation_prediction.md)**: Working with genetic or chemical perturbations
- **[Large-scale Data Handling](large_scale_data.md)**: Memory-mapped storage, sharding, and distributed training
