# Data Module (`scmodelforge.data`)

Load, preprocess, and iterate over single-cell gene expression datasets.

## Overview

The `scmodelforge.data` module provides the complete data pipeline for single-cell foundation model pretraining. It transforms raw AnnData objects into batched, vocabulary-aligned, preprocessed cell representations ready for tokenization and model input.

The typical pipeline flow is:

1. **Load** — Read `.h5ad` files or query CELLxGENE Census via `load_adata()` or `load_census_adata()`
2. **Vocabulary** — Build or load a `GeneVocab` mapping gene names to integer indices
3. **Preprocessing** — Configure on-the-fly normalization and log-transformation via `PreprocessingPipeline`
4. **Dataset** — Wrap data in `CellDataset` for per-cell iteration with gene alignment
5. **Dataloader** — Batch cells with `CellDataLoader`, handling variable-length sequences via padding

The design emphasizes memory efficiency (lazy loading via `AnnDataStore`, sparse matrix support) and flexibility (configurable preprocessing, multiple data sources, Census integration). All components integrate seamlessly with PyTorch's training loops and the `scmodelforge.tokenizers` module.

This module supports both local `.h5ad` files and direct queries to the CZ CELLxGENE Census for large-scale pretraining on standardized human and mouse datasets.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| `GeneVocab` | Gene name to integer index mapping with special tokens |
| `PreprocessingPipeline` | Configurable per-cell preprocessing (normalization, log1p) |
| `CellDataset` | PyTorch Dataset yielding individual cells with gene alignment |
| `CellDataLoader` | PyTorch DataLoader with variable-length padding and batching |
| `AnnDataStore` | Lazy multi-dataset manager with unified gene alignment |
| `build_obs_value_filter()` | Convert structured filters to Census SOMA query strings |
| `load_census_adata()` | Load AnnData from CELLxGENE Census |
| `load_adata()` | Unified dispatcher for local or Census data sources |
| `select_highly_variable_genes()` | Simple HVG selection via Fano factor |
| `collate_cells()` | Custom collate function for variable-length cells |

---

## API Reference

### `GeneVocab`

Unified gene vocabulary across datasets. Maps gene names (or Ensembl IDs) to integer indices. The first `NUM_SPECIAL_TOKENS` (4) indices are reserved for special tokens: `<pad>` (0), `<unk>` (1), `<mask>` (2), `<cls>` (3).

```python
from scmodelforge.data import GeneVocab
```

#### Constructor

```python
GeneVocab(gene_to_idx: dict[str, int])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene_to_idx` | `dict[str, int]` | *required* | Mapping from gene name to index. Indices must start at 4 or higher (0-3 are reserved for special tokens). |

**Raises:**
- `ValueError` — If any gene index is less than `NUM_SPECIAL_TOKENS` (4).

#### Class Methods

##### `from_genes()`

```python
@classmethod
GeneVocab.from_genes(genes: list[str] | np.ndarray) -> GeneVocab
```

Build a vocabulary from a list of gene names. Indices are assigned sequentially starting at 4.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genes` | `list[str]` or `np.ndarray` | *required* | Ordered list of gene names |

##### `from_adata()`

```python
@classmethod
GeneVocab.from_adata(
    adata: AnnData,
    key: str = "var_names"
) -> GeneVocab
```

Build a vocabulary from an AnnData object.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData` | *required* | AnnData object |
| `key` | `str` | `"var_names"` | Which var attribute to use. `"var_names"` uses `adata.var_names`; any other string uses `adata.var[key]` |

##### `from_file()`

```python
@classmethod
GeneVocab.from_file(path: str | Path) -> GeneVocab
```

Load a vocabulary from a JSON file. The JSON can be either a list of gene names (indices assigned automatically) or a dict mapping gene names to indices.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | *required* | Path to JSON file |

**Raises:**
- `ValueError` — If JSON is not a list or dict.

#### Instance Methods

##### `encode()`

```python
encode(gene_names: list[str] | np.ndarray) -> np.ndarray
```

Convert gene names to vocabulary indices. Unknown genes are mapped to `UNK_TOKEN_ID` (1).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene_names` | `list[str]` or `np.ndarray` | *required* | Gene names to encode |

**Returns:** `np.ndarray` of shape `(len(gene_names),)` with dtype `int64`.

##### `decode()`

```python
decode(indices: list[int] | np.ndarray) -> list[str]
```

Convert indices back to gene names. Special token indices return their string names (e.g., `"<pad>"`). Unknown indices return `"<unk>"`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `indices` | `list[int]` or `np.ndarray` | *required* | Indices to decode |

**Returns:** `list[str]` of gene names.

##### `get_alignment_indices()`

```python
get_alignment_indices(
    gene_names: list[str] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]
```

Get index mapping to align a dataset's genes to this vocabulary. Returns two parallel arrays for efficient gene extraction and reordering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene_names` | `list[str]` or `np.ndarray` | *required* | Gene names from the source dataset |

**Returns:**
- `source_indices` — `np.ndarray` of indices into `gene_names` for genes that exist in the vocabulary
- `vocab_indices` — `np.ndarray` of corresponding vocabulary indices

##### `save()`

```python
save(path: str | Path) -> None
```

Save the vocabulary to a JSON file as a `{gene: index}` mapping.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | *required* | Output file path |

#### Properties

- `pad_token_id: int` — Returns 0
- `unk_token_id: int` — Returns 1
- `mask_token_id: int` — Returns 2
- `cls_token_id: int` — Returns 3
- `genes: list[str]` — List of all gene names in vocabulary order (excluding special tokens)

#### Special Methods

- `__len__()` — Total vocabulary size including special tokens (4 + number of genes)
- `__contains__(gene: str)` — Check if gene name is in vocabulary
- `__getitem__(gene: str)` — Get index for a gene name (raises `KeyError` if not found)

#### Example

```python
from scmodelforge.data import GeneVocab
import anndata as ad

# Load AnnData
adata = ad.read_h5ad("data/pbmc.h5ad")

# Build vocabulary from AnnData
vocab = GeneVocab.from_adata(adata)
print(vocab)  # GeneVocab(n_genes=2000, total_size=2004)

# Encode gene names
gene_names = ["CD3D", "CD8A", "UNKNOWN_GENE"]
indices = vocab.encode(gene_names)
print(indices)  # array([  4,   5,   1])  — UNKNOWN_GENE → UNK (1)

# Decode back
decoded = vocab.decode(indices)
print(decoded)  # ['CD3D', 'CD8A', '<unk>']

# Save for reuse
vocab.save("vocab.json")

# Load later
vocab2 = GeneVocab.from_file("vocab.json")
```

---

### `PreprocessingPipeline`

Configurable preprocessing pipeline applied to each cell on-the-fly. Supports library-size normalization and log1p transformation.

```python
from scmodelforge.data import PreprocessingPipeline
```

#### Constructor

```python
PreprocessingPipeline(
    normalize: str | None = "library_size",
    target_sum: float | None = 1e4,
    log1p: bool = True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `str` or `None` | `"library_size"` | Normalization method. `"library_size"` divides by total counts and scales to `target_sum`. `None` skips normalization. |
| `target_sum` | `float` or `None` | `1e4` | Target library size after normalization (commonly 10,000) |
| `log1p` | `bool` | `True` | Whether to apply `log(1 + x)` transformation after normalization |

**Raises:**
- `ValueError` — If `normalize` is not `"library_size"` or `None`.

#### Methods

##### `__call__()`

```python
__call__(expression: np.ndarray) -> np.ndarray
```

Apply preprocessing to a single cell's expression vector.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expression` | `np.ndarray` | *required* | Raw expression values, shape `(n_genes,)` |

**Returns:** `np.ndarray` of preprocessed expression values (same shape), dtype `float32`.

#### Example

```python
from scmodelforge.data import PreprocessingPipeline
import numpy as np

# Configure preprocessing
pipeline = PreprocessingPipeline(
    normalize="library_size",
    target_sum=1e4,
    log1p=True
)

# Raw counts for a single cell
raw_expr = np.array([0., 5., 10., 0., 20.])

# Preprocess
preprocessed = pipeline(raw_expr)
print(preprocessed)
# Normalized to 10,000 total, then log1p applied

# Skip normalization, only log1p
pipeline_log_only = PreprocessingPipeline(normalize=None, log1p=True)
log_expr = pipeline_log_only(raw_expr)
```

---

### `CellDataset`

Map-style PyTorch Dataset that yields individual cells from one or more AnnData objects. Each cell is returned as a dictionary containing expression values, gene vocabulary indices, and metadata.

```python
from scmodelforge.data import CellDataset
```

#### Constructor

```python
CellDataset(
    adata: AnnData | str | Path | list[AnnData | str | Path],
    gene_vocab: GeneVocab,
    preprocessing: PreprocessingPipeline | None = None,
    obs_keys: list[str] | None = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData`, `str`, `Path`, or list | *required* | AnnData object(s) or path(s) to `.h5ad` file(s). Multiple datasets are concatenated. |
| `gene_vocab` | `GeneVocab` | *required* | Gene vocabulary for index alignment |
| `preprocessing` | `PreprocessingPipeline` or `None` | `None` | Optional preprocessing pipeline applied on-the-fly |
| `obs_keys` | `list[str]` or `None` | `None` | Observation metadata keys to pass through (e.g., `["cell_type", "batch"]`) |

#### Methods

##### `__getitem__()`

```python
__getitem__(idx: int) -> dict[str, Any]
```

Get a single cell by index.

**Returns:** Dictionary with keys:
- `expression` — `torch.Tensor` of shape `(n_genes,)`, dtype `float32`, with preprocessed expression values for non-zero genes in vocabulary
- `gene_indices` — `torch.Tensor` of shape `(n_genes,)`, dtype `int64`, with vocabulary indices
- `n_genes` — `int`, number of non-zero genes (before padding)
- `metadata` — `dict[str, str]` with cell metadata from `obs_keys`

##### `__len__()`

```python
__len__() -> int
```

Total number of cells across all datasets.

#### Example

```python
from scmodelforge.data import GeneVocab, PreprocessingPipeline, CellDataset
import anndata as ad

# Load data
adata = ad.read_h5ad("data/pbmc.h5ad")

# Build vocabulary
vocab = GeneVocab.from_adata(adata)

# Configure preprocessing
preproc = PreprocessingPipeline(normalize="library_size", target_sum=1e4, log1p=True)

# Create dataset
dataset = CellDataset(
    adata=adata,
    gene_vocab=vocab,
    preprocessing=preproc,
    obs_keys=["cell_type", "batch"]
)

print(len(dataset))  # Number of cells

# Get first cell
cell = dataset[0]
print(cell["expression"].shape)  # (n_nonzero_genes,)
print(cell["gene_indices"].shape)  # (n_nonzero_genes,)
print(cell["metadata"])  # {'cell_type': 'B cell', 'batch': 'batch1'}

# Multiple datasets
dataset_multi = CellDataset(
    adata=["data/pbmc1.h5ad", "data/pbmc2.h5ad"],
    gene_vocab=vocab,
    preprocessing=preproc
)
```

---

### `CellDataLoader`

PyTorch DataLoader wrapper with single-cell-specific defaults. Handles variable-length gene sequences by padding to the batch maximum and building attention masks.

```python
from scmodelforge.data import CellDataLoader
```

#### Constructor

```python
CellDataLoader(
    dataset: CellDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    seed: int | None = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `CellDataset` | *required* | A CellDataset instance |
| `batch_size` | `int` | `64` | Number of cells per batch |
| `num_workers` | `int` | `0` | Number of data loading workers (0 = main process) |
| `shuffle` | `bool` | `True` | Whether to shuffle data each epoch |
| `drop_last` | `bool` | `True` | Whether to drop the last incomplete batch |
| `pin_memory` | `bool` | `True` | Whether to pin memory for faster GPU transfer (disabled if CUDA unavailable) |
| `seed` | `int` or `None` | `None` | Random seed for reproducible shuffling |

#### Methods

##### `__iter__()`

```python
__iter__() -> Iterator[dict[str, Any]]
```

Iterate over batches. Each batch is a dictionary with keys:
- `expression` — `torch.Tensor` of shape `(batch_size, max_seq_len)`, padded with 0s
- `gene_indices` — `torch.Tensor` of shape `(batch_size, max_seq_len)`, padded with 0s (PAD token)
- `attention_mask` — `torch.Tensor` of shape `(batch_size, max_seq_len)`, 1 for real tokens, 0 for padding
- `n_genes` — `torch.Tensor` of shape `(batch_size,)`, number of genes per cell (before padding)
- `metadata` — `list[dict]` of length `batch_size` with per-cell metadata

##### `__len__()`

```python
__len__() -> int
```

Number of batches per epoch.

#### Example

```python
from scmodelforge.data import GeneVocab, PreprocessingPipeline, CellDataset, CellDataLoader
import anndata as ad

# Build dataset
adata = ad.read_h5ad("data/pbmc.h5ad")
vocab = GeneVocab.from_adata(adata)
preproc = PreprocessingPipeline()
dataset = CellDataset(adata, vocab, preproc)

# Create dataloader
loader = CellDataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    seed=42
)

print(len(loader))  # Number of batches

# Iterate over batches
for batch in loader:
    print(batch["expression"].shape)  # (32, max_len_in_batch)
    print(batch["attention_mask"].shape)  # (32, max_len_in_batch)
    break
```

---

### `AnnDataStore`

Manages one or more AnnData objects with automatic gene alignment to a shared vocabulary. Provides a unified interface to access cells by global index across multiple datasets. Used internally by `CellDataset`.

```python
from scmodelforge.data.anndata_store import AnnDataStore
```

#### Constructor

```python
AnnDataStore(
    adatas: list[AnnData | str | Path],
    gene_vocab: GeneVocab,
    obs_keys: list[str] | None = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adatas` | `list` | *required* | List of AnnData objects or file paths to `.h5ad` files |
| `gene_vocab` | `GeneVocab` | *required* | Shared gene vocabulary for alignment |
| `obs_keys` | `list[str]` or `None` | `None` | Observation metadata keys to extract. Keys not present in a dataset are filled with `"unknown"` |

#### Methods

##### `get_cell()`

```python
get_cell(global_idx: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]
```

Retrieve a single cell by global index (across all datasets). Returns only non-zero expressed genes that exist in the vocabulary.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_idx` | `int` | *required* | Global cell index (0 to `len(store) - 1`) |

**Returns:**
- `expression` — `np.ndarray` of non-zero expression values, dtype `float32`
- `gene_indices` — `np.ndarray` of vocabulary indices (parallel to expression), dtype `int64`
- `metadata` — `dict[str, str]` with cell metadata from `obs_keys`

**Raises:**
- `IndexError` — If `global_idx` is out of range.

##### `__len__()`

```python
__len__() -> int
```

Total number of cells across all datasets.

#### Properties

- `n_datasets: int` — Number of loaded datasets

#### Example

```python
from scmodelforge.data import GeneVocab
from scmodelforge.data.anndata_store import AnnDataStore
import anndata as ad

# Load datasets
adata1 = ad.read_h5ad("data/pbmc1.h5ad")
adata2 = ad.read_h5ad("data/pbmc2.h5ad")

# Build vocab
vocab = GeneVocab.from_adata(adata1)

# Create store
store = AnnDataStore(
    adatas=[adata1, adata2],
    gene_vocab=vocab,
    obs_keys=["cell_type"]
)

print(len(store))  # Total cells across both datasets
print(store.n_datasets)  # 2

# Get a cell
expr, genes, meta = store.get_cell(0)
print(expr.shape)  # (n_nonzero_genes,)
print(meta)  # {'cell_type': 'T cell'}
```

---

### `build_obs_value_filter()`

Convert a structured filter dictionary to a SOMA `obs_value_filter` string for CELLxGENE Census queries.

```python
from scmodelforge.data import build_obs_value_filter
```

#### Function Signature

```python
build_obs_value_filter(filters: dict[str, Any]) -> str | None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filters` | `dict[str, Any]` | *required* | Mapping of column names to values. Supported value types: `str` (equality), `list` (membership), `bool`, `int`, `float` |

**Returns:** SOMA-compatible filter string, or `None` if `filters` is empty.

#### Supported Filter Types

- **String equality**: `{"tissue": "brain"}` → `"tissue == 'brain'"`
- **List membership**: `{"tissue": ["brain", "lung"]}` → `"tissue in ['brain', 'lung']"`
- **Boolean**: `{"is_primary_data": True}` → `"is_primary_data == True"`
- **Numeric**: `{"n_genes": 500}` → `"n_genes == 500"`

Multiple filters are combined with `and`.

#### Example

```python
from scmodelforge.data import build_obs_value_filter

# Single filter
filter_str = build_obs_value_filter({"tissue": "brain"})
print(filter_str)  # "tissue == 'brain'"

# Multiple filters
filter_str = build_obs_value_filter({
    "tissue": ["brain", "lung"],
    "is_primary_data": True,
    "cell_type": "neuron"
})
print(filter_str)
# "tissue in ['brain', 'lung'] and is_primary_data == True and cell_type == 'neuron'"

# Empty filters
filter_str = build_obs_value_filter({})
print(filter_str)  # None
```

---

### `load_census_adata()`

Load an AnnData object from the CZ CELLxGENE Census. Requires the optional `cellxgene-census` dependency (`pip install scModelForge[census]`).

```python
from scmodelforge.data import load_census_adata
```

#### Function Signature

```python
load_census_adata(
    census_config: CensusConfig,
    obs_keys: list[str] | None = None
) -> AnnData
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `census_config` | `CensusConfig` | *required* | Census configuration (organism, version, filters, etc.) |
| `obs_keys` | `list[str]` or `None` | `None` | Extra `obs` column names to include (merged with `census_config.obs_columns`) |

**Returns:** `anndata.AnnData` loaded from Census.

**Raises:**
- `ImportError` — If `cellxgene-census` is not installed.

#### Example

```python
from scmodelforge.data import load_census_adata
from scmodelforge.config.schema import CensusConfig

# Configure Census query
census_cfg = CensusConfig(
    organism="Homo sapiens",
    census_version="stable",
    filters={"tissue": "brain", "is_primary_data": True},
    obs_columns=["cell_type", "tissue"]
)

# Load data
adata = load_census_adata(census_cfg)
print(adata.n_obs)  # Number of cells matching filters
print(adata.obs.columns)  # ['cell_type', 'tissue', ...]
```

---

### `load_adata()`

Unified dispatcher for loading AnnData from configured sources (local files or Census).

```python
from scmodelforge.data._utils import load_adata
```

#### Function Signature

```python
load_adata(
    data_config: DataConfig,
    adata: AnnData | None = None,
    obs_keys: list[str] | None = None
) -> AnnData
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_config` | `DataConfig` | *required* | Data configuration (source, paths, census settings, etc.) |
| `adata` | `AnnData` or `None` | `None` | Optional pre-loaded AnnData. If provided, returned directly (useful for testing). |
| `obs_keys` | `list[str]` or `None` | `None` | Extra `obs` column names (forwarded to Census loading) |

**Returns:** `anndata.AnnData`

**Raises:**
- `ValueError` — If `data_config.source` is not `"local"` or `"cellxgene_census"`.

#### Example

```python
from scmodelforge.data._utils import load_adata
from scmodelforge.config.schema import DataConfig, CensusConfig

# Local source
local_config = DataConfig(
    source="local",
    paths=["data/pbmc1.h5ad", "data/pbmc2.h5ad"]
)
adata = load_adata(local_config)

# Census source
census_config = DataConfig(
    source="cellxgene_census",
    census=CensusConfig(organism="Homo sapiens", filters={"tissue": "brain"})
)
adata = load_adata(census_config)
```

---

### `select_highly_variable_genes()`

Simple highly variable gene (HVG) selection based on dispersion (Fano factor: variance/mean). For more sophisticated methods, use `scanpy.pp.highly_variable_genes` at the AnnData level before constructing the dataset.

```python
from scmodelforge.data.preprocessing import select_highly_variable_genes
```

#### Function Signature

```python
select_highly_variable_genes(
    expressions: np.ndarray,
    n_top_genes: int
) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expressions` | `np.ndarray` | *required* | Expression matrix of shape `(n_cells, n_genes)` |
| `n_top_genes` | `int` | *required* | Number of genes to select |

**Returns:** `np.ndarray` boolean mask of shape `(n_genes,)` indicating selected genes.

#### Example

```python
from scmodelforge.data.preprocessing import select_highly_variable_genes
import anndata as ad

adata = ad.read_h5ad("data/pbmc.h5ad")
expr_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

# Select top 2000 HVGs
hvg_mask = select_highly_variable_genes(expr_matrix, n_top_genes=2000)
print(hvg_mask.sum())  # 2000

# Filter AnnData
adata_hvg = adata[:, hvg_mask].copy()
```

---

### `collate_cells()`

Custom collate function for variable-length cell data. Pads expression and gene_indices tensors to the maximum sequence length in the batch and builds an attention mask.

```python
from scmodelforge.data._utils import collate_cells
```

#### Function Signature

```python
collate_cells(
    batch: list[dict[str, Any]],
    pad_value: int = 0
) -> dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch` | `list[dict]` | *required* | List of dicts from `CellDataset.__getitem__`, each with keys `expression`, `gene_indices`, `n_genes`, `metadata` |
| `pad_value` | `int` | `0` | Value used to pad shorter sequences (default 0 = PAD token) |

**Returns:** Dictionary with batched tensors:
- `expression` — `torch.Tensor` of shape `(batch_size, max_len)`
- `gene_indices` — `torch.Tensor` of shape `(batch_size, max_len)`
- `attention_mask` — `torch.Tensor` of shape `(batch_size, max_len)`, 1 for real tokens, 0 for padding
- `n_genes` — `torch.Tensor` of shape `(batch_size,)`
- `metadata` — `list[dict]` of length `batch_size`

Note: This function is used automatically by `CellDataLoader` and typically does not need to be called directly.

---

## See Also

- **Tokenizers** — `scmodelforge.tokenizers` module for converting cells to model inputs
- **Training** — `scmodelforge.training.data_module.CellDataModule` for PyTorch Lightning integration
- **Configuration** — `scmodelforge.config.schema.DataConfig` for YAML-based data configuration
