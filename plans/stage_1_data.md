# Stage 1: scModelForge.data

## Overview

The data module handles the entire pipeline from raw biological data (AnnData/.h5ad) to GPU-ready batches. This is the highest-value component because every single-cell foundation model project currently builds this from scratch.

**Core responsibility:** AnnData in → PyTorch tensors out, efficiently and reproducibly.

**Dependencies:** Stage 0 (scaffolding must be in place)
**Blocks:** Stage 4 (training), Stage 5 (eval) — both need data loading

---

## Phase 1: Foundation (Months 1–3)

### Goals
- Stream cells from local `.h5ad` files into PyTorch DataLoader batches
- Support multi-worker loading with automatic sharding
- Build and manage gene vocabularies across datasets
- Configurable preprocessing (normalisation, HVG selection)
- Achieve >50k cells/sec throughput on single GPU

### Architecture

```
.h5ad file(s)
     │
     ▼
┌─────────────┐
│ DataRegistry │  ← knows about all data sources, resolves paths
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ AnnDataStore │  ← lazy-loads .h5ad, handles sparse/dense, gene alignment
└──────┬───────┘
       │
       ▼
┌───────────────────┐
│ CellDataset       │  ← PyTorch Dataset (map or iterable style)
│  - preprocessing  │     on-the-fly normalisation, HVG filtering
│  - gene alignment │     align to shared vocabulary
└──────┬────────────┘
       │
       ▼
┌──────────────────┐
│ CellDataLoader   │  ← wraps DataLoader with smart defaults
│  - sharding      │     multi-worker, multi-node aware
│  - collation     │     sparse-aware batching
└──────────────────┘
```

### File Structure

```
src/scmodelforge/data/
├── __init__.py           # Public API: CellDataset, CellDataLoader, GeneVocab
├── dataset.py            # CellDataset implementation
├── dataloader.py         # CellDataLoader wrapper
├── preprocessing.py      # Normalisation, HVG selection, transforms
├── gene_vocab.py         # Gene vocabulary management
├── anndata_store.py      # Lazy AnnData loading and indexing
└── _utils.py             # Sparse tensor conversion, batching helpers
```

### Key Classes and Interfaces

#### `GeneVocab`

Manages the mapping between gene identifiers and integer indices.

```python
class GeneVocab:
    """Unified gene vocabulary across datasets."""

    @classmethod
    def from_adata(cls, adata: AnnData, key: str = "var_names") -> "GeneVocab":
        """Build vocabulary from an AnnData object."""

    @classmethod
    def from_file(cls, path: str | Path) -> "GeneVocab":
        """Load vocabulary from a JSON/TSV file."""

    @classmethod
    def human_protein_coding(cls) -> "GeneVocab":
        """Load the default human protein-coding gene vocabulary."""

    def encode(self, gene_names: list[str]) -> np.ndarray:
        """Convert gene names to indices. Unknown genes map to <unk>."""

    def __len__(self) -> int: ...
    def __contains__(self, gene: str) -> bool: ...
    def save(self, path: str | Path) -> None: ...
```

**Special tokens:** `<pad>`, `<unk>`, `<mask>`, `<cls>` — stored at indices 0–3.

#### `CellDataset`

```python
class CellDataset(torch.utils.data.Dataset):
    """Dataset that yields individual cells from AnnData."""

    def __init__(
        self,
        adata: AnnData | str | Path | list[str | Path],
        gene_vocab: GeneVocab,
        preprocessing: PreprocessingPipeline | None = None,
        max_genes: int | None = None,
    ): ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns dict with keys: expression, gene_indices, metadata."""
        ...
```

**Return format:**
```python
{
    "expression": torch.Tensor,       # (n_genes,) — expression values
    "gene_indices": torch.Tensor,     # (n_genes,) — indices into gene_vocab
    "n_genes": int,                   # number of expressed genes (before padding)
    "metadata": {                     # cell-level metadata (cell_type, batch, etc.)
        "cell_type": str,
        "batch": str,
        ...
    },
}
```

#### `CellDataLoader`

```python
class CellDataLoader:
    """Wraps PyTorch DataLoader with single-cell-specific defaults."""

    def __init__(
        self,
        dataset: CellDataset,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = True,
    ): ...

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]: ...
    def __len__(self) -> int: ...
```

**Collation:** Custom collate function that handles variable-length gene lists by padding to the batch maximum (or a configured `max_genes`).

#### `PreprocessingPipeline`

```python
class PreprocessingPipeline:
    """Configurable preprocessing applied on-the-fly or cached."""

    def __init__(
        self,
        normalize: str | None = "library_size",  # "library_size", "log1p", "scran"
        target_sum: float | None = 1e4,
        hvg_selection: int | None = None,
        log1p: bool = True,
    ): ...

    def __call__(self, expression: np.ndarray, gene_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply preprocessing. Returns (processed_expression, filtered_gene_names)."""
        ...
```

### Design Decisions (Phase 1)

1. **Map-style Dataset for Phase 1:** Random access via `__getitem__`. Simpler to implement and debug. Iterable-style streaming added in Phase 2 for Census.
2. **Sparse throughout:** Keep data sparse as long as possible. Convert to dense only at collation time within each batch.
3. **Gene alignment at dataset init:** When loading multiple `.h5ad` files, align all to the shared `GeneVocab` at construction time. Store an index mapping per file.
4. **Preprocessing can be on-the-fly or cached:** On-the-fly is default. For large datasets, provide a `preprocess_and_cache()` utility that writes a processed `.h5ad`.
5. **Metadata is optional pass-through:** Cell metadata (cell_type, batch) is carried through for eval but not required for training.

### Config Integration

```yaml
data:
  source: local
  paths:
    - ./data/dataset1.h5ad
    - ./data/dataset2.h5ad
  gene_vocab: human_protein_coding  # or path to custom vocab file
  preprocessing:
    normalize: library_size
    target_sum: 10000
    hvg_selection: 2000
    log1p: true
  max_genes: 2048
  num_workers: 4
```

### Performance Targets

| Metric | Target | How to measure |
|---|---|---|
| Throughput (single GPU) | >50k cells/sec | Time to iterate full dataset |
| Memory (100k cells) | <4 GB RAM | Monitor RSS during loading |
| Multi-worker scaling | Near-linear to 8 workers | Throughput vs. num_workers |

### Tests (Phase 1)

- `test_gene_vocab.py`: Build vocab from AnnData, save/load, encode/decode, special tokens, unknown gene handling.
- `test_preprocessing.py`: Library size normalisation correctness, HVG selection, log1p, pipeline composition.
- `test_dataset.py`: Load single/multiple .h5ad files, gene alignment, correct shapes, sparse handling.
- `test_dataloader.py`: Batching, padding, multi-worker correctness, shuffle reproducibility with seed.
- `test_throughput.py` (marked `@pytest.mark.slow`): Benchmark throughput on a 100k cell synthetic dataset.

---

## Phase 2: Breadth (Months 4–6)

### CELLxGENE Census Integration

- Add `CensusDataset` that wraps `cellxgene_census` API.
- Supports streaming from Census without downloading full dataset.
- Filter by organism, tissue, assay, disease, etc.
- Implements the same `CellDataset` interface so it's interchangeable.

```yaml
data:
  source: cellxgene_census
  organism: homo_sapiens
  filters:
    tissue: [lung, heart, liver]
    is_primary_data: true
    assay: ["10x 3' v3"]
  census_version: "2024-07-01"  # Pin version for reproducibility
```

### Iterable-style Streaming

- Add `StreamingCellDataset(IterableDataset)` for datasets too large for random access.
- Supports reading from S3/GCS via `fsspec`.
- Automatic sharding across DDP workers and nodes.

### Cached Preprocessing

- `scmodelforge preprocess` CLI command that preprocesses and writes a new `.h5ad`.
- Stores preprocessing config in `.uns` for reproducibility.

### Batch-aware Sampling

- `BatchAwareSampler` that balances batches within each training mini-batch.
- Configurable batch key (e.g., `obs["batch"]`, `obs["donor"]`).
- Important for models that learn batch correction.

---

## Phase 3: Community & Scale (Months 7–12)

### Multi-species Support

- Unified gene vocabulary mapping between human and mouse orthologs.
- Use Ensembl BioMart ortholog tables for mapping.
- `GeneVocab.multi_species(organisms=["human", "mouse"])`.

### pertpy Integration

- Use pertpy for perturbation-specific preprocessing.
- Automatic detection and parsing of perturbation metadata (guide RNA, compound, dose).
- `PerturbationDataset` subclass with perturbation-aware collation.

### Large-scale Optimization

- Memory-mapped data loading for datasets >100M cells.
- Pre-sharded on-disk format (directory of `.h5ad` chunks) for parallel ingest.
- Integration with PyTorch distributed samplers for FSDP.

---

## Checklist

### Phase 1
- [ ] Implement `GeneVocab` with special tokens, save/load, human protein-coding default
- [ ] Implement `PreprocessingPipeline` (library_size, log1p, HVG)
- [ ] Implement `AnnDataStore` for lazy .h5ad loading
- [ ] Implement `CellDataset` with gene alignment and preprocessing
- [ ] Implement `CellDataLoader` with custom collation and padding
- [ ] Implement sparse-to-dense conversion utilities
- [ ] Add config parsing for `data:` section
- [ ] Write comprehensive tests for all components
- [ ] Benchmark throughput and verify >50k cells/sec target
- [ ] Write docstrings and API documentation

### Phase 2
- [ ] Implement `CensusDataset` wrapping cellxgene_census
- [ ] Implement `StreamingCellDataset` (IterableDataset)
- [ ] Add S3/GCS support via fsspec
- [ ] Implement `BatchAwareSampler`
- [ ] Add `scmodelforge preprocess` CLI command

### Phase 3
- [ ] Multi-species gene vocabulary with ortholog mapping
- [ ] pertpy integration for perturbation data
- [ ] Memory-mapped and pre-sharded data formats
