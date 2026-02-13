# Large-scale Data Handling

This tutorial covers techniques for working with datasets that don't fit in memory: sharding, streaming, cloud storage, and distributed sampling.

## Introduction

Single-cell atlases now contain millions to billions of cells. The Human Cell Atlas, Tabula Sapiens, and CELLxGENE census datasets can easily exceed available RAM on most machines. Loading everything into memory is impractical for these large-scale datasets.

scModelForge provides several strategies for large-scale training:

- **Memory-mapped sharding**: Convert datasets into memory-efficient shard directories
- **Streaming I/O**: Read H5AD files chunk-by-chunk without loading the full dataset
- **Cloud storage**: Access data directly from S3, GCS, or Azure Blob Storage
- **Distributed sampling**: Efficiently distribute data across multiple GPUs
- **Weighted sampling**: Handle class imbalance in large datasets
- **Gene selection**: Reduce memory by selecting gene subsets per batch

## When Do You Need This?

Use this guide to determine the right approach for your dataset size:

- **< 500k cells**: Standard in-memory loading is fine. Use the basic `CellDataset` and `CellDataModule`.
- **500k - 5M cells**: Consider sharding or streaming to reduce memory pressure.
- **5M+ cells**: Use sharding combined with distributed training (multi-GPU) for best performance.

Additional considerations:

- **Memory available**: If you have sufficient RAM (e.g., 256GB+), you may handle larger datasets in-memory.
- **Training epochs**: Multi-epoch training benefits from sharding (random access). Single-pass training can use streaming.
- **Cloud vs local**: Cloud data requires download caching or streaming from remote storage.

## Sharding

Sharding converts large H5AD files into a directory of memory-mapped shards. Each shard is stored as a numpy memmap file with parquet metadata, allowing lazy loading of only the cells needed for each batch.

### Converting to Shards

Use the CLI to convert an H5AD file into shards:

```bash
scmodelforge shard \
  --config my_config.yaml \
  --output-dir ./shards/ \
  --shard-size 100000
```

This command reads your dataset and splits it into shards of 100,000 cells each. The shard directory structure:

```
shards/
├── shard_000000.npy       # Memory-mapped expression data
├── shard_000000.parquet   # Cell metadata
├── shard_000001.npy
├── shard_000001.parquet
├── ...
└── shard_metadata.json    # Overall dataset info
```

The `--config` flag should point to a YAML config containing your `GeneVocab` and data preprocessing settings. The gene vocabulary ensures consistent gene ordering across shards.

### Training with Shards

Configure your training pipeline to use sharded data:

```yaml
data:
  source: local
  shards:
    enabled: true
    shard_dir: ./shards/
  preprocessing:
    normalize: true
    log_transform: true
```

When `shards.enabled` is true, the training pipeline uses `ShardedCellDataset` backed by `MemoryMappedStore`, which loads cells on-demand from the shard files.

### Python API for Sharding

You can also work with shards programmatically:

```python
from scmodelforge.data.sharding import convert_to_shards, validate_shard_dir
from scmodelforge.data.memmap_store import MemoryMappedStore
from scmodelforge.data.dataset import ShardedCellDataset
from scmodelforge.data.vocab import GeneVocab

# Convert H5AD to shards
vocab = GeneVocab.from_adata("large_atlas.h5ad", n_genes=5000)
convert_to_shards(
    adata_path="large_atlas.h5ad",
    output_dir="./shards/",
    gene_vocab=vocab,
    shard_size=100000,
)

# Validate shard directory
validate_shard_dir("./shards/")

# Create dataset from shards
store = MemoryMappedStore(shard_dir="./shards/")
dataset = ShardedCellDataset(store=store, gene_vocab=vocab)

# Access cells (loads only requested cell from memmap)
cell = dataset[0]
```

### Shard Size Selection

Choosing the right shard size balances memory usage and I/O efficiency:

- **100k-500k cells per shard**: Good default range for most workloads
- **Smaller shards (50k-100k)**: Lower memory footprint, more I/O operations
- **Larger shards (500k-1M)**: Fewer files, higher memory per shard

## Streaming

Streaming reads H5AD files chunk-by-chunk using HDF5 backed mode, avoiding loading the entire dataset into memory. This is useful for single-pass training or when you don't want to preprocess shards.

### Enabling Streaming

Configure streaming in your training YAML:

```yaml
data:
  source: local
  paths:
    - ./data/large_atlas.h5ad
  streaming: true
  streaming_chunk_size: 10000
  streaming_shuffle_buffer: 10000
  preprocessing:
    normalize: true
    log_transform: true
```

Key parameters:

- `streaming_chunk_size`: Number of cells to read per chunk. Smaller values reduce memory, larger values improve throughput.
- `streaming_shuffle_buffer`: Buffer size for shuffling. Acts as a sliding window for randomization.

### How Streaming Works

The `StreamingCellDataset` reads the H5AD file in backed mode and yields cells sequentially:

1. Opens H5AD with `anndata.read_h5ad(backed='r')`
2. Reads chunks of `streaming_chunk_size` cells
3. Fills a shuffle buffer and yields randomized cells
4. Continues until the entire dataset is consumed

Note that streaming datasets are `IterableDataset` instances, not map-style datasets. You cannot use random indexing (`dataset[42]`), only sequential iteration.

### Streaming vs Sharding

| Feature | Streaming | Sharding |
|---------|-----------|----------|
| Memory usage | Very low | Low |
| Random access | No | Yes |
| Multi-epoch efficiency | Lower (re-reads file) | Higher (memmapped) |
| Setup time | None | Requires conversion |
| Use case | Single-pass, exploration | Multi-epoch training |

### Python API for Streaming

```python
from scmodelforge.data.streaming import StreamingCellDataset
from scmodelforge.data.vocab import GeneVocab

vocab = GeneVocab.from_adata("large_atlas.h5ad", n_genes=5000)
dataset = StreamingCellDataset(
    h5ad_path="large_atlas.h5ad",
    gene_vocab=vocab,
    chunk_size=10000,
    shuffle_buffer_size=10000,
)

# Streaming datasets are iterable
for cell in dataset:
    print(cell.gene_ids.shape)
    break
```

## Cloud Storage

scModelForge supports reading H5AD files directly from cloud object storage: S3, Google Cloud Storage, and Azure Blob Storage. Files are cached locally on first access for performance.

### Cloud Storage Configuration

Install cloud dependencies:

```bash
pip install "scModelForge[cloud]"
```

Configure cloud data paths in your YAML:

```yaml
data:
  source: local
  paths:
    - s3://my-bucket/atlas/large_dataset.h5ad
    - gs://another-bucket/dataset2.h5ad
  cloud:
    cache_dir: /tmp/scmodelforge_cache
  preprocessing:
    normalize: true
    log_transform: true
```

Supported URL schemes:

- `s3://` - Amazon S3
- `gs://` or `gcs://` - Google Cloud Storage
- `az://` or `abfs://` - Azure Blob Storage

### How Cloud Caching Works

When you specify a cloud path:

1. scModelForge checks if the file exists in the local cache (hash-based lookup)
2. If not cached, downloads the file to `cache_dir`
3. Subsequent runs reuse the cached file (no re-download)

This approach avoids anndata version compatibility issues with `storage_options` and ensures consistent performance after the first download.

### Cloud Credentials

Configure cloud credentials using standard methods:

**AWS S3:**
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

**Google Cloud Storage:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**Azure Blob Storage:**
```bash
export AZURE_STORAGE_ACCOUNT_NAME=your_account
export AZURE_STORAGE_ACCOUNT_KEY=your_key
```

### Combining Cloud with Sharding

You can shard cloud-hosted datasets:

```bash
scmodelforge shard \
  --config my_config.yaml \
  --output-dir ./local_shards/ \
  --shard-size 100000
```

And reference the cloud path in your config:

```yaml
data:
  source: local
  paths:
    - s3://my-bucket/atlas/large_dataset.h5ad
  cloud:
    cache_dir: /tmp/scmodelforge_cache
```

The shard command downloads the file to cache, then converts it to local shards.

## Weighted Sampling

Large single-cell datasets often have severe class imbalance. For example, a dataset might contain 90% T cells and only 1% rare cell types. Weighted sampling helps ensure all cell types are represented in training.

### Enabling Weighted Sampling

Configure weighted sampling in your training config:

```yaml
training:
  sampling:
    strategy: weighted
    label_key: cell_type
    curriculum_warmup_epochs: 5
  batch_size: 128
  num_epochs: 50
```

Parameters:

- `strategy: weighted` - Use inverse-frequency weighting based on cell type counts
- `label_key` - Metadata key containing cell type labels (must exist in `adata.obs`)
- `curriculum_warmup_epochs` - Number of epochs to gradually transition from uniform to weighted sampling

### How Weighted Sampling Works

The `WeightedCellSampler`:

1. Counts occurrences of each cell type in the dataset
2. Computes inverse-frequency weights: rare types get higher sampling probability
3. Samples cells proportional to weights during training

Curriculum warmup gradually ramps up weighting:

- Epoch 0-1: Mostly uniform sampling
- Epoch 1-5: Linearly increase weight influence
- Epoch 5+: Full weighted sampling

This prevents the model from overfitting to rare classes early in training.

### When to Use Weighted Sampling

Use weighted sampling when:

- Cell type distribution is highly imbalanced (e.g., some types < 5% of data)
- You want balanced representation across all cell types
- Fine-tuning for classification tasks

Avoid weighted sampling when:

- Dataset is already balanced
- You want the model to learn the natural distribution
- Training on unlabeled data (no `label_key` available)

## Gene Selection

Gene selection reduces memory by limiting the number of genes processed per batch. Instead of using all genes in the vocabulary, you select a subset based on expression levels or random sampling.

### Enabling Gene Selection

Configure gene selection in your data config:

```yaml
data:
  source: local
  paths:
    - ./data/large_atlas.h5ad
  gene_selection:
    strategy: most_expressed
    n_genes: 2048
  preprocessing:
    normalize: true
    log_transform: true
```

Parameters:

- `strategy`: Selection strategy (see below)
- `n_genes`: Number of genes to select per batch

### Selection Strategies

**`all` (default)**: Use all genes in the vocabulary. No selection.

**`most_expressed`**: Select the top-k genes by total expression across the batch.

```yaml
data:
  gene_selection:
    strategy: most_expressed
    n_genes: 2048
```

This selects the 2048 most highly expressed genes for each batch, dynamically adapting to batch content.

**`random_expressed`**: Randomly sample k expressed genes from the batch.

```yaml
data:
  gene_selection:
    strategy: random_expressed
    n_genes: 2048
```

Provides more diversity than `most_expressed` but may include lowly expressed genes.

### Memory Savings

Gene selection reduces:

- **Embedding table size**: Fewer genes to embed per forward pass
- **Attention memory**: Smaller sequence length for transformer layers
- **Gradient memory**: Fewer parameters to backpropagate through

Example memory savings for a batch of 128 cells:

- Without selection (5000 genes): ~640k tokens
- With selection (2048 genes): ~262k tokens (~60% reduction)

## Distributed Training with Shards

When training with multiple GPUs using DDP or FSDP, the `DistributedShardSampler` assigns whole shards to different ranks, minimizing data transfer and improving efficiency.

### How It Works

Each GPU rank gets a disjoint subset of shards:

- Rank 0: shards 0, 3, 6, 9, ...
- Rank 1: shards 1, 4, 7, 10, ...
- Rank 2: shards 2, 5, 8, 11, ...

This ensures:

- No duplicate data across ranks (required for correct DDP gradients)
- Each rank reads from its own shards (reduced I/O contention)
- Balanced workload (shards are roughly equal size)

### Configuration

The distributed shard sampler is automatically used when you combine sharded data with multi-GPU training:

```yaml
data:
  shards:
    enabled: true
    shard_dir: ./shards/

training:
  num_devices: 4  # 4 GPUs
  strategy: ddp   # or fsdp
```

The `CellDataModule` detects distributed training and automatically configures the sampler.

### Python API

```python
from scmodelforge.data.distributed import DistributedShardSampler
from scmodelforge.data.dataset import ShardedCellDataset

dataset = ShardedCellDataset(store=store, gene_vocab=vocab)
sampler = DistributedShardSampler(
    dataset=dataset,
    num_replicas=4,  # Total number of GPUs
    rank=0,          # Current GPU rank
    shuffle=True,
)

dataloader = DataLoader(dataset, batch_size=128, sampler=sampler)
```

## Offline Preprocessing

For large datasets, it's often more efficient to preprocess once and reuse the processed data across experiments. This saves time and ensures consistency.

### Preprocessing Command

```bash
scmodelforge preprocess \
  --input raw_atlas.h5ad \
  --output processed_atlas.h5ad \
  --hvg 4000
```

This command:

1. Loads the raw H5AD file
2. Normalizes counts to 10,000 per cell
3. Log-transforms expression (log1p)
4. Selects 4000 highly variable genes
5. Saves the processed data to a new H5AD file

### Preprocessing from Config

You can also use a YAML config for more control:

```yaml
# preprocess_config.yaml
data:
  preprocessing:
    normalize: true
    log_transform: true
    hvg_flavor: seurat
```

```bash
scmodelforge preprocess \
  --input raw_atlas.h5ad \
  --config preprocess_config.yaml \
  --output processed_atlas.h5ad \
  --hvg 4000
```

### Cloud-aware Preprocessing

Preprocessing works with cloud paths:

```bash
scmodelforge preprocess \
  --input s3://my-bucket/raw_atlas.h5ad \
  --output ./processed_atlas.h5ad \
  --hvg 4000
```

The tool downloads from S3, preprocesses locally, and saves the result.

### When to Preprocess Offline

Use offline preprocessing when:

- Running multiple experiments on the same dataset
- Preprocessing is computationally expensive (e.g., batch correction)
- You want to version preprocessed data separately

Skip offline preprocessing when:

- Experimenting with different preprocessing strategies
- Using streaming (preprocessing happens on-the-fly)

## Performance Tips

### Data Loading

**Use multiple workers:**

```yaml
training:
  num_workers: 8
  batch_size: 128
```

Start with `num_workers = 4` and increase until you saturate CPU or I/O. Too many workers can cause overhead.

**Pin memory for GPU transfers:**

```yaml
training:
  pin_memory: true
```

This speeds up data transfer from CPU to GPU memory.

### Mixed Precision Training

Reduce memory usage with bfloat16 mixed precision:

```yaml
training:
  precision: bf16-mixed
```

This uses bfloat16 for forward/backward passes and float32 for parameter updates, reducing memory by ~50% with minimal accuracy loss.

### Shard Size Tuning

Experiment with shard sizes:

- **100k-200k cells**: Good for datasets with many rare cell types
- **300k-500k cells**: Balanced for most use cases
- **500k-1M cells**: Reduces number of files, higher memory per shard

Monitor memory usage and adjust accordingly.

### Cloud Data Caching

Cache cloud data locally for repeated experiments:

```yaml
data:
  cloud:
    cache_dir: /mnt/fast-ssd/scmodelforge_cache
```

Use a fast local SSD for the cache directory. The first run downloads the data; subsequent runs are as fast as local files.

### Streaming Chunk Size

Balance memory and throughput:

- **5k-10k cells**: Low memory, suitable for very large files
- **10k-20k cells**: Balanced default
- **20k-50k cells**: Higher throughput if memory allows

### Gradient Accumulation

If your batch size is limited by memory, use gradient accumulation:

```yaml
training:
  batch_size: 32
  accumulate_grad_batches: 4  # Effective batch size: 128
```

This simulates a batch size of 128 while only loading 32 cells at a time.

## Complete Example: Training on a 10M Cell Atlas

Here's a complete workflow for training on a large dataset:

**Step 1: Download and preprocess**

```bash
# Download from cloud and preprocess
scmodelforge preprocess \
  --input s3://cellxgene-data/atlas_10M.h5ad \
  --output ./atlas_10M_processed.h5ad \
  --hvg 5000
```

**Step 2: Convert to shards**

```bash
# Create shard directory
scmodelforge shard \
  --config base_config.yaml \
  --output-dir ./atlas_shards/ \
  --shard-size 200000
```

**Step 3: Configure training**

```yaml
# train_large_atlas.yaml
data:
  source: local
  shards:
    enabled: true
    shard_dir: ./atlas_shards/
  gene_selection:
    strategy: most_expressed
    n_genes: 2048

training:
  batch_size: 256
  num_epochs: 10
  num_workers: 8
  precision: bf16-mixed
  pin_memory: true
  accumulate_grad_batches: 2
  sampling:
    strategy: weighted
    label_key: cell_type
    curriculum_warmup_epochs: 3
  fsdp:
    enabled: true
    sharding_strategy: FULL_SHARD
    cpu_offload: false

model:
  architecture: transformer_encoder
  d_model: 512
  n_layers: 8
  n_heads: 8

tokenizer:
  strategy: rank_value
  max_length: 2048
```

**Step 4: Train on multiple GPUs**

```bash
scmodelforge train \
  --config train_large_atlas.yaml \
  --num-devices 4
```

This configuration:

- Uses sharding for efficient memory-mapped access
- Selects top 2048 genes per batch to reduce memory
- Trains with 256 cells per GPU, accumulated over 2 steps (effective batch size: 2048)
- Uses bfloat16 mixed precision
- Distributes across 4 GPUs with FSDP
- Balances cell types with weighted sampling

## What's Next

- **Distributed Training Tutorial**: Learn how to scale training across multiple nodes with FSDP.
- **Pretraining Tutorial**: Complete guide to pretraining foundation models on large atlases.
- **Fine-tuning Tutorial**: Adapt pretrained models to downstream tasks with limited data.
- **Cloud Deployment**: Deploy trained models as APIs using cloud infrastructure.

For more details on configuration options, see the Configuration Reference in the documentation.
