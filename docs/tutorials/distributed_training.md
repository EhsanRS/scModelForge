# Distributed Training with FSDP

Learn how to scale scModelForge training across multiple GPUs using PyTorch Lightning's distributed training strategies.

## Why Distributed Training?

Foundation models benefit from training on large datasets, but processing millions of cells on a single GPU is slow and memory-constrained. Distributed training splits the work across multiple GPUs, enabling:

- **Faster training**: Process more cells per second by parallelizing data loading and computation
- **Larger models**: Train models that don't fit on a single GPU by sharding parameters across devices
- **Bigger batches**: Increase effective batch size for better gradient estimates

scModelForge supports two distributed training strategies via PyTorch Lightning: Data Distributed Parallel (DDP) and Fully Sharded Data Parallel (FSDP).

## DDP vs FSDP: Which Should I Use?

### Data Distributed Parallel (DDP)

**How it works**: Each GPU maintains a complete copy of the model. Training data is split across GPUs, and gradients are synchronized after each backward pass.

**Memory usage**: Full model + optimizer state on each GPU

**Best for**:
- Models that fit comfortably on a single GPU
- Multi-GPU systems where communication bandwidth is a concern
- Simpler debugging and profiling workflows

### Fully Sharded Data Parallel (FSDP)

**How it works**: Model parameters, gradients, and optimizer states are sharded (split) across GPUs. Each GPU holds only a fraction of the model. During forward and backward passes, parameters are gathered on-demand via all-gather operations.

**Memory usage**: Model parameters and optimizer state divided by number of GPUs

**Best for**:
- Models that don't fit on a single GPU
- Maximizing effective batch size on available hardware
- Training very large models (billions of parameters)

### Decision Matrix

| Scenario | Recommended Strategy |
|----------|---------------------|
| Model fits on 1 GPU, want faster training | DDP |
| Model doesn't fit on 1 GPU | FSDP with FULL_SHARD |
| Want maximum batch size | FSDP + gradient accumulation |
| Debugging model architecture | DDP or NO_SHARD |
| Multi-node cluster with fast inter-node network | FSDP with FULL_SHARD |
| Multi-node cluster with slow inter-node network | FSDP with HYBRID_SHARD |

## DDP Training

To train with DDP, simply specify the number of GPUs and set the strategy to `ddp`:

```yaml
training:
  strategy: ddp
  num_gpus: 4
  precision: bf16-mixed
  batch_size: 64              # Per GPU, effective batch = 64 * 4 = 256
  max_epochs: 50
  gradient_clip: 1.0
```

Launch training with:

```bash
scmodelforge train --config config.yaml
```

PyTorch Lightning automatically detects available GPUs and spawns one process per GPU. Each process loads its own copy of the model and processes a different subset of the data.

**Important notes**:
- The `batch_size` is per GPU. With 4 GPUs and batch size 64, your effective batch size is 256.
- Each GPU worker loads data independently. Use `num_workers` in the data config to parallelize data loading per GPU.
- DDP requires no additional configuration beyond `strategy: ddp` and `num_gpus`.

## FSDP Training

For large models or memory-constrained scenarios, use FSDP:

```yaml
training:
  num_gpus: 4
  precision: bf16-mixed
  batch_size: 32
  max_epochs: 50
  gradient_clip: 1.0
  fsdp:
    sharding_strategy: FULL_SHARD
    cpu_offload: false
    activation_checkpointing: true
    min_num_params: 1000000
```

When the `fsdp` configuration block is present, the training pipeline automatically uses FSDP instead of DDP.

### FSDP Configuration Parameters

#### sharding_strategy

Controls how model parameters are distributed across GPUs:

- **FULL_SHARD** (default): Shard model parameters, gradients, and optimizer states across all GPUs. Maximum memory savings, highest communication overhead.
- **SHARD_GRAD_OP**: Shard only gradients and optimizer states. Parameters are replicated. Moderate memory savings, lower communication overhead than FULL_SHARD.
- **HYBRID_SHARD**: Shard within nodes (connected by fast interconnect), replicate across nodes. Best for multi-node clusters with varying network speeds.
- **NO_SHARD**: No sharding (equivalent to DDP). Useful for debugging FSDP-specific issues.

#### cpu_offload

When `true`, offloads parameters, gradients, and optimizer states to CPU when not actively in use. Significantly reduces GPU memory usage but slows training due to CPU-GPU data transfers.

**Use when**: Training very large models and running out of GPU memory even with FULL_SHARD.

**Cost**: 20-50% slower training depending on CPU-GPU bandwidth.

#### activation_checkpointing

When `true`, enables gradient checkpointing on `nn.TransformerEncoderLayer` modules. Instead of storing all intermediate activations for the backward pass, checkpointing discards them during the forward pass and recomputes them on-demand during backward.

**Trade-off**: Reduces memory usage by ~30-50% but increases training time by ~30%.

**Use when**: Running out of GPU memory during training, especially with large batch sizes or long sequences.

#### min_num_params

Minimum number of parameters required for a module to be wrapped by FSDP. Smaller values create more FSDP units (finer granularity), which can improve memory efficiency but increases communication overhead.

**Default**: 1,000,000 (1M parameters)

**Adjust when**: Fine-tuning sharding granularity. Increase for coarser sharding (less communication), decrease for finer sharding (better memory distribution).

### Sharding Strategy Comparison

| Strategy | Memory Savings | Communication Overhead | Best Use Case |
|----------|---------------|----------------------|---------------|
| FULL_SHARD | Maximum (params + grads + opt state) | Highest (all-gather every forward/backward) | Very large models, memory-constrained GPUs |
| SHARD_GRAD_OP | Moderate (grads + opt state) | Moderate (reduce-scatter on backward) | Medium-large models, moderate memory constraints |
| HYBRID_SHARD | High within nodes | Balanced (intra-node all-gather, inter-node reduce) | Multi-node clusters with fast intra-node, slow inter-node networks |
| NO_SHARD | None (same as DDP) | Lowest (reduce on backward) | Debugging, small models |

## Activation Checkpointing Deep Dive

Activation checkpointing is a memory-compute trade-off technique. Normally, the forward pass stores all intermediate activations in memory so they're available for the backward pass (to compute gradients). With checkpointing:

1. **Forward pass**: Compute activations but discard most of them (only checkpoint boundaries are saved)
2. **Backward pass**: Recompute discarded activations on-demand as needed for gradient calculation

**Memory impact**: Reduces activation memory by 50-70% for transformer models

**Compute impact**: Increases training time by 20-35% (one extra forward pass per backward pass)

**When to enable**:
- Out of memory errors during training
- Want to increase batch size beyond current memory limits
- Training very deep models (many transformer layers)

**Example configuration**:

```yaml
training:
  batch_size: 64              # Can increase due to checkpointing
  fsdp:
    sharding_strategy: FULL_SHARD
    activation_checkpointing: true
```

## Gradient Accumulation

Gradient accumulation simulates larger batch sizes by accumulating gradients over multiple forward-backward passes before updating model parameters:

```yaml
training:
  batch_size: 16
  gradient_accumulation: 4    # Effective batch size = 16 * 4 * num_gpus
  num_gpus: 4
  # Effective batch = 16 * 4 * 4 = 256
```

**How it works**:
1. Run forward and backward pass
2. Accumulate gradients without updating parameters
3. Repeat for `gradient_accumulation` steps
4. Update parameters using accumulated gradients
5. Reset gradients and repeat

**Use cases**:
- Memory-constrained systems: Use small per-GPU batch size but maintain large effective batch
- Batch size sensitivity: Experiment with very large effective batch sizes
- Reproducibility: Match effective batch sizes across different hardware configurations

**Important**: When using gradient accumulation, you may need to adjust your learning rate. A common heuristic is linear scaling: if you double the effective batch size, double the learning rate (or adjust the warmup schedule).

## Mixed Precision Training

scModelForge supports three precision modes via PyTorch Lightning:

### bf16-mixed (Recommended)

Uses bfloat16 for most operations, float32 for numerically sensitive operations (layer norms, loss computation).

**Advantages**:
- No loss scaling required (unlike float16)
- Same dynamic range as float32
- Faster training (2-3x speedup on Ampere and newer GPUs)
- Lower memory usage (50% reduction for activations)

**Requirements**: Ampere (A100, A30) or newer GPU architecture, or AMD MI250

**Configuration**:
```yaml
training:
  precision: bf16-mixed
```

### 16-mixed

Uses float16 for most operations, float32 for numerically sensitive operations. Requires loss scaling to prevent gradient underflow.

**Advantages**:
- Supported on older GPUs (Volta V100, Turing T4)
- Faster training than full precision
- Lower memory usage

**Disadvantages**:
- Requires loss scaling (automatic in Lightning)
- Narrower dynamic range than bf16
- Occasional numerical instability

**Configuration**:
```yaml
training:
  precision: 16-mixed
```

### 32-true

Full float32 precision throughout training.

**Advantages**:
- Maximum numerical stability
- Best for debugging

**Disadvantages**:
- Slower training
- Higher memory usage

**Configuration**:
```yaml
training:
  precision: 32-true
```

**Recommendation**: Always use `bf16-mixed` if your hardware supports it. Fall back to `16-mixed` for older GPUs. Only use `32-true` if you encounter numerical issues.

## Complete Example: Large Model on 8 GPUs

Here's a complete configuration for training a large transformer model using FSDP:

```yaml
data:
  source: local
  paths:
    - ./data/preprocessed.h5ad
  gene_vocab: human_protein_coding
  max_genes: 4096
  num_workers: 8
  streaming: true
  streaming_chunk_size: 10000
  streaming_shuffle_buffer: 50000

tokenizer:
  strategy: rank_value
  max_genes: 4096
  prepend_cls: true
  masking:
    mask_ratio: 0.15

model:
  architecture: transformer_encoder
  hidden_dim: 1024
  num_layers: 12
  num_heads: 16
  dropout: 0.1
  max_seq_len: 4096

training:
  num_gpus: 8
  batch_size: 16                      # 16 per GPU
  gradient_accumulation: 2            # Effective = 16 * 2 * 8 = 256
  max_epochs: 100
  precision: bf16-mixed
  gradient_clip: 1.0

  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 0.01

  scheduler:
    name: cosine_warmup
    warmup_steps: 5000
    total_steps: 500000

  fsdp:
    sharding_strategy: FULL_SHARD
    cpu_offload: false
    activation_checkpointing: true
    min_num_params: 1000000

  logger: wandb
  wandb_project: scmodelforge-large
  log_every_n_steps: 50
  checkpoint_dir: ./checkpoints/large-model
  save_top_k: 3
```

This configuration:
- Uses streaming dataset to handle large data files
- FULL_SHARD to maximize memory efficiency
- Activation checkpointing to reduce activation memory
- Gradient accumulation to achieve effective batch size of 256
- bf16-mixed precision for speed and memory
- 8 data loading workers per GPU for fast I/O

## Monitoring GPU Utilization

During training, monitor GPU utilization with:

```bash
nvidia-smi -l 1  # Update every second
```

Look for:
- **GPU utilization**: Should be 80-100% during training steps
- **Memory usage**: Should be high but not causing OOM errors
- **GPU power**: High power draw indicates efficient GPU usage

If GPU utilization is low:
- Check data loading bottleneck (increase `num_workers`)
- Increase batch size if memory permits
- Profile training loop to identify bottlenecks

## Multi-Node Training

PyTorch Lightning supports multi-node training via SLURM or torchrun. For a SLURM cluster:

```bash
#!/bin/bash
#SBATCH --job-name=scmodelforge
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8        # 8 GPUs per node
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00

# Activate environment
source .venv/bin/activate

# Launch training
srun scmodelforge train --config config.yaml
```

Your config should specify total GPUs:

```yaml
training:
  num_gpus: 32                      # 4 nodes * 8 GPUs
  num_nodes: 4
  fsdp:
    sharding_strategy: HYBRID_SHARD # Better for multi-node
```

Lightning automatically handles inter-node communication and rank assignment via SLURM environment variables.

## Practical Tips

1. **Start simple**: Begin with DDP on a single node before moving to FSDP
2. **Monitor memory**: Use `nvidia-smi` to track GPU memory usage during training
3. **Activation checkpointing first**: If running out of memory, enable activation checkpointing before trying cpu_offload
4. **Profile before optimizing**: Use PyTorch profiler to identify actual bottlenecks before making changes
5. **Effective batch size matters**: `batch_size * gradient_accumulation * num_gpus` is your true batch size
6. **Learning rate scaling**: Consider adjusting learning rate when changing effective batch size (linear scaling rule: 2x batch = 2x LR)
7. **Data loading bottleneck**: Increase `num_workers` if GPUs are underutilized during training
8. **Checkpoint compatibility**: FSDP checkpoints are compatible with non-FSDP loading for inference/fine-tuning
9. **Experiment tracking**: Use W&B or TensorBoard to compare different distributed configurations
10. **Sharded data**: For very large datasets, use the sharding feature to avoid data loading bottlenecks

## Troubleshooting

**Out of memory errors**:
1. Reduce `batch_size`
2. Enable `activation_checkpointing: true`
3. Use FULL_SHARD instead of SHARD_GRAD_OP
4. Enable `cpu_offload: true` (last resort)

**Slow training**:
1. Increase `num_workers` for data loading
2. Check if using streaming dataset for large data
3. Profile to identify bottlenecks
4. Ensure `precision: bf16-mixed` or `16-mixed`

**Communication bottleneck**:
1. Try HYBRID_SHARD for multi-node
2. Increase `min_num_params` for coarser sharding
3. Ensure fast interconnect (InfiniBand) for multi-node

**Hanging during initialization**:
1. Check network connectivity between nodes
2. Verify SLURM environment variables
3. Check firewall rules for distributed training ports

## What's Next

- [Large-Scale Data Processing](large_scale_data.md): Learn about sharding and streaming datasets for efficient data loading
- [Pretraining Tutorial](pretraining.md): End-to-end guide to pretraining foundation models
- [Fine-Tuning Tutorial](finetuning_cell_type.md): Adapt pretrained models to downstream tasks
