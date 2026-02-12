# `scmodelforge.finetuning`

Fine-tuning pretrained single-cell foundation models for downstream tasks.

## Overview

The `scmodelforge.finetuning` module provides infrastructure for adapting pretrained single-cell foundation models to specific downstream tasks like cell type classification or gene expression regression. It implements a complete fine-tuning pipeline with support for task-specific heads, discriminative learning rates, gradual backbone unfreezing, and parameter-efficient fine-tuning via LoRA adapters.

The module follows a compositional design: a pretrained backbone (loaded from a checkpoint) is paired with a task-specific prediction head (classification or regression). The `FineTuneModel` wraps both components and implements the forward pass: encoding cells via the backbone's `encode()` method, then predicting task outputs through the head. The `FineTuneLightningModule` handles training with advanced techniques like discriminative learning rates (lower LR for backbone, higher for head) and gradual unfreezing (train only the head initially, then unfreeze the backbone after N epochs).

For parameter efficiency, the module supports LoRA (Low-Rank Adaptation) via HuggingFace `peft`. LoRA inserts trainable low-rank matrices into transformer layers while freezing the original weights, dramatically reducing trainable parameters (often by 99 percent) while maintaining strong performance. When LoRA is enabled, freeze/unfreeze operations become no-ops since LoRA manages its own trainable/frozen parameter split.

The `FineTuneDataModule` handles task-labeled data: it loads AnnData with a label column (e.g. `cell_type`), encodes string labels to integers via `LabelEncoder`, performs stratified train/val splitting for classification tasks, and creates dataloaders without masking (fine-tuning uses full unmasked input). The `FineTunePipeline` orchestrates the complete workflow from config to trained model, integrating seamlessly with PyTorch Lightning's ecosystem.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| `FineTunePipeline` | End-to-end fine-tuning orchestrator with config-driven setup |
| `FineTuneModel` | Backbone + task head composition with freeze/unfreeze control |
| `FineTuneLightningModule` | Lightning module with discriminative LR and gradual unfreezing |
| `FineTuneDataModule` | Labeled data module with stratified splits |
| `LabelEncoder` | Bidirectional string label ↔ integer index mapping |
| `ClassificationHead` | MLP head mapping embeddings to class logits |
| `RegressionHead` | MLP head mapping embeddings to continuous outputs |
| `build_task_head` | Factory function for classification/regression heads |
| `load_pretrained_backbone` | Load weights from Lightning or raw checkpoint |
| `apply_lora` | Apply LoRA adapters to a backbone model |
| `has_lora` | Check if model has LoRA adapters |
| `save_lora_weights` | Save only LoRA adapter weights |
| `load_lora_weights` | Load LoRA adapters onto a base model |
| `count_lora_parameters` | Count trainable vs total parameters |

## API Reference

### `FineTunePipeline`

Config-driven fine-tuning pipeline orchestrating data loading, backbone loading, head construction, and Lightning training.

```python
from scmodelforge.config import load_config
from scmodelforge.finetuning import FineTunePipeline

config = load_config("configs/finetune_cell_type.yaml")
pipeline = FineTunePipeline(config)
trainer = pipeline.run()
```

#### Constructor

```python
FineTunePipeline(config: ScModelForgeConfig)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `ScModelForgeConfig` | *required* | Full configuration with `finetune` section set |

**Raises:**
- `ValueError` — If `config.finetune` is `None`

#### Methods

##### `run() -> pl.Trainer`

Execute the full fine-tuning pipeline.

**Workflow:**
1. Seed RNG with `pl.seed_everything`
2. Build `FineTuneDataModule` and call `setup()`
3. Infer `vocab_size` from gene vocabulary
4. Build backbone model from registry
5. Load pretrained checkpoint if provided
6. Apply LoRA if configured (with parameter count logging)
7. Infer `n_classes` from `LabelEncoder` if needed
8. Build task head via `build_task_head()`
9. Compose `FineTuneModel` with freeze logic
10. Create `FineTuneLightningModule` with discriminative LR
11. Build callbacks (checkpoint, LR monitor, throughput)
12. Build logger (wandb/tensorboard/csv)
13. Resolve devices and strategy
14. Create `pl.Trainer` and call `trainer.fit()`

**Returns:**
- `pl.Trainer` — The Lightning Trainer after fitting

**Example:**

```python
from scmodelforge.config import load_config
from scmodelforge.finetuning import FineTunePipeline

# Load fine-tuning config
config = load_config("configs/finetune_cell_type.yaml")

# Override for quick test
config.training.max_epochs = 5
config.finetune.freeze_backbone_epochs = 2

# Run pipeline
pipeline = FineTunePipeline(config)
trainer = pipeline.run()

# Access best checkpoint
best_ckpt = trainer.checkpoint_callback.best_model_path
print(f"Best checkpoint: {best_ckpt}")
```

---

### `FineTuneModel`

Wraps a pretrained backbone with a task-specific head. Forward pass uses `backbone.encode()` to get embeddings, then applies the head.

```python
from scmodelforge.finetuning import FineTuneModel, ClassificationHead
from scmodelforge.models import get_model

backbone = get_model("transformer_encoder", config.model)
head = ClassificationHead(input_dim=512, n_classes=10)
model = FineTuneModel(
    backbone=backbone,
    head=head,
    task="classification",
    freeze_backbone=True,
)
```

#### Constructor

```python
FineTuneModel(
    backbone: nn.Module,
    head: nn.Module,
    task: str = "classification",
    freeze_backbone: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backbone` | `nn.Module` | *required* | Pretrained model with `encode()` method returning `(B, H)` embeddings |
| `head` | `nn.Module` | *required* | Task-specific head (classification or regression) |
| `task` | `str` | `"classification"` | Task type: `"classification"` or `"regression"` |
| `freeze_backbone` | `bool` | `False` | If True, freeze all backbone parameters on construction |

#### Methods

##### `forward(...) -> ModelOutput`

Forward pass: encode then predict.

```python
forward(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    values: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    **kwargs: Any,
) -> ModelOutput
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_ids` | `torch.Tensor` | Token IDs of shape `(B, S)` |
| `attention_mask` | `torch.Tensor` | Mask of shape `(B, S)` |
| `values` | `torch.Tensor \| None` | Optional expression values of shape `(B, S)` |
| `labels` | `torch.Tensor \| None` | Task labels. Classification: `(B,)` long. Regression: `(B,)` or `(B, D)` float |
| `**kwargs` | `Any` | Extra keys forwarded to backbone (e.g. `bin_ids`) |

**Returns:**
- `ModelOutput` — With `loss` (if labels provided), `logits`, `embeddings`

**Loss computation:**
- Classification: `CrossEntropyLoss(logits, labels)`
- Regression: `MSELoss(logits, labels)`

##### `encode(...) -> torch.Tensor`

Extract cell embeddings from the backbone.

```python
encode(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    values: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor
```

**Returns:**
- `torch.Tensor` — Cell embeddings of shape `(B, H)`

##### `freeze_backbone() -> None`

Freeze all backbone parameters. No-op when LoRA is active.

##### `unfreeze_backbone() -> None`

Unfreeze all backbone parameters. No-op when LoRA is active.

##### `num_parameters(trainable_only: bool = True) -> int`

Count parameters.

**Parameters:**
- `trainable_only` (bool) — If True (default), count only trainable parameters

**Returns:**
- `int` — Parameter count

#### Properties

##### `has_lora: bool`

Whether the backbone has LoRA adapters applied.

**Example:**

```python
from scmodelforge.finetuning import FineTuneModel, ClassificationHead, apply_lora
from scmodelforge.models import get_model
from scmodelforge.config.schema import LoRAConfig

# Build model
backbone = get_model("transformer_encoder", config.model)
head = ClassificationHead(input_dim=512, n_classes=10, dropout=0.1)

# Apply LoRA
lora_config = LoRAConfig(enabled=True, rank=8, alpha=16)
backbone = apply_lora(backbone, lora_config)

# Compose fine-tune model
model = FineTuneModel(backbone, head, task="classification")

print(f"LoRA active: {model.has_lora}")
print(f"Trainable params: {model.num_parameters(trainable_only=True)}")
print(f"Total params: {model.num_parameters(trainable_only=False)}")
```

---

### `FineTuneLightningModule`

Lightning module for fine-tuning with discriminative learning rates, gradual unfreezing, and task-specific metrics.

```python
from scmodelforge.finetuning import FineTuneLightningModule

lightning_module = FineTuneLightningModule(
    model=finetune_model,
    optimizer_config=config.training.optimizer,
    scheduler_config=config.training.scheduler,
    task="classification",
    backbone_lr=1e-5,
    head_lr=1e-3,
    freeze_backbone_epochs=2,
)
```

#### Constructor

```python
FineTuneLightningModule(
    model: nn.Module,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig | None = None,
    task: str = "classification",
    backbone_lr: float | None = None,
    head_lr: float | None = None,
    freeze_backbone_epochs: int = 0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | A `FineTuneModel` wrapping backbone + head |
| `optimizer_config` | `OptimizerConfig` | *required* | Optimizer configuration (name, lr, weight_decay) |
| `scheduler_config` | `SchedulerConfig \| None` | `None` | Optional scheduler configuration |
| `task` | `str` | `"classification"` | Task type: `"classification"` or `"regression"` |
| `backbone_lr` | `float \| None` | `None` | Discriminative LR for backbone. `None` uses global LR |
| `head_lr` | `float \| None` | `None` | Discriminative LR for head. `None` uses global LR |
| `freeze_backbone_epochs` | `int` | `0` | Unfreeze backbone after this many epochs. `0` = no schedule |

#### Methods

##### `forward(batch: dict[str, torch.Tensor]) -> ModelOutput`

Forward pass through the fine-tune model.

##### `training_step(batch: dict, batch_idx: int) -> torch.Tensor`

Single training step with task-specific logging.

**Logs:**
- `train/loss` (progress bar, sync_dist)
- `train/accuracy` (classification only, sync_dist)

**Returns:**
- `torch.Tensor` — Loss for backpropagation

##### `validation_step(batch: dict, batch_idx: int) -> None`

Single validation step with task-specific logging.

**Logs:**
- `val/loss` (progress bar, sync_dist)
- `val/accuracy` (classification only, sync_dist)

##### `on_train_epoch_start() -> None`

Handle gradual unfreezing at the configured epoch. Logs when unfreezing occurs.

##### `configure_optimizers() -> dict[str, Any]`

Build optimizer with discriminative LRs for backbone and head.

**Parameter groups:**
- Backbone decay (weight_decay > 0)
- Backbone no-decay (bias, LayerNorm)
- Head decay
- Head no-decay

**Returns:**
- `dict` — With keys `optimizer` and optionally `lr_scheduler`

**Example:**

```python
from scmodelforge.config.schema import OptimizerConfig
from scmodelforge.finetuning import FineTuneModel, FineTuneLightningModule, ClassificationHead
import lightning.pytorch as pl

# Build model
finetune_model = FineTuneModel(backbone, head, task="classification", freeze_backbone=True)

# Configure with discriminative LR
opt_cfg = OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.01)
lightning_module = FineTuneLightningModule(
    model=finetune_model,
    optimizer_config=opt_cfg,
    task="classification",
    backbone_lr=1e-5,  # 10x lower than default
    head_lr=1e-3,      # 10x higher than default
    freeze_backbone_epochs=2,  # Unfreeze after 2 epochs
)

# Train
trainer = pl.Trainer(max_epochs=10)
trainer.fit(lightning_module, train_loader, val_loader)
```

---

### `FineTuneDataModule`

Data module for fine-tuning with task labels. Handles label encoding, stratified splitting (classification), and dataloader construction without masking.

```python
from scmodelforge.finetuning import FineTuneDataModule

data_module = FineTuneDataModule(
    data_config=config.data,
    tokenizer_config=config.tokenizer,
    finetune_config=config.finetune,
    training_batch_size=64,
    num_workers=4,
    val_split=0.1,
    seed=42,
)
data_module.setup()
```

#### Constructor

```python
FineTuneDataModule(
    data_config: DataConfig,
    tokenizer_config: TokenizerConfig,
    finetune_config: FinetuneConfig,
    training_batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: int = 42,
    adata: Any | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_config` | `DataConfig` | *required* | Data loading and preprocessing configuration |
| `tokenizer_config` | `TokenizerConfig` | *required* | Tokenizer strategy configuration |
| `finetune_config` | `FinetuneConfig` | *required* | Fine-tuning config (label_key, head, etc.) |
| `training_batch_size` | `int` | `64` | Batch size for train and val |
| `num_workers` | `int` | `4` | DataLoader worker count |
| `val_split` | `float` | `0.1` | Fraction of data reserved for validation |
| `seed` | `int` | `42` | Random seed for reproducible splits |
| `adata` | `Any \| None` | `None` | Optional pre-loaded AnnData (skips file loading) |

#### Properties

##### `gene_vocab: GeneVocab`

Gene vocabulary (available after `setup()`).

##### `tokenizer: BaseTokenizer`

Tokenizer instance (available after `setup()`).

##### `label_encoder: LabelEncoder | None`

Label encoder for classification tasks. `None` for regression.

#### Methods

##### `setup(stage: str | None = None) -> None`

Load data, encode labels, split into train/val.

**Workflow:**
1. Load AnnData from file or Census with `label_key` in obs
2. Build `GeneVocab` and `PreprocessingPipeline`
3. Build `CellDataset` with `obs_keys=[label_key]`
4. Build tokenizer (no masking)
5. Build labels: `LabelEncoder` for classification, float for regression
6. Stratified split for classification, random for regression
7. Wrap with `TokenizedCellDataset` (no masking)
8. Inject labels via `_LabeledTokenizedDataset`

**Notes:**
- Idempotent — calling multiple times is safe
- Classification splits are stratified (balanced class distribution)
- No masking is applied (fine-tuning uses full input)

##### `train_dataloader() -> DataLoader`

Training DataLoader with shuffle and custom collate function.

**Returns:**
- `DataLoader` — With shuffle=True, collate_fn=_finetune_collate

##### `val_dataloader() -> DataLoader`

Validation DataLoader without shuffle.

**Returns:**
- `DataLoader` — With shuffle=False, collate_fn=_finetune_collate

**Example:**

```python
from scmodelforge.config import load_config
from scmodelforge.finetuning import FineTuneDataModule

config = load_config("configs/finetune_cell_type.yaml")
data_module = FineTuneDataModule(
    data_config=config.data,
    tokenizer_config=config.tokenizer,
    finetune_config=config.finetune,
    training_batch_size=64,
    val_split=0.1,
)

data_module.setup()

# Inspect label encoder
print(f"Classes: {data_module.label_encoder.classes}")
print(f"N classes: {data_module.label_encoder.n_classes}")

# Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Inspect batch
batch = next(iter(train_loader))
print(batch.keys())  # dict_keys(['input_ids', 'attention_mask', 'values', 'task_labels', ...])
```

---

### `LabelEncoder`

Bidirectional mapping between string labels and integer indices.

```python
from scmodelforge.finetuning import LabelEncoder

labels = ["T cell", "B cell", "T cell", "Monocyte", "B cell"]
encoder = LabelEncoder(labels)
```

#### Constructor

```python
LabelEncoder(labels: Sequence[str])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | `Sequence[str]` | *required* | Raw string labels (e.g. cell type names). Unique values sorted for deterministic ordering |

#### Properties

##### `n_classes: int`

Number of unique classes.

##### `classes: list[str]`

Sorted list of class labels.

#### Methods

##### `encode(label: str) -> int`

Encode a string label to its integer index.

**Parameters:**
- `label` (str) — String label to encode

**Returns:**
- `int` — Integer index

**Raises:**
- `KeyError` — If label was not seen during construction

##### `decode(idx: int) -> str`

Decode an integer index back to its string label.

**Parameters:**
- `idx` (int) — Integer index to decode

**Returns:**
- `str` — Original string label

**Raises:**
- `IndexError` — If idx is out of range

**Example:**

```python
from scmodelforge.finetuning import LabelEncoder

labels = ["T cell", "B cell", "T cell", "Monocyte", "B cell", "NK cell"]
encoder = LabelEncoder(labels)

print(f"Classes: {encoder.classes}")  # ['B cell', 'Monocyte', 'NK cell', 'T cell']
print(f"N classes: {encoder.n_classes}")  # 4

# Encode
idx = encoder.encode("T cell")  # 3
print(f"T cell -> {idx}")

# Decode
label = encoder.decode(idx)  # "T cell"
print(f"{idx} -> {label}")

# Batch encode
int_labels = [encoder.encode(lbl) for lbl in labels]
print(int_labels)  # [3, 0, 3, 1, 0, 2]
```

---

### `ClassificationHead`

MLP head mapping cell embeddings to class logits.

```python
from scmodelforge.finetuning import ClassificationHead

head = ClassificationHead(
    input_dim=512,
    n_classes=10,
    hidden_dim=256,
    dropout=0.1,
)
```

#### Constructor

```python
ClassificationHead(
    input_dim: int,
    n_classes: int,
    hidden_dim: int | None = None,
    dropout: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | `int` | *required* | Dimension of input embeddings (backbone hidden dim) |
| `n_classes` | `int` | *required* | Number of output classes |
| `hidden_dim` | `int \| None` | `None` | Optional intermediate hidden layer dimension. `None` = direct projection |
| `dropout` | `float` | `0.1` | Dropout probability |

**Architecture:**
- If `hidden_dim` is None: `Dropout -> Linear(input_dim, n_classes)`
- If `hidden_dim` is set: `Linear(input_dim, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, n_classes)`

#### Methods

##### `forward(embeddings: torch.Tensor) -> torch.Tensor`

Forward pass.

**Parameters:**
- `embeddings` (torch.Tensor) — Cell embeddings of shape `(B, H)`

**Returns:**
- `torch.Tensor` — Logits of shape `(B, n_classes)`

---

### `RegressionHead`

MLP head mapping cell embeddings to continuous outputs.

```python
from scmodelforge.finetuning import RegressionHead

head = RegressionHead(
    input_dim=512,
    output_dim=1,
    hidden_dim=256,
    dropout=0.1,
)
```

#### Constructor

```python
RegressionHead(
    input_dim: int,
    output_dim: int = 1,
    hidden_dim: int | None = None,
    dropout: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | `int` | *required* | Dimension of input embeddings (backbone hidden dim) |
| `output_dim` | `int` | `1` | Number of output dimensions |
| `hidden_dim` | `int \| None` | `None` | Optional intermediate hidden layer dimension. `None` = direct projection |
| `dropout` | `float` | `0.1` | Dropout probability |

**Architecture:**
- If `hidden_dim` is None: `Dropout -> Linear(input_dim, output_dim)`
- If `hidden_dim` is set: `Linear(input_dim, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, output_dim)`

#### Methods

##### `forward(embeddings: torch.Tensor) -> torch.Tensor`

Forward pass.

**Parameters:**
- `embeddings` (torch.Tensor) — Cell embeddings of shape `(B, H)`

**Returns:**
- `torch.Tensor` — Predictions of shape `(B, output_dim)`

---

### `build_task_head`

Factory function to build a task head from configuration.

```python
from scmodelforge.finetuning import build_task_head
from scmodelforge.config.schema import TaskHeadConfig

config = TaskHeadConfig(task="classification", n_classes=10, hidden_dim=256, dropout=0.1)
head = build_task_head(config, input_dim=512)
```

#### Signature

```python
build_task_head(config: TaskHeadConfig, input_dim: int) -> nn.Module
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `TaskHeadConfig` | *required* | Task head configuration |
| `input_dim` | `int` | *required* | Backbone hidden dimension |

**Returns:**
- `nn.Module` — A `ClassificationHead` or `RegressionHead`

**Raises:**
- `ValueError` — If `config.task` is not `"classification"` or `"regression"`
- `ValueError` — If `config.task="classification"` and `config.n_classes` is `None`

**Example:**

```python
from scmodelforge.config.schema import TaskHeadConfig
from scmodelforge.finetuning import build_task_head

# Classification head
cls_config = TaskHeadConfig(
    task="classification",
    n_classes=20,
    hidden_dim=512,
    dropout=0.2,
)
cls_head = build_task_head(cls_config, input_dim=768)

# Regression head
reg_config = TaskHeadConfig(
    task="regression",
    output_dim=5,
    hidden_dim=256,
    dropout=0.1,
)
reg_head = build_task_head(reg_config, input_dim=768)
```

---

### `load_pretrained_backbone`

Load pretrained weights into a backbone model. Handles both raw state_dict files and Lightning checkpoint files.

```python
from scmodelforge.finetuning import load_pretrained_backbone
from scmodelforge.models import get_model

backbone = get_model("transformer_encoder", config.model)
load_pretrained_backbone(backbone, "checkpoints/pretrained.ckpt", strict=False)
```

#### Signature

```python
load_pretrained_backbone(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | The backbone model to load weights into |
| `checkpoint_path` | `str` | *required* | Path to the checkpoint file |
| `strict` | `bool` | `False` | If True, require exact match between checkpoint and model keys |

**Behavior:**
1. Load checkpoint with `torch.load()`
2. Extract `state_dict` (handles Lightning format with `"state_dict"` key)
3. Strip `"model."` prefix added by LightningModule
4. If `strict=False`: filter to keys present in model, log missing/unexpected keys
5. Load into model with `load_state_dict()`

**Example:**

```python
from scmodelforge.finetuning import load_pretrained_backbone
from scmodelforge.models import get_model
from scmodelforge.config import load_config

config = load_config("configs/finetune.yaml")
backbone = get_model("transformer_encoder", config.model)

# Load from Lightning checkpoint
load_pretrained_backbone(
    backbone,
    "checkpoints/epoch10-val_loss0.5432.ckpt",
    strict=False,  # Permissive loading
)

# Or from raw state dict
load_pretrained_backbone(
    backbone,
    "pretrained_weights.pt",
    strict=True,  # Exact match required
)
```

---

### `apply_lora`

Apply LoRA (Low-Rank Adaptation) adapters to a backbone model using HuggingFace `peft`.

```python
from scmodelforge.finetuning import apply_lora
from scmodelforge.config.schema import LoRAConfig

lora_config = LoRAConfig(enabled=True, rank=8, alpha=16, dropout=0.05)
backbone_with_lora = apply_lora(backbone, lora_config)
```

#### Signature

```python
apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | The backbone module to wrap |
| `config` | `LoRAConfig` | *required* | LoRA configuration (rank, alpha, dropout, target_modules, bias) |

**Returns:**
- `nn.Module` — A `PeftModel` wrapping the original model. Non-LoRA parameters frozen, only adapter parameters trainable

**Raises:**
- `ImportError` — If `peft` is not installed

**Default target modules:**
- `["out_proj", "linear1", "linear2"]` (attention output projection and FFN layers)

**Example:**

```python
from scmodelforge.finetuning import apply_lora, count_lora_parameters
from scmodelforge.config.schema import LoRAConfig
from scmodelforge.models import get_model

backbone = get_model("transformer_encoder", config.model)
print(f"Original params: {sum(p.numel() for p in backbone.parameters())}")

# Apply LoRA
lora_config = LoRAConfig(
    enabled=True,
    rank=8,
    alpha=16,
    dropout=0.05,
    target_modules=["out_proj", "linear1", "linear2"],
    bias="none",
)
backbone = apply_lora(backbone, lora_config)

# Check trainable params
trainable, total = count_lora_parameters(backbone)
print(f"Trainable: {trainable} / {total} ({100*trainable/total:.2f}%)")
# Trainable: 295936 / 33554432 (0.88%)
```

---

### `has_lora`

Check if a model is wrapped with peft LoRA.

```python
from scmodelforge.finetuning import has_lora

if has_lora(model):
    print("LoRA adapters active")
```

#### Signature

```python
has_lora(model: nn.Module) -> bool
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | The model to check |

**Returns:**
- `bool` — `True` if the model is a `PeftModel`

---

### `save_lora_weights`

Save only the LoRA adapter weights (not the full model).

```python
from scmodelforge.finetuning import save_lora_weights

save_lora_weights(model, "checkpoints/lora_adapters/")
```

#### Signature

```python
save_lora_weights(model: nn.Module, path: str) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | A `PeftModel` with LoRA adapters |
| `path` | `str` | *required* | Directory path to save adapter weights |

**Raises:**
- `ValueError` — If the model does not have LoRA adapters

---

### `load_lora_weights`

Load LoRA adapter weights onto a base model.

```python
from scmodelforge.finetuning import load_lora_weights

model_with_lora = load_lora_weights(base_model, "checkpoints/lora_adapters/")
```

#### Signature

```python
load_lora_weights(model: nn.Module, path: str) -> nn.Module
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | The base model (without LoRA adapters) |
| `path` | `str` | *required* | Directory path containing saved adapter weights |

**Returns:**
- `nn.Module` — A `PeftModel` with loaded adapter weights

**Raises:**
- `ImportError` — If `peft` is not installed

**Example:**

```python
from scmodelforge.finetuning import apply_lora, save_lora_weights, load_lora_weights
from scmodelforge.config.schema import LoRAConfig
from scmodelforge.models import get_model

# Train with LoRA
backbone = get_model("transformer_encoder", config.model)
lora_config = LoRAConfig(enabled=True, rank=8, alpha=16)
backbone = apply_lora(backbone, lora_config)

# ... train the model ...

# Save only adapters (lightweight)
save_lora_weights(backbone, "lora_adapters/")

# Later: load adapters onto fresh backbone
fresh_backbone = get_model("transformer_encoder", config.model)
backbone_with_adapters = load_lora_weights(fresh_backbone, "lora_adapters/")
```

---

### `count_lora_parameters`

Return trainable and total parameter counts.

```python
from scmodelforge.finetuning import count_lora_parameters

trainable, total = count_lora_parameters(model)
print(f"{trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")
```

#### Signature

```python
count_lora_parameters(model: nn.Module) -> tuple[int, int]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | A model, optionally wrapped with LoRA |

**Returns:**
- `tuple[int, int]` — `(trainable_params, total_params)`

**Example:**

```python
from scmodelforge.finetuning import apply_lora, count_lora_parameters
from scmodelforge.config.schema import LoRAConfig
from scmodelforge.models import get_model

backbone = get_model("transformer_encoder", config.model)

# Before LoRA
trainable, total = count_lora_parameters(backbone)
print(f"Before LoRA: {trainable:,} / {total:,}")

# After LoRA
lora_config = LoRAConfig(enabled=True, rank=8, alpha=16)
backbone = apply_lora(backbone, lora_config)
trainable, total = count_lora_parameters(backbone)
print(f"After LoRA: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
# After LoRA: 295,936 / 33,554,432 (0.88%)
```

## See Also

- [`scmodelforge.training`](training.md) — Pretraining pipeline and data modules
- [`scmodelforge.models`](models.md) — Model architectures with `encode()` methods
- [`scmodelforge.eval`](eval.md) — Evaluation benchmarks and metrics
- [`scmodelforge.data`](data.md) — Data loading and preprocessing
- [`scmodelforge.config`](config.md) — YAML configuration reference
