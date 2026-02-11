# Stage 4: scModelForge.training

## Overview

The training module provides a config-driven training loop built on PyTorch Lightning that handles distributed training, mixed precision, gradient accumulation, checkpointing, and logging. It orchestrates the data, tokenizer, and model modules into a complete training pipeline.

**Core responsibility:** Take a YAML config → produce a trained model checkpoint with logged metrics.

**Dependencies:** Stage 0 (scaffolding), Stage 1 (data), Stage 2 (tokenizers), Stage 3 (models)
**Blocks:** Stage 5 (eval callbacks are integrated into training)

---

## Phase 1: Foundation (Months 1–3)

### Goals
- Config-driven training: `scmodelforge train --config config.yaml` does everything
- PyTorch Lightning Trainer with DDP, bf16 mixed precision
- WandB and TensorBoard logging
- Checkpointing with resume capability
- Train on 4 GPUs with linear scaling efficiency >85%

### Architecture

```
YAML config
     │
     ▼
┌──────────────────┐
│ ConfigParser     │  ← validates and resolves config
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ TrainingPipeline │  ← orchestrator: wires data → tokenizer → model → trainer
└──────┬───────────┘
       │
       ├──→ CellDataModule (Lightning DataModule)
       ├──→ ScModelForgeLightningModule (Lightning Module)
       └──→ pl.Trainer (configured from YAML)
              │
              ├── Callbacks: checkpointing, LR monitor, eval
              ├── Logger: WandB / TensorBoard
              └── Strategy: DDP / FSDP
```

### File Structure

```
src/scmodelforge/training/
├── __init__.py              # Public API: train, TrainingPipeline
├── pipeline.py              # TrainingPipeline orchestrator
├── lightning_module.py      # ScModelForgeLightningModule
├── data_module.py           # CellDataModule (Lightning DataModule)
├── callbacks.py             # Custom callbacks (eval, logging, etc.)
├── optimizers.py            # Optimizer and scheduler factories
├── config_parser.py         # YAML → validated config objects
└── _utils.py                # Seed setting, environment detection
```

### Key Classes and Interfaces

#### `CellDataModule` (Lightning DataModule)

Wraps the data module's `CellDataset` and `CellDataLoader` into Lightning's `DataModule` interface.

```python
class CellDataModule(pl.LightningDataModule):
    """Lightning DataModule that wires scModelForge.data components."""

    def __init__(
        self,
        data_config: DataConfig,
        tokenizer_config: TokenizerConfig,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()
        self.data_config = data_config
        self.tokenizer_config = tokenizer_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

    def setup(self, stage: str | None = None):
        """Load data, build vocab, create tokenizer, split train/val."""
        # 1. Load AnnData from config paths
        # 2. Build or load GeneVocab
        # 3. Create tokenizer from config
        # 4. Apply preprocessing
        # 5. Split into train/val
        # 6. Create CellDataset instances
        ...

    def train_dataloader(self) -> DataLoader:
        return CellDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return CellDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @property
    def gene_vocab(self) -> GeneVocab:
        return self._gene_vocab

    @property
    def tokenizer(self) -> BaseTokenizer:
        return self._tokenizer
```

#### `ScModelForgeLightningModule`

Wraps any `ScModelForgeModel` into a Lightning Module with training/validation logic.

```python
class ScModelForgeLightningModule(pl.LightningModule):
    """Lightning module that wraps a ScModelForgeModel for training."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        masking: MaskingStrategy,
        optimizer_config: dict,
        scheduler_config: dict | None = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.masking = masking
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            values=batch.get("values"),
            labels=batch.get("labels"),
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # 1. Tokenize batch (if not already tokenized in data pipeline)
        # 2. Apply masking for pretraining
        # 3. Forward pass
        # 4. Log metrics
        masked_batch = self._apply_masking(batch)
        output = self.forward(masked_batch)

        self.log("train/loss", output.loss, prog_bar=True, sync_dist=True)
        self.log("train/perplexity", torch.exp(output.loss), sync_dist=True)

        return output.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        masked_batch = self._apply_masking(batch)
        output = self.forward(masked_batch)

        self.log("val/loss", output.loss, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", torch.exp(output.loss), sync_dist=True)

    def _apply_masking(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply masking strategy to batch for pretraining."""
        # Create labels from original input_ids
        # Replace masked positions with mask token
        # Return modified batch with labels
        ...

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.optimizer_config)
        if self.scheduler_config:
            scheduler = build_scheduler(optimizer, self.scheduler_config)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
```

#### `TrainingPipeline`

The top-level orchestrator that reads config and runs everything.

```python
class TrainingPipeline:
    """Orchestrates the full training pipeline from config."""

    def __init__(self, config: ScModelForgeConfig):
        self.config = config

    def run(self):
        """Execute the full training pipeline."""
        # 1. Set seed for reproducibility
        pl.seed_everything(self.config.training.seed)

        # 2. Create data module
        data_module = CellDataModule(
            data_config=self.config.data,
            tokenizer_config=self.config.tokenizer,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
        )

        # 3. Setup data to get vocab size
        data_module.setup()

        # 4. Create model
        model_config = self.config.model
        model_config.vocab_size = len(data_module.gene_vocab)
        model = get_model(model_config.architecture, model_config)

        # 5. Create masking strategy
        masking = MaskingStrategy(
            mask_ratio=self.config.model.mask_ratio,
            mask_token_id=data_module.gene_vocab.mask_token_id,
        )

        # 6. Create Lightning module
        lightning_module = ScModelForgeLightningModule(
            model=model,
            tokenizer=data_module.tokenizer,
            masking=masking,
            optimizer_config=self.config.training.optimizer,
            scheduler_config=self.config.training.scheduler,
        )

        # 7. Create callbacks
        callbacks = self._build_callbacks()

        # 8. Create logger
        logger = self._build_logger()

        # 9. Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            accelerator="auto",
            devices=self.config.training.num_gpus or "auto",
            strategy=self.config.training.strategy,
            precision=self.config.training.precision,
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=self.config.training.gradient_clip,
            accumulate_grad_batches=self.config.training.gradient_accumulation,
            log_every_n_steps=self.config.training.log_every_n_steps,
        )

        # 10. Train
        trainer.fit(lightning_module, datamodule=data_module)

        return trainer

    def _build_callbacks(self) -> list[pl.Callback]:
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=self.config.training.checkpoint_dir,
                filename="scmodelforge-{epoch:02d}-{val/loss:.4f}",
                save_top_k=3,
                monitor="val/loss",
                mode="min",
                save_last=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.RichProgressBar(),
        ]
        # Add eval callback if configured (from Stage 5)
        if self.config.eval.benchmarks:
            from scmodelforge.eval import EvalCallback
            callbacks.append(EvalCallback(self.config.eval))
        return callbacks

    def _build_logger(self):
        logger_type = self.config.training.logger
        if logger_type == "wandb":
            return pl.loggers.WandbLogger(
                project=self.config.training.wandb_project or "scmodelforge",
                name=self.config.training.run_name,
            )
        elif logger_type == "tensorboard":
            return pl.loggers.TensorBoardLogger(
                save_dir=self.config.training.log_dir or "logs",
                name=self.config.training.run_name,
            )
        return True  # Default Lightning logger
```

#### Optimizer and Scheduler Factories

```python
def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer from config dict."""
    name = config.get("name", "adamw")
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 0.01)

    # Separate weight decay for different parameter groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "layer_norm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
    elif name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, config: dict):
    """Build LR scheduler from config dict."""
    name = config.get("name", "cosine")
    warmup_steps = config.get("warmup_steps", 1000)
    total_steps = config.get("total_steps", 100000)

    if name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return {
            "scheduler": CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
            "interval": "step",
        }
    elif name == "linear":
        # Linear warmup then linear decay
        ...
    elif name == "cosine_warmup":
        # Linear warmup then cosine decay
        ...
```

### CLI Integration

```python
# In cli.py (extending Stage 0 skeleton)
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from checkpoint")
def train(config: str, resume: str | None):
    """Train a model from a YAML config."""
    from scmodelforge.training import TrainingPipeline
    from scmodelforge.config import load_config

    cfg = load_config(config)
    if resume:
        cfg.training.resume_from = resume

    pipeline = TrainingPipeline(cfg)
    pipeline.run()
```

### Config Integration

```yaml
training:
  # Core
  batch_size: 64
  max_epochs: 10
  seed: 42

  # Distributed
  strategy: ddp             # ddp | fsdp | auto
  num_gpus: 4
  precision: bf16-mixed     # 32 | 16-mixed | bf16-mixed

  # Optimizer
  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 0.01

  # Scheduler
  scheduler:
    name: cosine_warmup
    warmup_steps: 2000
    total_steps: 100000

  # Gradient
  gradient_clip: 1.0
  gradient_accumulation: 1

  # Logging
  logger: wandb             # wandb | tensorboard
  wandb_project: scmodelforge
  run_name: geneformer-v1
  log_every_n_steps: 50

  # Checkpointing
  checkpoint_dir: ./checkpoints
  save_top_k: 3

  # Data
  num_workers: 4
  val_split: 0.05

  # Resume
  resume_from: null          # Path to checkpoint
```

### Tests (Phase 1)

- `test_config_parser.py`: Load YAML config, validate, resolve defaults, error on invalid.
- `test_data_module.py`: CellDataModule setup, train/val split, dataloader creation.
- `test_lightning_module.py`: Training step, validation step, optimizer configuration, loss computation.
- `test_pipeline.py`: Full pipeline smoke test with tiny model on synthetic data (CPU only, 2 epochs).
- `test_optimizers.py`: Optimizer factory, scheduler factory, parameter group separation.
- `test_checkpoint.py`: Save and resume from checkpoint, verify training continues correctly.
- `test_ddp.py` (`@pytest.mark.gpu`): Multi-GPU training test (if available).

### Performance Targets

| Metric | Target |
|---|---|
| DDP scaling efficiency (4 GPU) | >85% |
| Checkpoint save/load time | <30s for 100M param model |
| Training throughput | >50k cells/sec (4x A100) |
| Memory per GPU (base model) | <20 GB (bf16) |

---

## Phase 2: Breadth (Months 4–6)

### Fine-tuning API

Add support for fine-tuning pretrained models on downstream tasks.

```python
@main.command()
@click.option("--base-model", required=True, help="Path or hub ID")
@click.option("--data", required=True, type=click.Path(exists=True))
@click.option("--task", required=True, type=click.Choice(["cell_type", "perturbation", "batch_integration"]))
@click.option("--eval", default=None)
def finetune(base_model: str, data: str, task: str, eval: str | None):
    """Fine-tune a pretrained model on downstream data."""
    ...
```

### LoRA / Adapter Support

- Integration with `peft` library for parameter-efficient fine-tuning.
- Configurable via YAML:
  ```yaml
  training:
    finetune:
      method: lora        # lora | adapter | full
      lora_r: 8
      lora_alpha: 16
      target_modules: [query, value]
  ```

### Fine-tuning Recipes

Pre-configured YAML recipes for common tasks:
- `configs/recipes/cell_type_annotation.yaml`
- `configs/recipes/perturbation_prediction.yaml`
- `configs/recipes/batch_integration.yaml`

Each recipe specifies appropriate:
- Task-specific head (classification, regression)
- Loss function
- Evaluation metrics
- Hyperparameters

### Additional Phase 2 Files

```
src/scmodelforge/training/
├── ...existing...
├── finetuning.py            # Fine-tuning pipeline and task heads
├── peft_integration.py      # LoRA/adapter wrappers
└── recipes/
    ├── __init__.py
    ├── cell_type.py
    ├── perturbation.py
    └── batch_integration.py
```

---

## Phase 3: Community & Scale (Months 7–12)

### FSDP for Large Models

- FSDP strategy configuration for models >1B parameters.
- Automatic sharding policy for transformer layers.
- Activation checkpointing for memory-constrained training.
- Mixed-precision with FSDP (bf16 compute, fp32 reduce).

### Advanced Training Features

- **Curriculum learning:** Start with easy examples, progress to harder ones.
- **Multi-task training:** Train on multiple pretraining objectives simultaneously.
- **Distributed data loading:** Coordinated loading across nodes for large datasets.

### Training Dashboard

- Rich real-time metrics dashboard (beyond WandB).
- Biological metrics alongside training loss.
- Cell embedding visualizations (UMAP projections) during training.

---

## Checklist

### Phase 1
- [ ] Implement `CellDataModule` (Lightning DataModule)
- [ ] Implement `ScModelForgeLightningModule` (Lightning Module)
- [ ] Implement `TrainingPipeline` orchestrator
- [ ] Implement optimizer factory with param group separation
- [ ] Implement scheduler factory (cosine, linear, cosine_warmup)
- [ ] Implement config parser (YAML → validated dataclasses)
- [ ] Wire up CLI `train` command
- [ ] Add WandB and TensorBoard logging
- [ ] Add checkpointing with resume
- [ ] Write comprehensive tests (including smoke test)
- [ ] Benchmark DDP scaling on multi-GPU
- [ ] Write docstrings and API documentation

### Phase 2
- [ ] Implement fine-tuning pipeline
- [ ] Add LoRA/adapter support via peft
- [ ] Create fine-tuning recipes (cell_type, perturbation, batch_integration)
- [ ] Wire up CLI `finetune` command

### Phase 3
- [ ] FSDP strategy implementation
- [ ] Activation checkpointing
- [ ] Multi-task training support
- [ ] Training dashboard
