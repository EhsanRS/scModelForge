# Changelog

## 0.1.0 (Unreleased)

Initial release of scModelForge, a complete toolkit for pretraining and fine-tuning single-cell foundation models.

### Bug Fixes

- Fix dependency: use `lightning` package instead of `pytorch-lightning` (the codebase imports `lightning.pytorch`)
- Fix tokenizer config propagation: strategy-specific `TokenizerConfig` fields (`n_bins`, `binning_method`, `embedding_path`, `embedding_dim`) are now passed to tokenizer constructors in training, fine-tuning, and CLI benchmark paths via new `build_tokenizer_kwargs()` helper
- Fix eval benchmark parser: `EvalHarness.from_config()` now supports the nested `{name, dataset, params}` spec format used in shipped configs, strips informational `dataset` key, and unpacks `params` as constructor kwargs
- Fix perturbation recipe: corrected parameter names (`control_key` → `control_label`, `n_top_degs` → `n_top_genes`) to match `PerturbationBenchmark` constructor
- Wire `AssessmentCallback` into `TrainingPipeline`: `eval.benchmarks` config now activates in-training evaluation via `_build_callbacks(data_module)`, with `CellDataModule.adata` property providing the validation AnnData
- Fix gradual unfreezing: backbone parameters are now always included in optimizer groups (even when frozen), so `freeze_backbone_epochs` correctly enables training of backbone params after unfreezing instead of silently leaving them out of the optimizer
- Fix weighted sampling curriculum: add `SamplerEpochCallback` that calls `sampler.set_epoch()` each epoch, so `curriculum_warmup_epochs` actually progresses instead of staying at epoch 0 behavior
- Fix `DistributedShardSampler(drop_last=True)`: now computes a globally consistent minimum per-rank cell count across all ranks and truncates to that value, ensuring equal step counts for DDP/FSDP synchronization with imbalanced shard sizes
- Fix streaming mode memory: `CellDataModule` with `streaming=True` no longer materializes the full AnnData during setup. Vocab is built by scanning `var_names` in backed mode (metadata only), and validation uses a bounded subset (capped at 10k cells) from the first file

### Stage 0: Scaffolding

- Project structure with src-layout and six core modules: `data`, `tokenizers`, `models`, `training`, `eval`, `finetuning`
- Configuration schema with YAML loading via OmegaConf, structured dataclasses for type safety
- CLI with `train`, `finetune`, and `benchmark` subcommands
- CI/CD via GitHub Actions: linting (ruff), testing (pytest), type checking (mypy), release automation
- Pre-commit hooks for code quality enforcement
- Sphinx documentation with MyST Markdown, sphinx-book-theme, autodoc2 for API docs
- Contributing guide, README, and changelog skeleton

### Stage 1: Data Module

- `GeneVocab` class for gene identifier mapping (Ensembl, symbols) with predefined vocabularies
- `PreprocessingPipeline` for normalization, HVG selection, and log transformation
- `AnnDataStore` for efficient data storage and retrieval
- `CellDataset` PyTorch dataset for single-cell expression data
- `CellDataLoader` with custom collation for sparse data
- Preprocessing utilities: normalization, HVG filtering, log1p transformation
- 59 comprehensive unit tests for data module
- CELLxGENE Census integration: remote data loading with `load_census_adata()`, filter building, `CensusConfig`
- Shared `load_adata()` dispatcher for local and Census data sources
- 25 additional tests for Census integration (84 total data module tests)

### Stage 2: Tokenizers Module

#### Phase 1: Core Tokenizers

- `TokenizedCell` and `MaskedTokenizedCell` dataclasses for tokenized representations
- `BaseTokenizer` abstract base class with `tokenize()`, `tokenize_batch()`, and `_collate()` methods
- `RankValueTokenizer` for Geneformer-style rank-value encoding
- `MaskingStrategy` for BERT-style 80/10/10 masking with CLS/PAD token protection
- Tokenizer registry: `register_tokenizer()`, `get_tokenizer()`, `list_tokenizers()`
- Utility functions: `ensure_tensor()`, `rank_genes_by_expression()`
- 69 comprehensive tests for base tokenizers and masking

#### Phase 2: Advanced Tokenizers

- `BinnedExpressionTokenizer` for scGPT-style discrete binning (uniform, quantile, adaptive)
- `ContinuousProjectionTokenizer` for TranscriptFormer-style continuous embeddings
- Binning utilities: `compute_bin_edges()`, `digitize_expression()`
- `TokenizedCell.bin_ids` field for expression bin storage (backward-compatible)
- `TokenizerConfig` extended with `n_bins` and `binning_method` parameters
- 49 additional tests for binned and continuous tokenizers (118 total tokenizer tests)

### Stage 3: Models Module

#### Phase 1: Transformer Encoder

- `ModelOutput` frozen dataclass for standardized model outputs
- `TransformerEncoder` BERT-style pre-norm encoder with masked gene prediction
- Model registry: `register_model()`, `get_model()`, `list_models()`
- Model components:
  - `GeneExpressionEmbedding` for gene ID and expression value embeddings
  - `MaskedGenePredictionHead` for gene classification
  - `ExpressionPredictionHead` for expression regression
  - `cls_pool()` and `mean_pool()` for sequence pooling
- Model utilities: `init_weights()`, `count_parameters()`
- Uses PyTorch `nn.TransformerEncoder` with `batch_first=True` and `norm_first=True`
- 36 comprehensive tests for transformer encoder and components

#### Phase 2: Advanced Architectures

- `AutoregressiveTransformer` GPT-style causal decoder for sequential generation
  - Dual prediction heads: gene ID classification + expression bin classification
  - Causal attention masking via `generate_causal_mask()`
  - Weighted loss: `gene_loss_weight * CE_gene + expression_loss_weight * CE_bin`
  - `encode()` method uses full bidirectional attention for embedding extraction
- `MaskedAutoencoder` asymmetric encoder-decoder architecture
  - Encoder processes only unmasked tokens for efficiency
  - Decoder reconstructs full sequence with learnable mask token embeddings
  - MSE loss at masked positions for expression reconstruction
- New components:
  - `BinPredictionHead` for expression bin classification
  - `generate_causal_mask()` for autoregressive attention
- `ModelConfig` extended with autoregressive and MAE parameters
- Lightning module updated to pass `**kwargs` for model-specific inputs
- 72 additional tests for autoregressive and MAE models (108 total model tests)

### Stage 4: Training Module

#### Phase 1: Pretraining Pipeline

- Training utilities: `get_environment_info()`, `log_training_config()`
- Optimizer builder: `build_optimizer()` with parameter groups and weight decay exclusions (AdamW, Adam)
- Scheduler builder: `build_scheduler()` with cosine warmup, cosine, and linear schedules
- `TokenizedCellDataset` wrapping `CellDataset` with tokenization and masking
- `CellDataModule` Lightning data module with train/val splitting
- `ScModelForgeLightningModule` for training and validation steps
- Training callbacks: `TrainingMetricsLogger` (cells/sec, step time, epoch time), `GradientNormLogger`
- `TrainingPipeline` end-to-end orchestration: seed setting, data loading, model construction, trainer initialization
- CLI `train` command wired with config loading and checkpoint resumption
- Design choice: tokenization and masking in `TokenizedCellDataset.__getitem__()`, collation via `tokenizer._collate()`
- Both train and validation apply masking (standard BERT pretraining approach)
- 60 comprehensive tests for training module

#### Phase 2: Fine-tuning Pipeline

- Fine-tuning task heads: `ClassificationHead`, `RegressionHead`, `build_task_head()`
- `FineTuneModel` wrapper: pretrained backbone + task head, freeze/unfreeze controls
- `load_pretrained_backbone()` utility for Lightning and raw checkpoints
- `LabelEncoder` for categorical label encoding
- `_LabeledTokenizedDataset` for labeled data with no masking
- `FineTuneDataModule` with stratified train/val splitting and custom collation
- `FineTuneLightningModule` with discriminative learning rates and gradual unfreezing
- `FineTunePipeline` end-to-end fine-tuning orchestration
- Configuration: `TaskHeadConfig`, `FinetuneConfig` with discriminative LR and freeze schedule
- CLI `finetune` command wired with config and checkpoint loading
- 54 comprehensive tests for fine-tuning pipeline
- LoRA adapter support:
  - `apply_lora()`, `has_lora()`, `save_lora_weights()`, `load_lora_weights()`, `count_lora_parameters()`
  - `LoRAConfig` dataclass with rank, alpha, dropout, target_modules, bias parameters
  - Default target modules: `["out_proj", "linear1", "linear2"]` (plain names, not regex)
  - `FineTuneModel.has_lora` property
  - Pipeline integration: LoRA applied between checkpoint load and head construction
  - `peft>=0.6` as optional dependency (`[peft]` extras)
- 24 additional tests for LoRA integration (78 total fine-tuning tests)

### Stage 5: Evaluation Module

#### Phase 1: Benchmarking Framework

- `BenchmarkResult` dataclass and `BaseBenchmark` abstract base class
- Benchmark registry: `register_benchmark()`, `get_benchmark()`, `list_benchmarks()`
- `extract_embeddings()` utility for model → numpy embeddings via datasets
- `LinearProbeBenchmark` for cell type classification with sklearn logistic regression (accuracy, F1)
- `EmbeddingQualityBenchmark` for scIB metrics: NMI, ARI, ASW (bio/batch), overall score (0.6 × bio + 0.4 × batch)
- `EvalHarness` orchestrator with `run()` and `run_on_embeddings()` methods, `from_config()` factory
- `AssessmentCallback` Lightning callback for periodic evaluation during training
- CLI `benchmark` command for standalone evaluation
- `EvalConfig.batch_size` added to schema
- 54 comprehensive tests for evaluation module (53 passing + 1 skipped when scib not installed)

### Summary

- 516 passing tests (+ 1 skipped: scIB tests when scib not installed)
- Test breakdown: 14 config, 84 data (59 base + 25 census), 118 tokenizers, 108 models, 60 training, 54 evaluation, 78 fine-tuning (54 base + 24 LoRA)
- Complete pretraining pipeline: config → data → tokenization → model → training → evaluation
- Full fine-tuning support with LoRA, discriminative LR, and gradual unfreezing
- Three tokenization strategies: rank-value, binned expression, continuous projection
- Three model architectures: transformer encoder, autoregressive transformer, masked autoencoder
- Two evaluation benchmarks: linear probe, embedding quality (scIB)
- CELLxGENE Census integration for remote data access
- PyTorch Lightning integration with distributed training, mixed precision, and automatic checkpointing
- Config-driven design with type-safe YAML loading via OmegaConf
- Comprehensive CLI with train, finetune, and benchmark commands
