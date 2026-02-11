"""Configuration schema for scModelForge.

All configuration is expressed as structured dataclasses that can be
instantiated from YAML files via OmegaConf. Each module will expand
its config section as it is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Per-module configs (stubs — expanded in their respective stages)
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Preprocessing options applied to raw expression data."""

    normalize: str | None = "library_size"
    target_sum: float | None = 1e4
    hvg_selection: int | None = None
    log1p: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    source: str = "local"
    paths: list[str] = field(default_factory=list)
    gene_vocab: str = "human_protein_coding"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    max_genes: int = 2048
    num_workers: int = 4


@dataclass
class MaskingConfig:
    """Masking strategy for pretraining."""

    mask_ratio: float = 0.15
    random_replace_ratio: float = 0.1
    keep_ratio: float = 0.1


@dataclass
class TokenizerConfig:
    """Configuration for tokenization strategy."""

    strategy: str = "rank_value"
    max_genes: int = 2048
    gene_vocab: str = "human_protein_coding"
    prepend_cls: bool = True
    masking: MaskingConfig = field(default_factory=MaskingConfig)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    architecture: str = "transformer_encoder"
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_dim: int | None = None
    dropout: float = 0.1
    max_seq_len: int = 2048
    pooling: str = "cls"
    activation: str = "gelu"
    use_expression_values: bool = True
    pretraining_task: str = "masked_gene_prediction"
    mask_ratio: float = 0.15
    vocab_size: int | None = None  # Inferred from gene vocab at runtime


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    name: str = "cosine_warmup"
    warmup_steps: int = 2000
    total_steps: int = 100_000


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    batch_size: int = 64
    max_epochs: int = 10
    seed: int = 42
    strategy: str = "ddp"
    num_gpus: int | None = None
    precision: str = "bf16-mixed"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig | None = field(default_factory=SchedulerConfig)
    gradient_clip: float = 1.0
    gradient_accumulation: int = 1
    logger: str = "wandb"
    wandb_project: str = "scmodelforge"
    run_name: str | None = None
    log_dir: str = "logs"
    log_every_n_steps: int = 50
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    num_workers: int = 4
    val_split: float = 0.05
    resume_from: str | None = None


@dataclass
class EvalConfig:
    """Configuration for evaluation benchmarks."""

    every_n_epochs: int = 2
    benchmarks: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class ScModelForgeConfig:
    """Top-level configuration combining all modules."""

    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str | Path) -> ScModelForgeConfig:
    """Load a ScModelForgeConfig from a YAML file.

    Parameters
    ----------
    path
        Path to a YAML configuration file.

    Returns
    -------
    ScModelForgeConfig
        Validated configuration object.
    """
    schema = OmegaConf.structured(ScModelForgeConfig)
    file_conf = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, file_conf)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]
