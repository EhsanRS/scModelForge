"""Configuration schema for scModelForge.

All configuration is expressed as structured dataclasses that can be
instantiated from YAML files via OmegaConf. Each module will expand
its config section as it is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from pathlib import Path


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
class CensusConfig:
    """CELLxGENE Census data source configuration.

    Attributes
    ----------
    organism
        Census organism name (e.g. ``"Homo sapiens"``, ``"Mus musculus"``).
    census_version
        Census release version or ``"latest"``.
    obs_value_filter
        Raw SOMA ``obs_value_filter`` string. Takes precedence over
        *filters* if both are set.
    var_value_filter
        Raw SOMA ``var_value_filter`` string for gene filtering.
    filters
        Structured filter dict auto-converted to a SOMA filter string.
        Example: ``{"tissue": ["brain", "lung"], "is_primary_data": True}``.
    obs_columns
        Additional ``obs`` metadata columns to include in the AnnData.
    """

    organism: str = "Homo sapiens"
    census_version: str = "latest"
    obs_value_filter: str | None = None
    var_value_filter: str | None = None
    filters: dict[str, Any] | None = None
    obs_columns: list[str] | None = None


@dataclass
class MultiSpeciesConfig:
    """Multi-species gene vocabulary configuration.

    Attributes
    ----------
    enabled
        Whether to enable multi-species gene mapping.
    organisms
        List of organisms to include.
    canonical_organism
        Canonical namespace organism (genes from other organisms
        are mapped to this organism's gene names).
    include_one2many
        Whether to include one-to-many orthologs.
    ortholog_table
        Path to a custom ortholog TSV file. ``None`` uses the
        bundled Ensembl human-mouse table.
    """

    enabled: bool = False
    organisms: list[str] = field(default_factory=lambda: ["human", "mouse"])
    canonical_organism: str = "human"
    include_one2many: bool = False
    ortholog_table: str | None = None


@dataclass
class PerturbationConfig:
    """Perturbation data configuration.

    Attributes
    ----------
    enabled
        Whether to enable perturbation-aware data handling.
    perturbation_key
        Column in ``adata.obs`` containing perturbation labels.
        ``None`` triggers auto-detection.
    control_label
        Label used for control cells (case-insensitive).
    dose_key
        Column in ``adata.obs`` for dose values. ``None`` triggers
        auto-detection.
    dose_unit
        Unit of dose values (e.g. ``"uM"``).
    """

    enabled: bool = False
    perturbation_key: str | None = None
    control_label: str = "control"
    dose_key: str | None = None
    dose_unit: str | None = None


@dataclass
class ShardConfig:
    """Shard-based data loading configuration.

    Attributes
    ----------
    enabled
        Whether to use memory-mapped shards instead of in-memory data.
    shard_dir
        Path to the shard directory (must contain ``manifest.json``).
    shard_size
        Max cells per shard when converting with ``scmodelforge shard``.
    """

    enabled: bool = False
    shard_dir: str = ""
    shard_size: int = 500_000


@dataclass
class GeneSelectionConfig:
    """Batch-level gene selection configuration.

    Attributes
    ----------
    strategy
        Gene selection strategy: ``"all"`` (keep all genes),
        ``"most_expressed"`` (top-k by batch expression sum),
        or ``"random_expressed"`` (random subset of expressed genes).
    n_genes
        Number of genes to keep per batch. Required when
        ``strategy != "all"``.
    """

    strategy: str = "all"
    n_genes: int | None = None


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    source: str = "local"
    paths: list[str] = field(default_factory=list)
    gene_vocab: str = "human_protein_coding"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    max_genes: int = 2048
    num_workers: int = 4
    census: CensusConfig = field(default_factory=CensusConfig)
    multi_species: MultiSpeciesConfig = field(default_factory=MultiSpeciesConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    shards: ShardConfig = field(default_factory=ShardConfig)
    gene_selection: GeneSelectionConfig = field(default_factory=GeneSelectionConfig)
    streaming: bool = False
    streaming_chunk_size: int = 10_000
    streaming_shuffle_buffer: int = 10_000


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
    n_bins: int = 51
    binning_method: str = "uniform"
    embedding_path: str | None = None
    embedding_dim: int = 200
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
    # Autoregressive model options
    n_bins: int = 51
    gene_loss_weight: float = 1.0
    expression_loss_weight: float = 1.0
    # Masked autoencoder decoder options
    decoder_dim: int | None = None  # Defaults to hidden_dim // 2
    decoder_layers: int = 4
    decoder_heads: int | None = None  # Defaults to num_heads


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
class SamplingConfig:
    """Weighted sampling configuration.

    Attributes
    ----------
    strategy
        Sampling strategy: ``"random"`` (uniform shuffle) or
        ``"weighted"`` (inverse-frequency weighted by class label).
    label_key
        ``obs`` column used for class labels when ``strategy="weighted"``.
    replacement
        Whether to sample with replacement (standard for weighted sampling).
    curriculum_warmup_epochs
        Number of epochs over which to ramp from uniform to full
        inverse-frequency weights. ``0`` disables curriculum.
    """

    strategy: str = "random"
    label_key: str = "cell_type"
    replacement: bool = True
    curriculum_warmup_epochs: int = 0


@dataclass
class FSDPConfig:
    """Fully Sharded Data Parallel configuration.

    Attributes
    ----------
    sharding_strategy
        FSDP sharding strategy: ``"FULL_SHARD"``, ``"SHARD_GRAD_OP"``,
        ``"NO_SHARD"``, or ``"HYBRID_SHARD"``.
    cpu_offload
        Whether to offload parameters and gradients to CPU.
    activation_checkpointing
        Whether to enable activation checkpointing on
        ``nn.TransformerEncoderLayer`` modules.
    min_num_params
        Minimum number of parameters for auto-wrap policy.
    """

    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False
    activation_checkpointing: bool = False
    min_num_params: int = 1_000_000


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
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    fsdp: FSDPConfig | None = None


@dataclass
class EvalConfig:
    """Configuration for evaluation benchmarks."""

    every_n_epochs: int = 2
    batch_size: int = 256
    benchmarks: list[Any] = field(default_factory=list)


@dataclass
class TaskHeadConfig:
    """Configuration for a fine-tuning task head.

    Attributes
    ----------
    task
        Task type: ``"classification"`` or ``"regression"``.
    n_classes
        Number of output classes (classification only). Inferred from data
        if ``None``.
    output_dim
        Output dimension for regression tasks.
    hidden_dim
        Optional hidden layer dimension. ``None`` means a direct projection.
    dropout
        Dropout probability in the head.
    """

    task: str = "classification"
    n_classes: int | None = None
    output_dim: int = 1
    hidden_dim: int | None = None
    dropout: float = 0.1


@dataclass
class LoRAConfig:
    """LoRA adapter configuration.

    Attributes
    ----------
    enabled
        Whether to apply LoRA adapters.
    rank
        LoRA rank (r). Typical: 4, 8, 16.
    alpha
        LoRA scaling factor (alpha). Usually alpha = rank or 2*rank.
    dropout
        Dropout applied to LoRA layers.
    target_modules
        Module name patterns to apply LoRA to. ``None`` uses defaults
        (out_proj, linear1, linear2).
    bias
        Bias handling: ``"none"``, ``"all"``, or ``"lora_only"``.
    """

    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning a pretrained backbone.

    Attributes
    ----------
    checkpoint_path
        Path to the pretrained model checkpoint.
    freeze_backbone
        If ``True``, freeze all backbone parameters throughout training.
    freeze_backbone_epochs
        Unfreeze backbone after this many epochs (0 = no schedule).
    label_key
        Column in ``adata.obs`` containing task labels.
    head
        Task head configuration.
    backbone_lr
        Discriminative learning rate for the backbone. ``None`` uses the
        global optimizer LR.
    head_lr
        Discriminative learning rate for the head. ``None`` uses the
        global optimizer LR.
    lora
        LoRA adapter configuration.
    """

    checkpoint_path: str = ""
    freeze_backbone: bool = False
    freeze_backbone_epochs: int = 0
    label_key: str = "cell_type"
    head: TaskHeadConfig = field(default_factory=TaskHeadConfig)
    backbone_lr: float | None = None
    head_lr: float | None = None
    lora: LoRAConfig = field(default_factory=LoRAConfig)


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
    finetune: FinetuneConfig | None = None


def save_config(config: ScModelForgeConfig, path: str | Path) -> None:
    """Save a ScModelForgeConfig to a YAML file.

    Parameters
    ----------
    config
        Configuration object to save.
    path
        Output YAML file path.
    """
    from pathlib import Path as _Path

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    structured = OmegaConf.structured(config)
    OmegaConf.save(structured, path)


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
