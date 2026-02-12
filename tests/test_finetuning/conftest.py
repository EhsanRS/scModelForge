"""Shared fixtures for fine-tuning module tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from anndata import AnnData

from scmodelforge.config.schema import (
    DataConfig,
    FinetuneConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessingConfig,
    SchedulerConfig,
    ScModelForgeConfig,
    TaskHeadConfig,
    TokenizerConfig,
    TrainingConfig,
)
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.models.transformer_encoder import TransformerEncoder


@pytest.fixture()
def tiny_adata_3types() -> AnnData:
    """30 cells x 20 genes with 3 cell types (type_0, type_1, type_2)."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 30, 20
    data = rng.poisson(lam=2, size=(n_cells, n_genes)).astype(np.float32)
    X = sp.csr_matrix(data)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    cell_types = [f"type_{i % 3}" for i in range(n_cells)]
    return AnnData(X=X, var={"gene_name": gene_names}, obs={"cell_type": cell_types})


@pytest.fixture()
def tiny_adata_regression() -> AnnData:
    """30 cells x 20 genes with continuous target values."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 30, 20
    data = rng.poisson(lam=2, size=(n_cells, n_genes)).astype(np.float32)
    X = sp.csr_matrix(data)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    targets = rng.standard_normal(n_cells).astype(np.float32)
    return AnnData(X=X, var={"gene_name": gene_names}, obs={"target": targets})


@pytest.fixture()
def tiny_vocab(tiny_adata_3types: AnnData) -> GeneVocab:
    """GeneVocab built from tiny_adata_3types."""
    return GeneVocab.from_adata(tiny_adata_3types)


@pytest.fixture()
def tiny_model_config(tiny_vocab: GeneVocab) -> ModelConfig:
    """Tiny ModelConfig for fast tests."""
    return ModelConfig(
        architecture="transformer_encoder",
        vocab_size=len(tiny_vocab),
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        dropout=0.0,
        max_seq_len=64,
        pooling="cls",
        activation="gelu",
        use_expression_values=True,
    )


@pytest.fixture()
def tiny_backbone(tiny_model_config: ModelConfig) -> TransformerEncoder:
    """A small TransformerEncoder backbone."""
    return TransformerEncoder.from_config(tiny_model_config)


@pytest.fixture()
def tiny_ft_config() -> FinetuneConfig:
    """FinetuneConfig with classification defaults."""
    return FinetuneConfig(
        checkpoint_path="",
        freeze_backbone=False,
        freeze_backbone_epochs=0,
        label_key="cell_type",
        head=TaskHeadConfig(task="classification", n_classes=3),
    )


@pytest.fixture()
def tiny_training_config() -> TrainingConfig:
    """Tiny TrainingConfig for fast CPU tests."""
    return TrainingConfig(
        batch_size=4,
        max_epochs=2,
        seed=42,
        precision="32-true",
        num_gpus=0,
        optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01),
        scheduler=SchedulerConfig(name="cosine_warmup", warmup_steps=2, total_steps=20),
        gradient_clip=1.0,
        gradient_accumulation=1,
        logger="csv",
        log_every_n_steps=1,
        num_workers=0,
        val_split=0.2,
        log_dir="/tmp/scmodelforge_ft_test_logs",
        checkpoint_dir="/tmp/scmodelforge_ft_test_checkpoints",
    )


@pytest.fixture()
def tiny_tokenizer_config() -> TokenizerConfig:
    """Tiny TokenizerConfig for rank_value tokenizer (no masking used)."""
    return TokenizerConfig(
        strategy="rank_value",
        max_genes=32,
        prepend_cls=True,
    )


@pytest.fixture()
def tiny_data_config() -> DataConfig:
    """Tiny DataConfig (paths left empty; adata injected directly)."""
    return DataConfig(
        preprocessing=PreprocessingConfig(
            normalize="library_size",
            target_sum=1e4,
            log1p=True,
        ),
        max_genes=32,
        num_workers=0,
    )


@pytest.fixture()
def tiny_full_ft_config(
    tiny_data_config: DataConfig,
    tiny_tokenizer_config: TokenizerConfig,
    tiny_training_config: TrainingConfig,
    tiny_ft_config: FinetuneConfig,
) -> ScModelForgeConfig:
    """Complete ScModelForgeConfig with finetune section."""
    return ScModelForgeConfig(
        data=tiny_data_config,
        tokenizer=tiny_tokenizer_config,
        model=ModelConfig(
            architecture="transformer_encoder",
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            ffn_dim=64,
            dropout=0.0,
            max_seq_len=64,
            pooling="cls",
            activation="gelu",
            use_expression_values=True,
        ),
        training=tiny_training_config,
        finetune=tiny_ft_config,
    )


@pytest.fixture()
def pretrained_checkpoint(tmp_path, tiny_adata_3types: AnnData) -> str:
    """Save a tiny model checkpoint and return the path."""
    vocab = GeneVocab.from_adata(tiny_adata_3types)
    config = ModelConfig(
        architecture="transformer_encoder",
        vocab_size=len(vocab),
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        dropout=0.0,
        max_seq_len=64,
        pooling="cls",
        activation="gelu",
        use_expression_values=True,
    )
    model = TransformerEncoder.from_config(config)
    ckpt_path = str(tmp_path / "pretrained.pt")
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


@pytest.fixture()
def pretrained_lightning_checkpoint(tmp_path, tiny_adata_3types: AnnData) -> str:
    """Save a Lightning-style checkpoint and return the path."""
    vocab = GeneVocab.from_adata(tiny_adata_3types)
    config = ModelConfig(
        architecture="transformer_encoder",
        vocab_size=len(vocab),
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        dropout=0.0,
        max_seq_len=64,
        pooling="cls",
        activation="gelu",
        use_expression_values=True,
    )
    model = TransformerEncoder.from_config(config)
    # Mimic Lightning checkpoint format
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
    ckpt_path = str(tmp_path / "lightning_pretrained.ckpt")
    torch.save({"state_dict": state_dict}, ckpt_path)
    return ckpt_path


@pytest.fixture()
def ft_batch(tiny_vocab: GeneVocab) -> dict[str, torch.Tensor]:
    """A small batch dict with task_labels for testing."""
    batch_size = 4
    seq_len = 10
    rng = torch.Generator().manual_seed(42)
    return {
        "input_ids": torch.randint(4, len(tiny_vocab), (batch_size, seq_len), generator=rng),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "values": torch.randn(batch_size, seq_len),
        "task_labels": torch.tensor([0, 1, 2, 0], dtype=torch.long),
    }
