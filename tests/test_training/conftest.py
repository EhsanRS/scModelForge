"""Shared fixtures for training module tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

from scmodelforge.config.schema import (
    DataConfig,
    MaskingConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessingConfig,
    SchedulerConfig,
    ScModelForgeConfig,
    TokenizerConfig,
    TrainingConfig,
)
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.models.transformer_encoder import TransformerEncoder


@pytest.fixture()
def tiny_adata() -> AnnData:
    """20 cells x 30 genes with sparse counts and 2 cell types."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 20, 30
    data = rng.poisson(lam=2, size=(n_cells, n_genes)).astype(np.float32)
    X = sp.csr_matrix(data)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    obs_data = {"cell_type": [f"type_{i % 2}" for i in range(n_cells)]}
    return AnnData(X=X, var={"gene_name": gene_names}, obs=obs_data)


@pytest.fixture()
def tiny_vocab(tiny_adata: AnnData) -> GeneVocab:
    """GeneVocab built from the tiny AnnData."""
    return GeneVocab.from_adata(tiny_adata)


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
        log_dir="/tmp/scmodelforge_test_logs",
        checkpoint_dir="/tmp/scmodelforge_test_checkpoints",
    )


@pytest.fixture()
def tiny_tokenizer_config() -> TokenizerConfig:
    """Tiny TokenizerConfig for rank_value tokenizer."""
    return TokenizerConfig(
        strategy="rank_value",
        max_genes=32,
        prepend_cls=True,
        masking=MaskingConfig(mask_ratio=0.15, random_replace_ratio=0.1, keep_ratio=0.1),
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
def tiny_full_config(
    tiny_data_config: DataConfig,
    tiny_tokenizer_config: TokenizerConfig,
    tiny_training_config: TrainingConfig,
) -> ScModelForgeConfig:
    """Complete ScModelForgeConfig for smoke tests."""
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
    )


@pytest.fixture()
def dummy_model(tiny_model_config: ModelConfig) -> TransformerEncoder:
    """A small TransformerEncoder for testing."""
    return TransformerEncoder.from_config(tiny_model_config)
