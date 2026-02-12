"""Shared fixtures for model tests."""

from __future__ import annotations

import pytest
import torch

from scmodelforge.config.schema import ModelConfig


@pytest.fixture()
def tiny_config() -> ModelConfig:
    """A tiny ModelConfig for fast tests."""
    return ModelConfig(
        architecture="transformer_encoder",
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        dropout=0.0,
        max_seq_len=32,
        pooling="cls",
        activation="gelu",
        use_expression_values=True,
    )


@pytest.fixture()
def dummy_batch() -> dict[str, torch.Tensor]:
    """A minimal batch with B=2, S=10."""
    torch.manual_seed(42)
    batch_size, seq_len, vocab_size = 2, 10, 100
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # Make last 2 positions padding in 2nd sample
    attention_mask[1, -2:] = 0
    input_ids[1, -2:] = 0
    values = torch.rand(batch_size, seq_len)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    # Set labels at 3 masked positions per sample
    labels[0, 1] = input_ids[0, 1].item()
    labels[0, 3] = input_ids[0, 3].item()
    labels[1, 2] = input_ids[1, 2].item()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "values": values,
        "labels": labels,
    }


# --- Autoregressive transformer fixtures ---


@pytest.fixture()
def ar_config() -> ModelConfig:
    """A tiny ModelConfig for AutoregressiveTransformer tests."""
    return ModelConfig(
        architecture="autoregressive_transformer",
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        dropout=0.0,
        max_seq_len=32,
        pooling="cls",
        activation="gelu",
        use_expression_values=True,
        n_bins=51,
        gene_loss_weight=1.0,
        expression_loss_weight=1.0,
    )


@pytest.fixture()
def ar_batch() -> dict[str, torch.Tensor]:
    """A batch with bin_ids for autoregressive model testing."""
    torch.manual_seed(42)
    batch_size, seq_len, vocab_size, n_bins = 2, 10, 100, 51
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[1, -2:] = 0
    input_ids[1, -2:] = 0
    values = torch.rand(batch_size, seq_len)
    bin_ids = torch.randint(0, n_bins, (batch_size, seq_len))
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[0, 1] = input_ids[0, 1].item()
    labels[0, 3] = input_ids[0, 3].item()
    labels[0, 5] = input_ids[0, 5].item()
    labels[1, 2] = input_ids[1, 2].item()
    labels[1, 4] = input_ids[1, 4].item()
    labels[1, 6] = input_ids[1, 6].item()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "values": values,
        "labels": labels,
        "bin_ids": bin_ids,
    }


# --- Masked autoencoder fixtures ---


@pytest.fixture()
def mae_config() -> ModelConfig:
    """A tiny ModelConfig for MaskedAutoencoder tests."""
    return ModelConfig(
        architecture="masked_autoencoder",
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        dropout=0.0,
        max_seq_len=32,
        pooling="mean",
        activation="gelu",
        use_expression_values=True,
        decoder_dim=32,
        decoder_layers=1,
        decoder_heads=2,
    )


@pytest.fixture()
def mae_batch() -> dict[str, torch.Tensor]:
    """A batch with masked_positions for MAE testing."""
    torch.manual_seed(42)
    batch_size, seq_len, vocab_size = 2, 10, 100
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[1, -2:] = 0
    input_ids[1, -2:] = 0
    values = torch.rand(batch_size, seq_len)
    masked_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    masked_positions[0, 1] = True
    masked_positions[0, 3] = True
    masked_positions[0, 5] = True
    masked_positions[1, 2] = True
    masked_positions[1, 4] = True
    masked_positions[1, 6] = True
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[masked_positions] = 1  # Any non -100 value
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "values": values,
        "labels": labels,
        "masked_positions": masked_positions,
    }
