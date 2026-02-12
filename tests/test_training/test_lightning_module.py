"""Tests for ScModelForgeLightningModule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from scmodelforge.config.schema import OptimizerConfig, SchedulerConfig
from scmodelforge.training.lightning_module import ScModelForgeLightningModule

if TYPE_CHECKING:
    from scmodelforge.models.transformer_encoder import TransformerEncoder


@pytest.fixture()
def _module(dummy_model: TransformerEncoder) -> ScModelForgeLightningModule:
    return ScModelForgeLightningModule(
        model=dummy_model,
        optimizer_config=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01),
        scheduler_config=SchedulerConfig(name="cosine_warmup", warmup_steps=2, total_steps=20),
    )


@pytest.fixture()
def _dummy_batch(tiny_model_config) -> dict[str, torch.Tensor]:
    """A minimal masked batch matching tiny_model_config."""
    torch.manual_seed(42)
    B, S = 2, 10
    vocab_size = tiny_model_config.vocab_size
    input_ids = torch.randint(4, vocab_size, (B, S))  # avoid special tokens 0-3
    attention_mask = torch.ones(B, S, dtype=torch.long)
    values = torch.rand(B, S)
    labels = torch.full((B, S), -100, dtype=torch.long)
    # Set labels at a few positions
    labels[0, 1] = input_ids[0, 1].item()
    labels[0, 3] = input_ids[0, 3].item()
    labels[1, 2] = input_ids[1, 2].item()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "values": values,
        "labels": labels,
    }


class TestScModelForgeLightningModule:
    """Tests for the Lightning module."""

    def test_forward_returns_model_output(self, _module, _dummy_batch) -> None:
        from scmodelforge.models.protocol import ModelOutput

        output = _module(_dummy_batch)
        assert isinstance(output, ModelOutput)
        assert output.loss is not None
        assert output.logits is not None

    def test_training_step_returns_loss(self, _module, _dummy_batch) -> None:
        loss = _module.training_step(_dummy_batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.requires_grad

    def test_validation_step_returns_none(self, _module, _dummy_batch) -> None:
        result = _module.validation_step(_dummy_batch, batch_idx=0)
        assert result is None

    def test_configure_optimizers_with_scheduler(self, _module) -> None:
        result = _module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["interval"] == "step"

    def test_configure_optimizers_without_scheduler(self, dummy_model) -> None:
        module = ScModelForgeLightningModule(
            model=dummy_model,
            optimizer_config=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01),
            scheduler_config=None,
        )
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" not in result

    def test_loss_is_finite(self, _module, _dummy_batch) -> None:
        loss = _module.training_step(_dummy_batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_perplexity_is_positive(self, _module, _dummy_batch) -> None:
        output = _module(_dummy_batch)
        perplexity = torch.exp(output.loss.clamp(max=100))
        assert perplexity > 0

    def test_hyperparameters_saved(self, _module) -> None:
        hp = _module.hparams
        assert "optimizer_config" in hp or hasattr(hp, "optimizer_config")
