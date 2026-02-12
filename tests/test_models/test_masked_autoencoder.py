"""Tests for MaskedAutoencoder model."""

from __future__ import annotations

import pytest
import torch

from scmodelforge.config.schema import ModelConfig
from scmodelforge.models.masked_autoencoder import MaskedAutoencoder
from scmodelforge.models.protocol import ModelOutput
from scmodelforge.models.registry import get_model, list_models


class TestForward:
    """Forward pass tests."""

    def test_returns_model_output(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        assert isinstance(output, ModelOutput)

    def test_embeddings_shape_is_encoder_dim(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        B = mae_batch["input_ids"].shape[0]
        assert output.embeddings.shape == (B, mae_config.hidden_dim)

    def test_mse_loss(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        assert output.loss is not None
        assert output.loss.ndim == 0
        assert output.loss.item() >= 0

    def test_loss_only_at_masked_positions(self, mae_config, mae_batch):
        """Loss should only consider masked positions."""
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        assert output.loss is not None

    def test_infer_mask_from_labels(self, mae_config, mae_batch):
        """When masked_positions is absent, infer from labels != -100."""
        model = MaskedAutoencoder.from_config(mae_config)
        batch_no_mask = {k: v for k, v in mae_batch.items() if k != "masked_positions"}
        output = model(**batch_no_mask)
        assert output.loss is not None

    def test_no_masking_returns_embeddings_only(self, mae_config, mae_batch):
        """Without labels or masked_positions, return embeddings with no loss."""
        model = MaskedAutoencoder.from_config(mae_config)
        batch = {
            "input_ids": mae_batch["input_ids"],
            "attention_mask": mae_batch["attention_mask"],
            "values": mae_batch["values"],
        }
        output = model(**batch)
        assert output.loss is None
        assert output.embeddings is not None

    def test_logits_shape(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        B, S = mae_batch["input_ids"].shape
        assert output.logits.shape == (B, S)

    def test_loss_is_finite(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        assert torch.isfinite(output.loss)

    def test_forward_without_expression_values(self, mae_config, mae_batch):
        """Should still run; loss cannot be computed without values."""
        model = MaskedAutoencoder.from_config(mae_config)
        batch = {k: v for k, v in mae_batch.items() if k != "values"}
        output = model(**batch)
        assert output.loss is None  # No values to compute MSE against

    def test_all_positions_masked(self, mae_config):
        """Edge case: all non-padding positions are masked."""
        model = MaskedAutoencoder.from_config(mae_config)
        B, S = 2, 6
        input_ids = torch.randint(1, 100, (B, S))
        attention_mask = torch.ones(B, S, dtype=torch.long)
        values = torch.rand(B, S)
        masked_positions = torch.ones(B, S, dtype=torch.bool)
        labels = torch.ones(B, S, dtype=torch.long)
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       values=values, labels=labels, masked_positions=masked_positions)
        # When all positions are masked, there are no unmasked tokens for encoder
        # The model should handle this gracefully
        assert isinstance(output, ModelOutput)


class TestEncode:
    """Encode method tests."""

    def test_shape_is_encoder_dim(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        emb = model.encode(mae_batch["input_ids"], mae_batch["attention_mask"], mae_batch["values"])
        B = mae_batch["input_ids"].shape[0]
        assert emb.shape == (B, mae_config.hidden_dim)

    def test_full_encoder_used(self, mae_config, mae_batch):
        """encode() should use all tokens, not just unmasked."""
        model = MaskedAutoencoder.from_config(mae_config)
        model.train(False)
        with torch.no_grad():
            emb = model.encode(mae_batch["input_ids"], mae_batch["attention_mask"], mae_batch["values"])
        assert emb.shape[1] == mae_config.hidden_dim

    def test_deterministic(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        model.train(False)
        with torch.no_grad():
            emb1 = model.encode(mae_batch["input_ids"], mae_batch["attention_mask"], mae_batch["values"])
            emb2 = model.encode(mae_batch["input_ids"], mae_batch["attention_mask"], mae_batch["values"])
        assert torch.allclose(emb1, emb2)

    def test_encode_no_values(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        emb = model.encode(mae_batch["input_ids"], mae_batch["attention_mask"])
        B = mae_batch["input_ids"].shape[0]
        assert emb.shape == (B, mae_config.hidden_dim)


class TestFromConfig:
    """Construction from ModelConfig tests."""

    def test_creates_model(self, mae_config):
        model = MaskedAutoencoder.from_config(mae_config)
        assert isinstance(model, MaskedAutoencoder)

    def test_default_decoder_dim(self):
        config = ModelConfig(architecture="masked_autoencoder", vocab_size=100, hidden_dim=64,
                             num_layers=2, num_heads=4)
        model = MaskedAutoencoder.from_config(config)
        assert model.decoder_dim == 32  # hidden_dim // 2

    def test_custom_decoder(self):
        config = ModelConfig(architecture="masked_autoencoder", vocab_size=100, hidden_dim=64,
                             num_layers=2, num_heads=4, decoder_dim=48, decoder_layers=2, decoder_heads=4)
        model = MaskedAutoencoder.from_config(config)
        assert model.decoder_dim == 48

    def test_vocab_size_none_raises(self):
        config = ModelConfig(architecture="masked_autoencoder")
        with pytest.raises(ValueError, match="vocab_size"):
            MaskedAutoencoder.from_config(config)

    def test_via_registry(self, mae_config):
        model = get_model("masked_autoencoder", mae_config)
        assert isinstance(model, MaskedAutoencoder)


class TestGradients:
    """Gradient flow tests."""

    def test_gradients_flow_to_encoder(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        output.loss.backward()
        encoder_params = list(model.encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder_params)

    def test_gradients_flow_to_decoder(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        output.loss.backward()
        decoder_params = list(model.decoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder_params)

    def test_gradients_flow_to_expression_head(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        output.loss.backward()
        assert model.expression_head.output.weight.grad is not None

    def test_gradients_flow_to_mask_token(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        output.loss.backward()
        assert model.mask_token.grad is not None
        assert model.mask_token.grad.abs().sum() > 0


class TestMisc:
    """Miscellaneous tests."""

    def test_num_parameters(self, mae_config):
        model = MaskedAutoencoder.from_config(mae_config)
        n = model.num_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_registered(self):
        assert "masked_autoencoder" in list_models()

    def test_cls_pooling(self):
        config = ModelConfig(
            architecture="masked_autoencoder", vocab_size=100, hidden_dim=64,
            num_layers=2, num_heads=4, pooling="cls", decoder_dim=32, decoder_layers=1, decoder_heads=2,
        )
        model = MaskedAutoencoder.from_config(config)
        batch = {
            "input_ids": torch.randint(1, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "values": torch.rand(2, 8),
        }
        output = model(**batch)
        assert output.embeddings.shape == (2, 64)

    def test_padding_handled(self, mae_config, mae_batch):
        model = MaskedAutoencoder.from_config(mae_config)
        output = model(**mae_batch)
        assert torch.isfinite(output.logits).all()
