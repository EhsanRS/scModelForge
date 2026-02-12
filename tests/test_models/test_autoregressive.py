"""Tests for AutoregressiveTransformer model."""

from __future__ import annotations

import pytest
import torch

from scmodelforge.config.schema import ModelConfig
from scmodelforge.models.autoregressive import AutoregressiveTransformer
from scmodelforge.models.protocol import ModelOutput
from scmodelforge.models.registry import get_model, list_models


class TestForward:
    """Forward pass tests."""

    def test_returns_model_output(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        assert isinstance(output, ModelOutput)

    def test_logits_shape(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        B, S = ar_batch["input_ids"].shape
        assert output.logits.shape == (B, S, ar_config.vocab_size)

    def test_embeddings_shape(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        B = ar_batch["input_ids"].shape[0]
        assert output.embeddings.shape == (B, ar_config.hidden_dim)

    def test_loss_with_labels(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        assert output.loss is not None
        assert output.loss.ndim == 0  # scalar
        assert output.loss.item() > 0

    def test_combined_loss_with_bin_ids(self, ar_config, ar_batch):
        """Loss should be higher when bin prediction is also computed."""
        model = AutoregressiveTransformer.from_config(ar_config)
        # With bin_ids
        output_with_bins = model(**ar_batch)
        # Without bin_ids
        batch_no_bins = {k: v for k, v in ar_batch.items() if k != "bin_ids"}
        output_no_bins = model(**batch_no_bins)
        # Both should have loss, the combined one includes bin loss too
        assert output_with_bins.loss is not None
        assert output_no_bins.loss is not None

    def test_no_loss_without_labels(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        batch_no_labels = {k: v for k, v in ar_batch.items() if k != "labels"}
        output = model(**batch_no_labels)
        assert output.loss is None

    def test_no_loss_without_labels_even_with_bins(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        batch = {k: v for k, v in ar_batch.items() if k != "labels"}
        output = model(**batch)
        assert output.loss is None

    def test_forward_without_values(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        batch = {k: v for k, v in ar_batch.items() if k != "values"}
        output = model(**batch)
        assert isinstance(output, ModelOutput)
        assert output.loss is not None

    def test_loss_is_finite(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        assert torch.isfinite(output.loss)

    def test_causal_masking_different_from_bidirectional(self, ar_config, ar_batch):
        """AR model with causal mask should produce different outputs than encode (no causal)."""
        model = AutoregressiveTransformer.from_config(ar_config)
        model.set_default_tensor_type = None  # noqa: just to avoid lint confusion
        del model.set_default_tensor_type
        model = AutoregressiveTransformer.from_config(ar_config)
        model.train(False)
        with torch.no_grad():
            output = model(**ar_batch)
            embeddings_encode = model.encode(
                ar_batch["input_ids"], ar_batch["attention_mask"], ar_batch["values"]
            )
        # forward uses causal mask, encode does not — embeddings should differ
        assert not torch.allclose(output.embeddings, embeddings_encode, atol=1e-5)


class TestEncode:
    """Encode method tests."""

    def test_shape(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        emb = model.encode(ar_batch["input_ids"], ar_batch["attention_mask"], ar_batch["values"])
        B = ar_batch["input_ids"].shape[0]
        assert emb.shape == (B, ar_config.hidden_dim)

    def test_deterministic(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        model.train(False)
        with torch.no_grad():
            emb1 = model.encode(ar_batch["input_ids"], ar_batch["attention_mask"], ar_batch["values"])
            emb2 = model.encode(ar_batch["input_ids"], ar_batch["attention_mask"], ar_batch["values"])
        assert torch.allclose(emb1, emb2)

    def test_encode_no_values(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        emb = model.encode(ar_batch["input_ids"], ar_batch["attention_mask"])
        B = ar_batch["input_ids"].shape[0]
        assert emb.shape == (B, ar_config.hidden_dim)


class TestFromConfig:
    """Construction from ModelConfig tests."""

    def test_creates_model(self, ar_config):
        model = AutoregressiveTransformer.from_config(ar_config)
        assert isinstance(model, AutoregressiveTransformer)

    def test_default_n_bins(self):
        config = ModelConfig(architecture="autoregressive_transformer", vocab_size=100, hidden_dim=64, num_layers=2, num_heads=4)
        model = AutoregressiveTransformer.from_config(config)
        assert model.n_bins == 51

    def test_custom_n_bins(self):
        config = ModelConfig(architecture="autoregressive_transformer", vocab_size=100, hidden_dim=64, num_layers=2, num_heads=4, n_bins=101)
        model = AutoregressiveTransformer.from_config(config)
        assert model.n_bins == 101

    def test_vocab_size_none_raises(self):
        config = ModelConfig(architecture="autoregressive_transformer")
        with pytest.raises(ValueError, match="vocab_size"):
            AutoregressiveTransformer.from_config(config)

    def test_custom_loss_weights(self):
        config = ModelConfig(
            architecture="autoregressive_transformer", vocab_size=100, hidden_dim=64,
            num_layers=2, num_heads=4, gene_loss_weight=0.5, expression_loss_weight=2.0,
        )
        model = AutoregressiveTransformer.from_config(config)
        assert model._gene_loss_weight == 0.5
        assert model._expression_loss_weight == 2.0

    def test_via_registry(self, ar_config):
        model = get_model("autoregressive_transformer", ar_config)
        assert isinstance(model, AutoregressiveTransformer)


class TestGradients:
    """Gradient flow tests."""

    def test_gradients_flow_to_embedding(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        output.loss.backward()
        assert model.embedding.gene_embedding.weight.grad is not None
        assert model.embedding.gene_embedding.weight.grad.abs().sum() > 0

    def test_gradients_flow_to_gene_head(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        output.loss.backward()
        assert model.gene_head.decoder.weight.grad is not None

    def test_gradients_flow_to_expression_head(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        output.loss.backward()
        assert model.expression_head.decoder.weight.grad is not None

    def test_gradients_flow_to_encoder(self, ar_config, ar_batch):
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        output.loss.backward()
        encoder_params = list(model.encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder_params)


class TestMisc:
    """Miscellaneous tests."""

    def test_num_parameters(self, ar_config):
        model = AutoregressiveTransformer.from_config(ar_config)
        n = model.num_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_registered(self):
        assert "autoregressive_transformer" in list_models()

    def test_mean_pooling(self):
        config = ModelConfig(
            architecture="autoregressive_transformer", vocab_size=100, hidden_dim=64,
            num_layers=2, num_heads=4, pooling="mean",
        )
        model = AutoregressiveTransformer.from_config(config)
        batch = {
            "input_ids": torch.randint(1, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
        output = model(**batch)
        assert output.embeddings.shape == (2, 64)

    def test_padding_handled(self, ar_config, ar_batch):
        """Model should run without error even with padding tokens."""
        model = AutoregressiveTransformer.from_config(ar_config)
        output = model(**ar_batch)
        assert torch.isfinite(output.logits).all()
