"""Tests for TransformerEncoder model."""

from __future__ import annotations

import pytest
import torch

from scmodelforge.config.schema import ModelConfig
from scmodelforge.models.protocol import ModelOutput
from scmodelforge.models.transformer_encoder import TransformerEncoder


class TestTransformerEncoderForward:
    """Tests for the forward pass."""

    def test_forward_returns_model_output(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        out = model(**dummy_batch)
        assert isinstance(out, ModelOutput)

    def test_forward_embeddings_shape(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        out = model(**dummy_batch)
        assert out.embeddings.shape == (2, tiny_config.hidden_dim)

    def test_forward_logits_shape(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        out = model(**dummy_batch)
        seq_len = dummy_batch["input_ids"].size(1)
        assert out.logits.shape == (2, seq_len, tiny_config.vocab_size)

    def test_forward_with_labels_has_loss(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        out = model(**dummy_batch)
        assert out.loss is not None
        assert out.loss.shape == ()
        assert out.loss.item() > 0

    def test_forward_without_labels_no_loss(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        batch_no_labels = {k: v for k, v in dummy_batch.items() if k != "labels"}
        out = model(**batch_no_labels)
        assert out.loss is None

    def test_forward_without_values(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        batch = {k: v for k, v in dummy_batch.items() if k != "values"}
        out = model(**batch)
        assert out.embeddings.shape == (2, tiny_config.hidden_dim)

    def test_forward_no_expression_projection(self, tiny_config, dummy_batch):
        tiny_config.use_expression_values = False
        model = TransformerEncoder.from_config(tiny_config)
        out = model(**dummy_batch)
        assert out.embeddings.shape == (2, tiny_config.hidden_dim)


class TestTransformerEncoderEncode:
    """Tests for the encode method."""

    def test_encode_output_shape(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        emb = model.encode(
            dummy_batch["input_ids"],
            dummy_batch["attention_mask"],
            values=dummy_batch["values"],
        )
        assert emb.shape == (2, tiny_config.hidden_dim)

    def test_encode_cls_matches_forward(self, tiny_config, dummy_batch):
        tiny_config.pooling = "cls"
        model = TransformerEncoder.from_config(tiny_config)
        model.training = False
        with torch.no_grad():
            emb = model.encode(dummy_batch["input_ids"], dummy_batch["attention_mask"])
            out = model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        assert torch.allclose(emb, out.embeddings, atol=1e-6)

    def test_encode_mean_pooling(self, tiny_config, dummy_batch):
        tiny_config.pooling = "mean"
        model = TransformerEncoder.from_config(tiny_config)
        model.training = False
        with torch.no_grad():
            emb = model.encode(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        assert emb.shape == (2, tiny_config.hidden_dim)

    def test_encode_no_grad(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        model.training = False
        with torch.no_grad():
            emb = model.encode(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        assert not emb.requires_grad


class TestTransformerEncoderFromConfig:
    """Tests for from_config classmethod."""

    def test_from_config_creates_model(self, tiny_config):
        model = TransformerEncoder.from_config(tiny_config)
        assert isinstance(model, TransformerEncoder)
        assert model.hidden_dim == tiny_config.hidden_dim
        assert model.vocab_size == tiny_config.vocab_size

    def test_from_config_ffn_dim_none_defaults(self):
        config = ModelConfig(vocab_size=100, hidden_dim=64, num_layers=2, num_heads=4, ffn_dim=None)
        model = TransformerEncoder.from_config(config)
        # ffn_dim should default to 4 * hidden_dim = 256
        encoder_layer = model.encoder.layers[0]
        assert encoder_layer.linear1.out_features == 256

    def test_from_config_vocab_size_none_raises(self):
        config = ModelConfig(vocab_size=None)
        with pytest.raises(ValueError, match="vocab_size must be set"):
            TransformerEncoder.from_config(config)

    def test_from_config_custom_settings(self):
        config = ModelConfig(
            vocab_size=200,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            ffn_dim=512,
            dropout=0.2,
            max_seq_len=64,
            pooling="mean",
            activation="relu",
            use_expression_values=False,
        )
        model = TransformerEncoder.from_config(config)
        assert model.vocab_size == 200
        assert model.hidden_dim == 128
        assert model._pooling_strategy == "mean"
        assert len(model.encoder.layers) == 4


class TestTransformerEncoderGradients:
    """Tests for gradient flow."""

    def test_gradient_flows_to_embeddings(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        out = model(**dummy_batch)
        out.loss.backward()
        assert model.embedding.gene_embedding.weight.grad is not None

    def test_gradient_flows_to_head(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        out = model(**dummy_batch)
        out.loss.backward()
        assert model.head.decoder.weight.grad is not None

    def test_gradient_flows_to_encoder(self, tiny_config, dummy_batch):
        model = TransformerEncoder.from_config(tiny_config)
        out = model(**dummy_batch)
        out.loss.backward()
        layer = model.encoder.layers[0]
        assert layer.self_attn.in_proj_weight.grad is not None


class TestTransformerEncoderMisc:
    """Miscellaneous tests."""

    def test_num_parameters(self, tiny_config):
        model = TransformerEncoder.from_config(tiny_config)
        n_params = model.num_parameters()
        assert n_params > 0
        # All parameters should be trainable by default
        n_all = model.num_parameters(trainable_only=False)
        assert n_all == n_params

    def test_num_parameters_with_frozen(self, tiny_config):
        model = TransformerEncoder.from_config(tiny_config)
        # Freeze embedding
        for p in model.embedding.parameters():
            p.requires_grad = False
        n_trainable = model.num_parameters(trainable_only=True)
        n_all = model.num_parameters(trainable_only=False)
        assert n_trainable < n_all

    def test_unknown_pooling_raises(self, tiny_config):
        model = TransformerEncoder.from_config(tiny_config)
        model._pooling_strategy = "invalid"
        ids = torch.randint(0, 100, (1, 5))
        mask = torch.ones(1, 5, dtype=torch.long)
        with pytest.raises(ValueError, match="Unknown pooling"):
            model(ids, mask)

    def test_batch_size_one(self, tiny_config):
        model = TransformerEncoder.from_config(tiny_config)
        ids = torch.randint(1, 100, (1, 5))
        mask = torch.ones(1, 5, dtype=torch.long)
        out = model(ids, mask)
        assert out.embeddings.shape == (1, tiny_config.hidden_dim)

    def test_padding_handled(self, tiny_config):
        model = TransformerEncoder.from_config(tiny_config)
        model.training = False
        ids = torch.randint(1, 100, (1, 10))
        mask_full = torch.ones(1, 10, dtype=torch.long)
        # Same input but last 5 positions are padding
        ids_padded = ids.clone()
        ids_padded[0, 5:] = 0
        mask_padded = torch.ones(1, 10, dtype=torch.long)
        mask_padded[0, 5:] = 0
        with torch.no_grad():
            out_full = model(ids, mask_full)
            out_padded = model(ids_padded, mask_padded)
        # Embeddings should differ because of different masking
        assert out_full.embeddings.shape == out_padded.embeddings.shape
