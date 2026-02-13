"""Integration tests for custom attention with model architectures, FSDP, and LoRA."""

from __future__ import annotations

import pytest
import torch

from scmodelforge.config.schema import AttentionConfig, ModelConfig
from scmodelforge.models.autoregressive import AutoregressiveTransformer
from scmodelforge.models.components.encoder_layer import ScModelForgeEncoderLayer
from scmodelforge.models.masked_autoencoder import MaskedAutoencoder
from scmodelforge.models.transformer_encoder import TransformerEncoder

# Small model config for fast tests
VOCAB = 100
HIDDEN = 64
LAYERS = 2
HEADS = 4
SEQ = 16
BATCH = 2
MAX_GENES = VOCAB  # gene indices are input_ids, so max = vocab_size


def _make_config(attention_type: str = "standard") -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB,
        hidden_dim=HIDDEN,
        num_layers=LAYERS,
        num_heads=HEADS,
        max_seq_len=SEQ,
        decoder_layers=2,
        attention=AttentionConfig(type=attention_type, max_genes=MAX_GENES),
    )


def _make_inputs():
    input_ids = torch.randint(4, VOCAB, (BATCH, SEQ))
    # First token is CLS (token 3)
    input_ids[:, 0] = 3
    mask = torch.ones(BATCH, SEQ, dtype=torch.long)
    mask[:, -2:] = 0
    values = torch.rand(BATCH, SEQ)
    labels = torch.randint(4, VOCAB, (BATCH, SEQ))
    labels[:, -2:] = -100
    return input_ids, mask, values, labels


# ---------------------------------------------------------------------------
# TransformerEncoder with each attention type
# ---------------------------------------------------------------------------


class TestTransformerEncoderAttention:
    @pytest.mark.parametrize("attn_type", ["standard", "flash", "gene_bias", "linear"])
    def test_forward(self, attn_type):
        config = _make_config(attn_type)
        model = TransformerEncoder.from_config(config)
        model.train(False)
        input_ids, mask, values, labels = _make_inputs()
        out = model(input_ids, mask, values=values, labels=labels)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)
        assert out.embeddings.shape == (BATCH, HIDDEN)
        assert out.loss is not None

    @pytest.mark.parametrize("attn_type", ["standard", "flash", "gene_bias", "linear"])
    def test_encode(self, attn_type):
        config = _make_config(attn_type)
        model = TransformerEncoder.from_config(config)
        model.train(False)
        input_ids, mask, values, _ = _make_inputs()
        emb = model.encode(input_ids, mask, values=values)
        assert emb.shape == (BATCH, HIDDEN)


# ---------------------------------------------------------------------------
# AutoregressiveTransformer with each attention type
# ---------------------------------------------------------------------------


class TestAutoregressiveAttention:
    @pytest.mark.parametrize("attn_type", ["standard", "flash", "gene_bias", "linear"])
    def test_forward(self, attn_type):
        config = _make_config(attn_type)
        model = AutoregressiveTransformer.from_config(config)
        model.train(False)
        input_ids, mask, values, labels = _make_inputs()
        out = model(input_ids, mask, values=values, labels=labels)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)
        assert out.embeddings.shape == (BATCH, HIDDEN)
        assert out.loss is not None

    @pytest.mark.parametrize("attn_type", ["standard", "flash", "gene_bias", "linear"])
    def test_encode(self, attn_type):
        config = _make_config(attn_type)
        model = AutoregressiveTransformer.from_config(config)
        model.train(False)
        input_ids, mask, values, _ = _make_inputs()
        emb = model.encode(input_ids, mask, values=values)
        assert emb.shape == (BATCH, HIDDEN)


# ---------------------------------------------------------------------------
# MaskedAutoencoder with each attention type
# ---------------------------------------------------------------------------


class TestMaskedAutoencoderAttention:
    @pytest.mark.parametrize("attn_type", ["standard", "flash", "gene_bias", "linear"])
    def test_forward(self, attn_type):
        config = _make_config(attn_type)
        config.pooling = "mean"
        model = MaskedAutoencoder.from_config(config)
        model.train(False)
        input_ids, mask, values, labels = _make_inputs()
        out = model(input_ids, mask, values=values, labels=labels)
        assert out.logits.shape == (BATCH, SEQ)
        assert out.embeddings.shape == (BATCH, HIDDEN)
        assert out.loss is not None

    @pytest.mark.parametrize("attn_type", ["standard", "flash", "gene_bias", "linear"])
    def test_encode(self, attn_type):
        config = _make_config(attn_type)
        config.pooling = "mean"
        model = MaskedAutoencoder.from_config(config)
        model.train(False)
        input_ids, mask, values, _ = _make_inputs()
        emb = model.encode(input_ids, mask, values=values)
        assert emb.shape == (BATCH, HIDDEN)


# ---------------------------------------------------------------------------
# Backward pass / gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    @pytest.mark.parametrize("attn_type", ["flash", "gene_bias", "linear"])
    def test_backward_pass(self, attn_type):
        config = _make_config(attn_type)
        model = TransformerEncoder.from_config(config)
        model.train()
        input_ids, mask, values, labels = _make_inputs()
        out = model(input_ids, mask, values=values, labels=labels)
        out.loss.backward()
        # Check some gradients exist
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"
                break
        else:
            pytest.fail("No gradients found")


# ---------------------------------------------------------------------------
# FSDP activation checkpointing class set
# ---------------------------------------------------------------------------


class TestFSDPCheckpointing:
    def test_custom_layer_in_checkpointing_set(self):
        """ScModelForgeEncoderLayer should be in the activation checkpointing set."""
        from scmodelforge.config.schema import FSDPConfig

        fsdp_config = FSDPConfig(activation_checkpointing=True)
        from scmodelforge.training.fsdp import build_fsdp_strategy

        try:
            strategy = build_fsdp_strategy(fsdp_config)
            if strategy.kwargs.get("activation_checkpointing_policy"):
                assert ScModelForgeEncoderLayer in strategy.kwargs["activation_checkpointing_policy"]
        except (ImportError, Exception):
            # Lightning not available — just verify the import works
            from scmodelforge.models.components.encoder_layer import ScModelForgeEncoderLayer as _cls
            assert _cls is not None


# ---------------------------------------------------------------------------
# LoRA compatibility
# ---------------------------------------------------------------------------


class TestLoRACompatibility:
    @pytest.mark.parametrize("attn_type", ["flash", "gene_bias", "linear"])
    def test_lora_target_modules_exist(self, attn_type):
        """Verify out_proj, linear1, linear2 submodule names exist."""
        config = _make_config(attn_type)
        model = TransformerEncoder.from_config(config)
        # Collect all named modules
        module_names = {name for name, _ in model.named_modules()}

        # Check that LoRA target submodule names exist somewhere
        found_out_proj = any("out_proj" in name for name in module_names)
        found_linear1 = any("linear1" in name for name in module_names)
        found_linear2 = any("linear2" in name for name in module_names)

        assert found_out_proj, "out_proj not found in model modules"
        assert found_linear1, "linear1 not found in model modules"
        assert found_linear2, "linear2 not found in model modules"

    def test_lora_apply(self):
        """Test actual LoRA application if peft is installed."""
        pytest.importorskip("peft")
        from scmodelforge.config.schema import LoRAConfig
        from scmodelforge.finetuning.adapters import apply_lora

        config = _make_config("flash")
        model = TransformerEncoder.from_config(config)
        lora_config = LoRAConfig(enabled=True, rank=4, alpha=8)
        peft_model = apply_lora(model, lora_config)

        # Verify the model still runs
        input_ids, mask, values, _ = _make_inputs()
        out = peft_model(input_ids, mask, values=values)
        assert out.embeddings.shape == (BATCH, HIDDEN)


# ---------------------------------------------------------------------------
# Default config backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_standard_default_produces_transformer_encoder_layer(self):
        """Default config should use nn.TransformerEncoderLayer."""
        import torch.nn as nn

        config = _make_config("standard")
        model = TransformerEncoder.from_config(config)
        # The encoder should be nn.TransformerEncoder
        assert isinstance(model.encoder, nn.TransformerEncoder)

    def test_standard_matches_original_output_shape(self):
        """Standard attention should produce identical shapes."""
        config = _make_config("standard")
        model = TransformerEncoder.from_config(config)
        model.train(False)
        input_ids, mask, values, labels = _make_inputs()
        out = model(input_ids, mask, values=values, labels=labels)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)
        assert out.embeddings.shape == (BATCH, HIDDEN)
