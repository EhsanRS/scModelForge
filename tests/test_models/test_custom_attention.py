"""Unit tests for custom attention mechanisms, encoder layers, and encoders."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scmodelforge.config.schema import AttentionConfig, ModelConfig
from scmodelforge.models.components.custom_attention import (
    FlashSelfAttention,
    GeneGeneAttention,
    LinearAttention,
    build_attention,
)
from scmodelforge.models.components.encoder import ScModelForgeEncoder
from scmodelforge.models.components.encoder_layer import ScModelForgeEncoderLayer

# Shared test constants
D_MODEL = 64
NHEAD = 4
DIM_FF = 128
BATCH = 2
SEQ = 8
MAX_GENES = 100


# ---------------------------------------------------------------------------
# FlashSelfAttention
# ---------------------------------------------------------------------------


class TestFlashSelfAttention:
    def test_output_shape(self):
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        out, weights = attn(x, x, x)
        assert out.shape == (BATCH, SEQ, D_MODEL)
        assert weights is None

    def test_with_key_padding_mask(self):
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
        mask[:, -2:] = True  # mask last 2 positions
        out, _ = attn(x, x, x, key_padding_mask=mask)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_with_attn_mask(self):
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        causal = torch.nn.Transformer.generate_square_subsequent_mask(SEQ)
        out, _ = attn(x, x, x, attn_mask=causal)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_submodule_names(self):
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        names = {name for name, _ in attn.named_parameters()}
        assert "in_proj_weight" in names
        assert "in_proj_bias" in names
        assert "out_proj.weight" in names
        assert "out_proj.bias" in names

    def test_dropout(self):
        attn = FlashSelfAttention(D_MODEL, NHEAD, dropout=0.5)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        attn.train()
        out1, _ = attn(x, x, x)
        attn.train(False)
        out2, _ = attn(x, x, x)
        # In inference mode there should be no dropout — outputs deterministic
        out3, _ = attn(x, x, x)
        assert torch.allclose(out2, out3)


# ---------------------------------------------------------------------------
# GeneGeneAttention
# ---------------------------------------------------------------------------


class TestGeneGeneAttention:
    def test_output_shape(self):
        attn = GeneGeneAttention(D_MODEL, NHEAD, max_genes=MAX_GENES)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        gene_idx = torch.randint(0, MAX_GENES, (BATCH, SEQ))
        out, _ = attn(x, x, x, gene_indices=gene_idx)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_without_gene_indices(self):
        """Should work (no bias added) when gene_indices is not provided."""
        attn = GeneGeneAttention(D_MODEL, NHEAD, max_genes=MAX_GENES)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        out, _ = attn(x, x, x)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_gene_bias_parameter(self):
        attn = GeneGeneAttention(D_MODEL, NHEAD, max_genes=MAX_GENES)
        assert attn.gene_bias.shape == (MAX_GENES, MAX_GENES)

    def test_gene_bias_affects_output(self):
        """Gene indices should change the output vs no indices."""
        attn = GeneGeneAttention(D_MODEL, NHEAD, max_genes=MAX_GENES)
        attn.train(False)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        gene_idx = torch.randint(0, MAX_GENES, (BATCH, SEQ))
        out_with, _ = attn(x, x, x, gene_indices=gene_idx)
        out_without, _ = attn(x, x, x)
        # They should differ (gene_bias is non-zero after init)
        assert not torch.allclose(out_with, out_without, atol=1e-6)

    def test_with_key_padding_mask(self):
        attn = GeneGeneAttention(D_MODEL, NHEAD, max_genes=MAX_GENES)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
        mask[:, -2:] = True
        gene_idx = torch.randint(0, MAX_GENES, (BATCH, SEQ))
        out, _ = attn(x, x, x, key_padding_mask=mask, gene_indices=gene_idx)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_submodule_names(self):
        attn = GeneGeneAttention(D_MODEL, NHEAD, max_genes=MAX_GENES)
        names = {name for name, _ in attn.named_parameters()}
        assert "out_proj.weight" in names
        assert "gene_bias" in names

    def test_gene_bias_init_std(self):
        attn = GeneGeneAttention(D_MODEL, NHEAD, max_genes=MAX_GENES, gene_bias_init_std=0.01)
        # Check that the std is roughly correct (with tolerance for finite samples)
        assert attn.gene_bias.std().item() < 0.05


# ---------------------------------------------------------------------------
# LinearAttention
# ---------------------------------------------------------------------------


class TestLinearAttention:
    def test_output_shape(self):
        attn = LinearAttention(D_MODEL, NHEAD)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        out, weights = attn(x, x, x)
        assert out.shape == (BATCH, SEQ, D_MODEL)
        assert weights is None

    def test_with_key_padding_mask(self):
        attn = LinearAttention(D_MODEL, NHEAD)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
        mask[:, -2:] = True
        out, _ = attn(x, x, x, key_padding_mask=mask)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_submodule_names(self):
        attn = LinearAttention(D_MODEL, NHEAD)
        names = {name for name, _ in attn.named_parameters()}
        assert "in_proj_weight" in names
        assert "out_proj.weight" in names

    def test_elu_feature_map(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = LinearAttention._elu_feature_map(x)
        # ELU(-1)+1 ~ 0.632, ELU(0)+1 = 1.0, ELU(1)+1 = 2.0
        assert result[1].item() == pytest.approx(1.0)
        assert result[2].item() == pytest.approx(2.0)
        assert result[0].item() > 0  # Must be positive

    def test_output_is_finite(self):
        """Linear attention should produce finite outputs."""
        attn = LinearAttention(D_MODEL, NHEAD)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        out, _ = attn(x, x, x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# build_attention factory
# ---------------------------------------------------------------------------


class TestBuildAttention:
    def test_standard(self):
        attn = build_attention("standard", D_MODEL, NHEAD)
        assert isinstance(attn, nn.MultiheadAttention)

    def test_flash(self):
        attn = build_attention("flash", D_MODEL, NHEAD)
        assert isinstance(attn, FlashSelfAttention)

    def test_gene_bias(self):
        attn = build_attention("gene_bias", D_MODEL, NHEAD, max_genes=MAX_GENES)
        assert isinstance(attn, GeneGeneAttention)

    def test_linear(self):
        attn = build_attention("linear", D_MODEL, NHEAD)
        assert isinstance(attn, LinearAttention)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown attention type"):
            build_attention("nonexistent", D_MODEL, NHEAD)


# ---------------------------------------------------------------------------
# ScModelForgeEncoderLayer
# ---------------------------------------------------------------------------


class TestScModelForgeEncoderLayer:
    @pytest.fixture()
    def flash_layer(self):
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        return ScModelForgeEncoderLayer(attn, D_MODEL, DIM_FF)

    def test_output_shape(self, flash_layer):
        x = torch.randn(BATCH, SEQ, D_MODEL)
        out = flash_layer(x)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_with_padding_mask(self, flash_layer):
        x = torch.randn(BATCH, SEQ, D_MODEL)
        mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
        mask[:, -2:] = True
        out = flash_layer(x, src_key_padding_mask=mask)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_submodule_names_for_lora(self, flash_layer):
        """LoRA needs: self_attn, linear1, linear2."""
        child_names = {name for name, _ in flash_layer.named_children()}
        assert "self_attn" in child_names
        assert "linear1" in child_names
        assert "linear2" in child_names
        assert "norm1" in child_names
        assert "norm2" in child_names

    def test_kwargs_propagation(self):
        """Gene bias attention requires gene_indices kwarg."""
        attn = GeneGeneAttention(D_MODEL, NHEAD, max_genes=MAX_GENES)
        layer = ScModelForgeEncoderLayer(attn, D_MODEL, DIM_FF)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        gene_idx = torch.randint(0, MAX_GENES, (BATCH, SEQ))
        out = layer(x, gene_indices=gene_idx)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_activation_gelu(self):
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        layer = ScModelForgeEncoderLayer(attn, D_MODEL, DIM_FF, activation="gelu")
        x = torch.randn(BATCH, SEQ, D_MODEL)
        out = layer(x)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_activation_relu(self):
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        layer = ScModelForgeEncoderLayer(attn, D_MODEL, DIM_FF, activation="relu")
        x = torch.randn(BATCH, SEQ, D_MODEL)
        out = layer(x)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_invalid_activation(self):
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        with pytest.raises(ValueError, match="Unknown activation"):
            ScModelForgeEncoderLayer(attn, D_MODEL, DIM_FF, activation="bad")


# ---------------------------------------------------------------------------
# ScModelForgeEncoder
# ---------------------------------------------------------------------------


class TestScModelForgeEncoder:
    def test_output_shape(self):
        import copy
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        layer = ScModelForgeEncoderLayer(attn, D_MODEL, DIM_FF)
        layers = nn.ModuleList([layer, copy.deepcopy(layer)])
        encoder = ScModelForgeEncoder(layers)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        out = encoder(x)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_with_norm(self):
        import copy
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        layer = ScModelForgeEncoderLayer(attn, D_MODEL, DIM_FF)
        layers = nn.ModuleList([layer, copy.deepcopy(layer)])
        norm = nn.LayerNorm(D_MODEL)
        encoder = ScModelForgeEncoder(layers, norm=norm)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        out = encoder(x)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_kwargs_propagation(self):
        import copy
        attn = GeneGeneAttention(D_MODEL, NHEAD, max_genes=MAX_GENES)
        layer = ScModelForgeEncoderLayer(attn, D_MODEL, DIM_FF)
        layers = nn.ModuleList([layer, copy.deepcopy(layer)])
        encoder = ScModelForgeEncoder(layers)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        gene_idx = torch.randint(0, MAX_GENES, (BATCH, SEQ))
        out = encoder(x, gene_indices=gene_idx)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_mask_and_padding(self):
        import copy
        attn = FlashSelfAttention(D_MODEL, NHEAD)
        layer = ScModelForgeEncoderLayer(attn, D_MODEL, DIM_FF)
        layers = nn.ModuleList([layer, copy.deepcopy(layer)])
        encoder = ScModelForgeEncoder(layers)
        x = torch.randn(BATCH, SEQ, D_MODEL)
        mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
        mask[:, -2:] = True
        out = encoder(x, src_key_padding_mask=mask)
        assert out.shape == (BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# build_encoder_layer / build_encoder builders
# ---------------------------------------------------------------------------


class TestBuilders:
    def test_build_encoder_layer_standard(self):
        from scmodelforge.models.components.attention import build_encoder_layer
        layer = build_encoder_layer("standard", D_MODEL, NHEAD, DIM_FF)
        assert isinstance(layer, nn.TransformerEncoderLayer)

    def test_build_encoder_layer_flash(self):
        from scmodelforge.models.components.attention import build_encoder_layer
        layer = build_encoder_layer("flash", D_MODEL, NHEAD, DIM_FF)
        assert isinstance(layer, ScModelForgeEncoderLayer)
        assert isinstance(layer.self_attn, FlashSelfAttention)

    def test_build_encoder_layer_gene_bias(self):
        from scmodelforge.models.components.attention import build_encoder_layer
        layer = build_encoder_layer("gene_bias", D_MODEL, NHEAD, DIM_FF, max_genes=MAX_GENES)
        assert isinstance(layer, ScModelForgeEncoderLayer)
        assert isinstance(layer.self_attn, GeneGeneAttention)

    def test_build_encoder_layer_linear(self):
        from scmodelforge.models.components.attention import build_encoder_layer
        layer = build_encoder_layer("linear", D_MODEL, NHEAD, DIM_FF)
        assert isinstance(layer, ScModelForgeEncoderLayer)
        assert isinstance(layer.self_attn, LinearAttention)

    def test_build_encoder_standard(self):
        from scmodelforge.models.components.attention import build_encoder, build_encoder_layer
        layer = build_encoder_layer("standard", D_MODEL, NHEAD, DIM_FF)
        encoder = build_encoder("standard", layer, num_layers=2)
        assert isinstance(encoder, nn.TransformerEncoder)

    def test_build_encoder_flash(self):
        from scmodelforge.models.components.attention import build_encoder, build_encoder_layer
        layer = build_encoder_layer("flash", D_MODEL, NHEAD, DIM_FF)
        encoder = build_encoder("flash", layer, num_layers=2)
        assert isinstance(encoder, nn.TransformerEncoder)

    def test_build_encoder_gene_bias(self):
        from scmodelforge.models.components.attention import build_encoder, build_encoder_layer
        layer = build_encoder_layer("gene_bias", D_MODEL, NHEAD, DIM_FF, max_genes=MAX_GENES)
        encoder = build_encoder("gene_bias", layer, num_layers=2)
        assert isinstance(encoder, ScModelForgeEncoder)
        assert len(encoder.layers) == 2

    def test_build_encoder_linear(self):
        from scmodelforge.models.components.attention import build_encoder, build_encoder_layer
        layer = build_encoder_layer("linear", D_MODEL, NHEAD, DIM_FF)
        encoder = build_encoder("linear", layer, num_layers=3)
        assert isinstance(encoder, ScModelForgeEncoder)
        assert len(encoder.layers) == 3


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestAttentionConfig:
    def test_defaults(self):
        cfg = AttentionConfig()
        assert cfg.type == "standard"
        assert cfg.max_genes == 30000
        assert cfg.gene_bias_init_std == 0.02
        assert cfg.linear_feature_map == "elu"

    def test_model_config_has_attention(self):
        cfg = ModelConfig(vocab_size=100)
        assert isinstance(cfg.attention, AttentionConfig)
        assert cfg.attention.type == "standard"

    def test_model_config_custom_attention(self):
        cfg = ModelConfig(vocab_size=100, attention=AttentionConfig(type="flash"))
        assert cfg.attention.type == "flash"
