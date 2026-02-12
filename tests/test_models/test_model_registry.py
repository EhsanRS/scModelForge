"""Tests for the model registry."""

from __future__ import annotations

import pytest
import torch.nn as nn

from scmodelforge.config.schema import ModelConfig
from scmodelforge.models.registry import _MODEL_REGISTRY, get_model, list_models, register_model


class TestModelRegistry:
    """Tests for model registration, lookup, and listing."""

    def test_transformer_encoder_registered(self):
        assert "transformer_encoder" in _MODEL_REGISTRY

    def test_list_models_includes_transformer_encoder(self):
        names = list_models()
        assert "transformer_encoder" in names
        assert names == sorted(names)

    def test_get_model_returns_instance(self):
        config = ModelConfig(vocab_size=100, hidden_dim=64, num_layers=2, num_heads=4)
        model = get_model("transformer_encoder", config)
        assert isinstance(model, nn.Module)

    def test_get_model_unknown_raises(self):
        config = ModelConfig(vocab_size=100)
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent_model", config)

    def test_get_model_error_lists_available(self):
        config = ModelConfig(vocab_size=100)
        with pytest.raises(ValueError, match="transformer_encoder"):
            get_model("bad_name", config)

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):

            @register_model("transformer_encoder")
            class DuplicateModel(nn.Module):
                pass

    def test_register_new_model(self):
        name = "_test_temp_model"
        try:

            @register_model(name)
            class TempModel(nn.Module):
                @classmethod
                def from_config(cls, config):
                    return cls()

            assert name in list_models()
            config = ModelConfig(vocab_size=100)
            model = get_model(name, config)
            assert isinstance(model, TempModel)
        finally:
            _MODEL_REGISTRY.pop(name, None)

    def test_list_models_sorted(self):
        names = list_models()
        assert names == sorted(names)
