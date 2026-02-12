"""Tests for HuggingFace Hub integration (save/load/push)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from scmodelforge.config.schema import MaskingConfig, ModelConfig, TokenizerConfig
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.models.hub import (
    CONFIG_NAME,
    README_NAME,
    VOCAB_NAME,
    WEIGHTS_SAFETENSORS_NAME,
    WEIGHTS_TORCH_NAME,
    _build_config_dict,
    _config_dict_to_model_config,
    _config_dict_to_tokenizer_config,
    _is_hub_repo_id,
    load_pretrained,
    load_pretrained_with_vocab,
    push_to_hub,
    save_pretrained,
)
from scmodelforge.models.registry import get_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_model_config() -> ModelConfig:
    """Tiny TransformerEncoder config for fast tests."""
    return ModelConfig(
        architecture="transformer_encoder",
        vocab_size=100,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        dropout=0.0,
        max_seq_len=16,
        pooling="cls",
    )


@pytest.fixture()
def tiny_model(tiny_model_config):
    """Tiny TransformerEncoder model."""
    return get_model("transformer_encoder", tiny_model_config)


@pytest.fixture()
def tiny_vocab() -> GeneVocab:
    """Vocabulary with 10 genes."""
    return GeneVocab.from_genes([f"GENE{i}" for i in range(10)])


@pytest.fixture()
def tiny_tokenizer_config() -> TokenizerConfig:
    """A simple tokenizer config."""
    return TokenizerConfig(
        strategy="rank_value",
        max_genes=16,
        prepend_cls=True,
        masking=MaskingConfig(mask_ratio=0.15),
    )


@pytest.fixture()
def ar_model_config() -> ModelConfig:
    """Tiny AutoregressiveTransformer config."""
    return ModelConfig(
        architecture="autoregressive_transformer",
        vocab_size=100,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        dropout=0.0,
        max_seq_len=16,
        n_bins=10,
    )


@pytest.fixture()
def mae_model_config() -> ModelConfig:
    """Tiny MaskedAutoencoder config."""
    return ModelConfig(
        architecture="masked_autoencoder",
        vocab_size=100,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        dropout=0.0,
        max_seq_len=16,
        pooling="mean",
        decoder_dim=16,
        decoder_layers=1,
        decoder_heads=2,
    )


# ---------------------------------------------------------------------------
# TestBuildConfigDict
# ---------------------------------------------------------------------------


class TestBuildConfigDict:
    """Tests for _build_config_dict()."""

    def test_has_version_key(self, tiny_model_config):
        d = _build_config_dict(tiny_model_config)
        assert "scmodelforge_version" in d
        assert isinstance(d["scmodelforge_version"], str)

    def test_has_model_type(self, tiny_model_config):
        d = _build_config_dict(tiny_model_config)
        assert d["model_type"] == "transformer_encoder"

    def test_model_config_fields(self, tiny_model_config):
        d = _build_config_dict(tiny_model_config)
        assert d["model_config"]["hidden_dim"] == 32
        assert d["model_config"]["num_layers"] == 1
        assert d["model_config"]["vocab_size"] == 100

    def test_tokenizer_included(self, tiny_model_config, tiny_tokenizer_config):
        d = _build_config_dict(tiny_model_config, tiny_tokenizer_config)
        assert "tokenizer_config" in d
        assert d["tokenizer_config"]["strategy"] == "rank_value"

    def test_tokenizer_omitted_when_none(self, tiny_model_config):
        d = _build_config_dict(tiny_model_config)
        assert "tokenizer_config" not in d


# ---------------------------------------------------------------------------
# TestConfigRoundTrip
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    """Tests for config serialize then deserialize roundtrip."""

    def test_model_config_roundtrip(self, tiny_model_config):
        d = _build_config_dict(tiny_model_config)
        restored = _config_dict_to_model_config(d)
        assert restored.hidden_dim == tiny_model_config.hidden_dim
        assert restored.num_layers == tiny_model_config.num_layers
        assert restored.vocab_size == tiny_model_config.vocab_size
        assert restored.architecture == tiny_model_config.architecture

    def test_tokenizer_config_roundtrip(self, tiny_model_config, tiny_tokenizer_config):
        d = _build_config_dict(tiny_model_config, tiny_tokenizer_config)
        restored = _config_dict_to_tokenizer_config(d)
        assert restored is not None
        assert restored.strategy == "rank_value"
        assert restored.masking.mask_ratio == pytest.approx(0.15)

    def test_unknown_fields_ignored(self, tiny_model_config):
        d = _build_config_dict(tiny_model_config)
        d["model_config"]["totally_new_field"] = "should_be_ignored"
        restored = _config_dict_to_model_config(d)
        assert restored.hidden_dim == tiny_model_config.hidden_dim
        assert not hasattr(restored, "totally_new_field")

    def test_tokenizer_config_none_when_absent(self, tiny_model_config):
        d = _build_config_dict(tiny_model_config)
        assert _config_dict_to_tokenizer_config(d) is None


# ---------------------------------------------------------------------------
# TestSavePretrained
# ---------------------------------------------------------------------------


class TestSavePretrained:
    """Tests for save_pretrained()."""

    def test_creates_directory(self, tmp_path, tiny_model, tiny_model_config):
        out = tmp_path / "model_output"
        save_pretrained(tiny_model, out, model_config=tiny_model_config)
        assert out.is_dir()

    def test_config_json_valid(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        config_path = tmp_path / "m" / CONFIG_NAME
        assert config_path.exists()
        with open(config_path) as f:
            d = json.load(f)
        assert d["model_type"] == "transformer_encoder"
        assert d["model_config"]["vocab_size"] == 100

    def test_weights_file_present(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        d = tmp_path / "m"
        has_safetensors = (d / WEIGHTS_SAFETENSORS_NAME).exists()
        has_torch = (d / WEIGHTS_TORCH_NAME).exists()
        assert has_safetensors or has_torch

    def test_readme_present(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        assert (tmp_path / "m" / README_NAME).exists()

    def test_gene_vocab_saved(self, tmp_path, tiny_model, tiny_model_config, tiny_vocab):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config, gene_vocab=tiny_vocab)
        assert (tmp_path / "m" / VOCAB_NAME).exists()

    def test_gene_vocab_omitted(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        assert not (tmp_path / "m" / VOCAB_NAME).exists()

    def test_torch_fallback(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(
            tiny_model, tmp_path / "m", model_config=tiny_model_config, safe_serialization=False
        )
        assert (tmp_path / "m" / WEIGHTS_TORCH_NAME).exists()

    def test_vocab_size_none_raises(self, tmp_path, tiny_model):
        bad_config = ModelConfig(vocab_size=None)
        with pytest.raises(ValueError, match="vocab_size"):
            save_pretrained(tiny_model, tmp_path / "m", model_config=bad_config)

    def test_returns_path(self, tmp_path, tiny_model, tiny_model_config):
        result = save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        assert isinstance(result, Path)
        assert result == tmp_path / "m"


# ---------------------------------------------------------------------------
# TestLoadPretrained
# ---------------------------------------------------------------------------


class TestLoadPretrained:
    """Tests for load_pretrained()."""

    def test_roundtrip_transformer_encoder(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        loaded_model, config_dict = load_pretrained(tmp_path / "m")
        assert config_dict["model_type"] == "transformer_encoder"
        # Verify same class
        assert type(loaded_model).__name__ == "TransformerEncoder"

    def test_roundtrip_autoregressive(self, tmp_path, ar_model_config):
        model = get_model("autoregressive_transformer", ar_model_config)
        save_pretrained(model, tmp_path / "m", model_config=ar_model_config, safe_serialization=False)
        loaded_model, config_dict = load_pretrained(tmp_path / "m")
        assert config_dict["model_type"] == "autoregressive_transformer"
        assert type(loaded_model).__name__ == "AutoregressiveTransformer"

    def test_roundtrip_masked_autoencoder(self, tmp_path, mae_model_config):
        model = get_model("masked_autoencoder", mae_model_config)
        save_pretrained(model, tmp_path / "m", model_config=mae_model_config, safe_serialization=False)
        loaded_model, config_dict = load_pretrained(tmp_path / "m")
        assert config_dict["model_type"] == "masked_autoencoder"
        assert type(loaded_model).__name__ == "MaskedAutoencoder"

    def test_forward_pass_works(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        loaded_model, _ = load_pretrained(tmp_path / "m")
        loaded_model.eval()  # noqa: B023
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        out = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
        assert out.embeddings.shape == (batch_size, 32)

    def test_weights_match(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        loaded_model, _ = load_pretrained(tmp_path / "m")
        for (k1, v1), (k2, v2) in zip(
            sorted(tiny_model.state_dict().items()),
            sorted(loaded_model.state_dict().items()),
            strict=True,
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2), f"Mismatch in {k1}"

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pretrained("/nonexistent/path/to/model")


# ---------------------------------------------------------------------------
# TestLoadPretrainedWithVocab
# ---------------------------------------------------------------------------


class TestLoadPretrainedWithVocab:
    """Tests for load_pretrained_with_vocab()."""

    def test_vocab_loaded(self, tmp_path, tiny_model, tiny_model_config, tiny_vocab):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config, gene_vocab=tiny_vocab)
        _, _, vocab = load_pretrained_with_vocab(tmp_path / "m")
        assert vocab is not None
        assert len(vocab.genes) == 10

    def test_vocab_none_when_absent(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        _, _, vocab = load_pretrained_with_vocab(tmp_path / "m")
        assert vocab is None

    def test_gene_names_preserved(self, tmp_path, tiny_model, tiny_model_config, tiny_vocab):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config, gene_vocab=tiny_vocab)
        _, _, vocab = load_pretrained_with_vocab(tmp_path / "m")
        assert vocab.genes == [f"GENE{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# TestPushToHub
# ---------------------------------------------------------------------------


class TestPushToHub:
    """Tests for push_to_hub()."""

    def test_calls_create_repo_and_upload(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)

        mock_api = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(HfApi=MagicMock(return_value=mock_api))}):
            url = push_to_hub(tmp_path / "m", "user/test-model")

        assert "user/test-model" in url
        mock_api.create_repo.assert_called_once()
        mock_api.upload_folder.assert_called_once()

    def test_private_flag(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)

        mock_api = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(HfApi=MagicMock(return_value=mock_api))}):
            push_to_hub(tmp_path / "m", "user/test-model", private=True)

        call_kwargs = mock_api.create_repo.call_args
        assert call_kwargs.kwargs.get("private") is True or call_kwargs[1].get("private") is True

    def test_import_error_when_not_installed(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)

        with patch.dict("sys.modules", {"huggingface_hub": None}), pytest.raises(
            ImportError, match="huggingface-hub"
        ):
            push_to_hub(tmp_path / "m", "user/test-model")

    def test_nonexistent_dir_raises(self):
        mock_api = MagicMock()
        with patch.dict(
            "sys.modules", {"huggingface_hub": MagicMock(HfApi=MagicMock(return_value=mock_api))}
        ), pytest.raises(FileNotFoundError, match="does not exist"):
            push_to_hub("/nonexistent/dir", "user/test-model")


# ---------------------------------------------------------------------------
# TestIsHubRepoId
# ---------------------------------------------------------------------------


class TestIsHubRepoId:
    """Tests for _is_hub_repo_id()."""

    def test_local_dir_not_hub(self, tmp_path):
        d = tmp_path / "some" / "dir"
        d.mkdir(parents=True)
        assert _is_hub_repo_id(str(d)) is False

    def test_user_model_detected(self):
        assert _is_hub_repo_id("user/model-name") is True

    def test_simple_name_not_hub(self):
        assert _is_hub_repo_id("just-a-name") is False

    def test_local_file_not_hub(self, tmp_path):
        f = tmp_path / "path/to/file"
        f.parent.mkdir(parents=True)
        f.touch()
        assert _is_hub_repo_id(str(f)) is False


# ---------------------------------------------------------------------------
# TestModelCard
# ---------------------------------------------------------------------------


class TestModelCard:
    """Tests for auto-generated model card."""

    def test_yaml_frontmatter_present(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        readme = (tmp_path / "m" / README_NAME).read_text()
        assert readme.startswith("---")
        assert "library_name: scmodelforge" in readme

    def test_architecture_info_present(self, tmp_path, tiny_model, tiny_model_config):
        save_pretrained(tiny_model, tmp_path / "m", model_config=tiny_model_config)
        readme = (tmp_path / "m" / README_NAME).read_text()
        assert "transformer_encoder" in readme
        assert "32" in readme  # hidden_dim
