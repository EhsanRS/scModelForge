"""Tests for configuration schema and loading."""

from __future__ import annotations

from pathlib import Path

from scmodelforge.config.schema import ScModelForgeConfig, load_config

EXAMPLE_CONFIG = Path(__file__).parent.parent / "configs" / "examples" / "geneformer_basic.yaml"


def test_example_config_loads():
    """The shipped example config can be loaded without errors."""
    assert EXAMPLE_CONFIG.exists(), f"Example config not found at {EXAMPLE_CONFIG}"
    config = load_config(str(EXAMPLE_CONFIG))
    assert isinstance(config, ScModelForgeConfig)


def test_example_config_values():
    """The shipped example config has expected values."""
    config = load_config(str(EXAMPLE_CONFIG))

    assert config.data.source == "local"
    assert config.data.preprocessing.normalize == "library_size"
    assert config.data.preprocessing.hvg_selection == 2000

    assert config.tokenizer.strategy == "rank_value"
    assert config.tokenizer.max_genes == 2048
    assert config.tokenizer.masking.mask_ratio == 0.15

    assert config.model.architecture == "transformer_encoder"
    assert config.model.hidden_dim == 512
    assert config.model.num_layers == 12
    assert config.model.num_heads == 8

    assert config.training.batch_size == 64
    assert config.training.precision == "bf16-mixed"
    assert config.training.optimizer.name == "adamw"
    assert config.training.scheduler.name == "cosine_warmup"

    assert config.eval.every_n_epochs == 2
    assert len(config.eval.benchmarks) == 2


def test_config_defaults():
    """Default config has sensible values for all fields."""
    config = ScModelForgeConfig()

    assert config.data.source == "local"
    assert config.data.max_genes == 2048
    assert config.tokenizer.strategy == "rank_value"
    assert config.model.architecture == "transformer_encoder"
    assert config.model.hidden_dim == 512
    assert config.training.batch_size == 64
    assert config.training.seed == 42
    assert config.eval.every_n_epochs == 2


def test_config_partial_override(tmp_path):
    """Partial YAML overrides merge correctly with defaults."""
    config_path = tmp_path / "partial.yaml"
    config_path.write_text("model:\n  hidden_dim: 256\n  num_layers: 6\n")

    config = load_config(str(config_path))

    # Overridden values
    assert config.model.hidden_dim == 256
    assert config.model.num_layers == 6

    # Defaults preserved
    assert config.model.num_heads == 8
    assert config.model.architecture == "transformer_encoder"
    assert config.data.source == "local"
