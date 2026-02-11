"""Smoke tests to verify basic package installation and CLI."""

from __future__ import annotations


def test_import():
    """Package can be imported and exposes a version string."""
    import scmodelforge

    assert hasattr(scmodelforge, "__version__")
    assert isinstance(scmodelforge.__version__, str)


def test_version_format():
    """Version follows semver-like X.Y.Z format."""
    from scmodelforge import __version__

    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_cli_help():
    """CLI --help exits cleanly."""
    from click.testing import CliRunner

    from scmodelforge.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "scModelForge" in result.output


def test_cli_version():
    """CLI --version prints the correct version."""
    from click.testing import CliRunner

    from scmodelforge.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_train_help():
    """CLI train --help exits cleanly."""
    from click.testing import CliRunner

    from scmodelforge.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_cli_benchmark_help():
    """CLI benchmark --help exits cleanly."""
    from click.testing import CliRunner

    from scmodelforge.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_config_schema_instantiation():
    """Default config schema can be instantiated."""
    from scmodelforge.config.schema import ScModelForgeConfig

    config = ScModelForgeConfig()
    assert config.data.source == "local"
    assert config.tokenizer.strategy == "rank_value"
    assert config.model.architecture == "transformer_encoder"
    assert config.training.batch_size == 64
    assert config.training.precision == "bf16-mixed"


def test_config_load_from_yaml(tmp_path):
    """Config can be loaded from a YAML file."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        """\
data:
  source: local
  paths:
    - /tmp/test.h5ad

model:
  hidden_dim: 256
  num_layers: 6

training:
  batch_size: 32
  max_epochs: 5
"""
    )

    from scmodelforge.config.schema import load_config

    config = load_config(str(config_path))
    assert config.data.paths == ["/tmp/test.h5ad"]
    assert config.model.hidden_dim == 256
    assert config.model.num_layers == 6
    assert config.training.batch_size == 32
    assert config.training.max_epochs == 5
    # Defaults should still be present
    assert config.tokenizer.strategy == "rank_value"


def test_mini_adata_fixture(mini_adata):
    """The mini_adata fixture produces valid AnnData."""
    assert mini_adata.shape == (100, 200)
    assert "cell_type" in mini_adata.obs.columns
    assert "batch" in mini_adata.obs.columns
    assert "gene_name" in mini_adata.var.columns
    assert "ensembl_id" in mini_adata.var.columns


def test_tmp_h5ad_fixture(tmp_h5ad):
    """The tmp_h5ad fixture writes a readable .h5ad file."""
    import anndata as ad

    adata = ad.read_h5ad(tmp_h5ad)
    assert adata.shape == (100, 200)
