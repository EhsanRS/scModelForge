"""End-to-end integration tests for the zoo module."""

from __future__ import annotations

import numpy as np
from click.testing import CliRunner


class TestHarnessRunExternal:
    """DummyAdapter -> harness.run_external() -> BenchmarkResult."""

    def test_run_external_produces_results(self, dummy_adapter, zoo_adata):
        from scmodelforge.config.schema import EvalConfig
        from scmodelforge.eval.base import BenchmarkResult
        from scmodelforge.eval.harness import EvalHarness

        config = EvalConfig(every_n_epochs=1, batch_size=16, benchmarks=["linear_probe"])
        harness = EvalHarness.from_config(config)

        results = harness.run_external(dummy_adapter, {"test_ds": zoo_adata})
        assert len(results) >= 1
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert results[0].dataset_name == "test_ds"

    def test_run_external_multiple_datasets(self, dummy_adapter, zoo_adata):
        from scmodelforge.config.schema import EvalConfig
        from scmodelforge.eval.harness import EvalHarness

        config = EvalConfig(every_n_epochs=1, batch_size=16, benchmarks=["linear_probe"])
        harness = EvalHarness.from_config(config)

        datasets = {"ds1": zoo_adata, "ds2": zoo_adata}
        results = harness.run_external(dummy_adapter, datasets)
        ds_names = {r.dataset_name for r in results}
        assert "ds1" in ds_names
        assert "ds2" in ds_names

    def test_run_external_with_device_and_batch(self, dummy_adapter, zoo_adata):
        from scmodelforge.config.schema import EvalConfig
        from scmodelforge.eval.harness import EvalHarness

        config = EvalConfig(every_n_epochs=1, batch_size=16, benchmarks=["linear_probe"])
        harness = EvalHarness.from_config(config)

        # Should not raise
        results = harness.run_external(
            dummy_adapter,
            {"test": zoo_adata},
            batch_size=8,
            device="cpu",
        )
        assert len(results) >= 1


class TestCLIBenchmarkExternal:
    """CLI benchmark --external-model integration."""

    def test_help_shows_external_options(self):
        from scmodelforge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--external-model" in result.output
        assert "--external-source" in result.output
        assert "--device" in result.output

    def test_external_model_requires_data(self):
        from scmodelforge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["benchmark", "--config", "x.yaml", "--external-model", "dummy"])
        # Should fail because config file doesn't exist or data missing
        assert result.exit_code != 0

    def test_model_required_without_external(self, tmp_path):
        """Without --external-model, --model is required."""
        # Create a minimal config file
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            "model:\n  architecture: transformer_encoder\n"
            "tokenizer:\n  strategy: rank_value\n"
        )

        # Create a minimal h5ad
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp

        adata = ad.AnnData(
            X=sp.csr_matrix(np.ones((5, 5), dtype=np.float32)),
            obs=pd.DataFrame({"cell_type": ["A"] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        )
        data_path = tmp_path / "data.h5ad"
        adata.write_h5ad(data_path)

        from scmodelforge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["benchmark", "--config", str(config_path), "--data", str(data_path)])
        assert result.exit_code != 0
        assert "--model is required" in result.output or "Missing" in result.output

    def test_external_model_cli_e2e(self, tmp_path, zoo_adata):
        """Full CLI flow with a DummyAdapter registered temporarily."""
        from scmodelforge.zoo.registry import _REGISTRY

        from .conftest import DummyAdapter

        # Temporarily register dummy adapter
        name = "_cli_test_dummy"
        _REGISTRY[name] = DummyAdapter

        try:
            # Write config and data
            config_path = tmp_path / "cfg.yaml"
            config_path.write_text(
                "model:\n  architecture: transformer_encoder\n"
                "tokenizer:\n  strategy: rank_value\n"
                "eval:\n  benchmarks: ['linear_probe']\n"
            )
            data_path = tmp_path / "data.h5ad"
            zoo_adata.write_h5ad(data_path)
            output_path = tmp_path / "results.json"

            from scmodelforge.cli import main

            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "benchmark",
                    "--config",
                    str(config_path),
                    "--data",
                    str(data_path),
                    "--external-model",
                    name,
                    "--output",
                    str(output_path),
                ],
            )
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert output_path.exists()

            import json

            with open(output_path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) >= 1
        finally:
            _REGISTRY.pop(name, None)
