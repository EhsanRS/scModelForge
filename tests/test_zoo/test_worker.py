"""Tests for zoo._worker — subprocess worker for isolated embedding extraction."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _get_src_path() -> str:
    """Return the scModelForge source directory path."""
    return str(Path(__file__).resolve().parents[2] / "src")


def _make_adata(tmp_path: Path) -> Path:
    """Create a small test .h5ad and return its path."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 10, 50
    X = sp.csr_matrix(rng.poisson(3.0, (n_cells, n_genes)).astype(np.float32))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    h5ad_path = tmp_path / "input.h5ad"
    adata.write_h5ad(h5ad_path)
    return h5ad_path


WORKER_SCRIPT = str(Path(__file__).resolve().parents[2] / "src" / "scmodelforge" / "zoo" / "_worker.py")


class TestWorkerConfigParsing:
    """Worker config JSON parsing."""

    def test_missing_config_file_fails(self, tmp_path) -> None:
        """Worker exits non-zero when config file doesn't exist."""
        result = subprocess.run(
            [sys.executable, WORKER_SCRIPT, "--config", str(tmp_path / "missing.json")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_invalid_json_fails(self, tmp_path) -> None:
        """Worker exits non-zero on invalid JSON."""
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("{invalid}")
        result = subprocess.run(
            [sys.executable, WORKER_SCRIPT, "--config", str(bad_config)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_missing_required_key_fails(self, tmp_path) -> None:
        """Worker exits non-zero when required keys are missing."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"scmodelforge_src_path": _get_src_path()}))
        result = subprocess.run(
            [sys.executable, WORKER_SCRIPT, "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0


class TestWorkerE2E:
    """End-to-end worker tests using the DummyAdapter from conftest."""

    def test_extract_embeddings_with_dummy(self, tmp_path) -> None:
        """Worker successfully extracts embeddings using DummyAdapter from conftest."""
        h5ad_path = _make_adata(tmp_path)
        output_path = tmp_path / "output.npy"

        config = {
            "scmodelforge_src_path": _get_src_path(),
            # Use the conftest DummyAdapter — import path relative to tests/
            "adapter_module": "scmodelforge.zoo._worker_test_helper",
            "adapter_class": "WorkerTestAdapter",
            "adapter_kwargs": {},
            "input_h5ad": str(h5ad_path),
            "output_npy": str(output_path),
            "batch_size": 32,
            "device": "cpu",
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        # Create a minimal adapter module the worker can import
        helper_dir = Path(_get_src_path()) / "scmodelforge" / "zoo"
        helper_file = helper_dir / "_worker_test_helper.py"
        helper_existed = helper_file.exists()
        try:
            helper_file.write_text(
                'from __future__ import annotations\n'
                'from typing import Any\n'
                'import numpy as np\n'
                'from anndata import AnnData\n'
                'from scmodelforge.zoo.base import BaseModelAdapter, ExternalModelInfo\n'
                '\n'
                'class WorkerTestAdapter(BaseModelAdapter):\n'
                '    @property\n'
                '    def info(self) -> ExternalModelInfo:\n'
                '        return ExternalModelInfo(name="worker_test", hidden_dim=32)\n'
                '    def _require_package(self) -> None: pass\n'
                '    def load_model(self) -> None: pass\n'
                '    def _get_model_genes(self) -> list[str]: return []\n'
                '    def extract_embeddings(self, adata: AnnData, *, batch_size: int | None = None, device: str | None = None) -> np.ndarray:\n'
                '        self._ensure_loaded()\n'
                '        return np.ones((adata.n_obs, 32), dtype=np.float32)\n'
            )

            result = subprocess.run(
                [sys.executable, WORKER_SCRIPT, "--config", str(config_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, f"Worker failed: {result.stderr}"
            assert output_path.exists()

            embeddings = np.load(output_path)
            assert embeddings.shape == (10, 32)
            assert np.allclose(embeddings, 1.0)
        finally:
            if not helper_existed and helper_file.exists():
                helper_file.unlink()

    def test_worker_bad_module_fails(self, tmp_path) -> None:
        """Worker exits non-zero when adapter module doesn't exist."""
        h5ad_path = _make_adata(tmp_path)
        config = {
            "scmodelforge_src_path": _get_src_path(),
            "adapter_module": "scmodelforge.zoo.nonexistent_adapter_module",
            "adapter_class": "FakeAdapter",
            "adapter_kwargs": {},
            "input_h5ad": str(h5ad_path),
            "output_npy": str(tmp_path / "output.npy"),
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = subprocess.run(
            [sys.executable, WORKER_SCRIPT, "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "ModuleNotFoundError" in result.stderr or "No module" in result.stderr
