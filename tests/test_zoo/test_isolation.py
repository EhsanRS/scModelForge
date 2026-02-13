"""Tests for zoo.isolation — subprocess isolation for external model adapters."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scmodelforge.zoo._env_registry import create_env_info, save_env_info
from scmodelforge.zoo.base import BaseModelAdapter, ExternalModelInfo
from scmodelforge.zoo.isolation import (
    IsolatedAdapter,
    _find_uv,
    _get_scmodelforge_src_path,
    install_env,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockAdapter(BaseModelAdapter):
    """Minimal adapter for testing isolation."""

    @property
    def info(self) -> ExternalModelInfo:
        return ExternalModelInfo(name="mock_iso", full_name="Mock Isolated", hidden_dim=16, pip_package="mock-pkg")

    @classmethod
    def isolation_deps(cls) -> list[str]:
        return ["mock-pkg>=1.0", "torch>=2.0"]

    def _require_package(self) -> None:
        pass

    def load_model(self) -> None:
        pass

    def _get_model_genes(self) -> list[str]:
        return []

    def extract_embeddings(
        self, adata: ad.AnnData, *, batch_size: int | None = None, device: str | None = None
    ) -> np.ndarray:
        return np.ones((adata.n_obs, 16), dtype=np.float32)


def _make_installed_env(tmp_path: Path, model_name: str = "mock_iso") -> None:
    """Create a fake installed environment directory."""
    info = create_env_info(model_name, base_dir=tmp_path, deps=["mock-pkg"], status="installed")
    # Create fake python and worker
    venv_bin = Path(info.venv_path) / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    python_path = venv_bin / "python"
    python_path.write_text("#!/bin/sh\n")
    python_path.chmod(0o755)
    info.python_path = str(python_path)
    # Copy worker script
    worker_src = Path(__file__).resolve().parents[2] / "src" / "scmodelforge" / "zoo" / "_worker.py"
    worker_dst = Path(info.env_path) / "worker.py"
    shutil.copy2(worker_src, worker_dst)
    save_env_info(info)


def _small_adata() -> ad.AnnData:
    rng = np.random.default_rng(42)
    X = sp.csr_matrix(rng.poisson(3.0, (5, 20)).astype(np.float32))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(5)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(20)])
    return ad.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Tests: _find_uv
# ---------------------------------------------------------------------------


class TestFindUv:
    def test_finds_uv_on_path(self) -> None:
        """If uv is installed, _find_uv returns a path."""
        # uv may or may not be installed — just test the function doesn't crash
        try:
            result = _find_uv()
            assert "uv" in result
        except FileNotFoundError:
            pytest.skip("uv not installed")

    def test_raises_when_uv_missing(self) -> None:
        with patch("shutil.which", return_value=None), pytest.raises(FileNotFoundError, match="uv"):
            _find_uv()


class TestGetSrcPath:
    def test_returns_src_directory(self) -> None:
        path = _get_scmodelforge_src_path()
        assert Path(path).is_dir()
        assert (Path(path) / "scmodelforge").is_dir()


# ---------------------------------------------------------------------------
# Tests: install_env
# ---------------------------------------------------------------------------


class TestInstallEnv:
    def test_install_with_mocked_uv(self, tmp_path) -> None:
        """install_env creates env directory and saves metadata."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

        with (
            patch("scmodelforge.zoo.isolation.subprocess.run", mock_run),
            patch("scmodelforge.zoo.isolation._find_uv", return_value="/usr/bin/uv"),
        ):
            install_env("mock_iso", adapter_cls=_MockAdapter, env_dir=tmp_path)

        # Check calls: one for venv creation, one for pip install
        assert mock_run.call_count == 2
        venv_call_args = mock_run.call_args_list[0]
        assert "venv" in venv_call_args[0][0]
        pip_call_args = mock_run.call_args_list[1]
        assert "install" in pip_call_args[0][0]
        # Should include base deps + isolation deps
        all_deps_str = " ".join(pip_call_args[0][0])
        assert "anndata" in all_deps_str
        assert "mock-pkg" in all_deps_str

        # Metadata saved
        from scmodelforge.zoo._env_registry import load_env_info

        info = load_env_info("mock_iso", base_dir=tmp_path)
        assert info is not None
        assert info.status == "installed"

    def test_install_with_python_version(self, tmp_path) -> None:
        """install_env passes --python flag when python_version is set."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

        with (
            patch("scmodelforge.zoo.isolation.subprocess.run", mock_run),
            patch("scmodelforge.zoo.isolation._find_uv", return_value="/usr/bin/uv"),
        ):
            install_env("mock_iso", adapter_cls=_MockAdapter, env_dir=tmp_path, python_version="3.10")

        venv_call_args = mock_run.call_args_list[0][0][0]
        assert "--python" in venv_call_args
        assert "3.10" in venv_call_args

    def test_install_with_extra_deps(self, tmp_path) -> None:
        """Extra deps are appended to install command."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

        with (
            patch("scmodelforge.zoo.isolation.subprocess.run", mock_run),
            patch("scmodelforge.zoo.isolation._find_uv", return_value="/usr/bin/uv"),
        ):
            install_env("mock_iso", adapter_cls=_MockAdapter, env_dir=tmp_path, extra_deps=["flash-attn>=2.0"])

        pip_call_args = mock_run.call_args_list[1][0][0]
        assert "flash-attn>=2.0" in pip_call_args

    def test_install_venv_failure_sets_error_status(self, tmp_path) -> None:
        """If venv creation fails, status is set to 'error'."""
        mock_run = MagicMock(return_value=MagicMock(returncode=1, stderr="venv creation failed"))

        with (
            patch("scmodelforge.zoo.isolation.subprocess.run", mock_run),
            patch("scmodelforge.zoo.isolation._find_uv", return_value="/usr/bin/uv"),
            pytest.raises(RuntimeError, match="Failed to create venv"),
        ):
            install_env("mock_iso", adapter_cls=_MockAdapter, env_dir=tmp_path)

        from scmodelforge.zoo._env_registry import load_env_info

        info = load_env_info("mock_iso", base_dir=tmp_path)
        assert info is not None
        assert info.status == "error"

    def test_install_pip_failure_sets_error_status(self, tmp_path) -> None:
        """If pip install fails, status is set to 'error'."""

        def side_effect(cmd, **kwargs):
            if "venv" in cmd:
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=1, stderr="pip install failed")

        mock_run = MagicMock(side_effect=side_effect)

        with (
            patch("scmodelforge.zoo.isolation.subprocess.run", mock_run),
            patch("scmodelforge.zoo.isolation._find_uv", return_value="/usr/bin/uv"),
            pytest.raises(RuntimeError, match="Failed to install"),
        ):
            install_env("mock_iso", adapter_cls=_MockAdapter, env_dir=tmp_path)

        from scmodelforge.zoo._env_registry import load_env_info

        info = load_env_info("mock_iso", base_dir=tmp_path)
        assert info is not None
        assert info.status == "error"

    def test_install_resolves_from_registry(self, tmp_path) -> None:
        """install_env can resolve adapter_cls from the registry."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

        with (
            patch("scmodelforge.zoo.isolation.subprocess.run", mock_run),
            patch("scmodelforge.zoo.isolation._find_uv", return_value="/usr/bin/uv"),
            patch("scmodelforge.zoo.registry.get_external_model", return_value=_MockAdapter()),
        ):
            install_env("mock_iso", env_dir=tmp_path)

        # Should have called subprocess.run for venv + pip
        assert mock_run.call_count == 2


# ---------------------------------------------------------------------------
# Tests: IsolatedAdapter
# ---------------------------------------------------------------------------


class TestIsolatedAdapterInit:
    def test_raises_when_env_not_installed(self, tmp_path) -> None:
        """IsolatedAdapter raises RuntimeError if env doesn't exist."""
        with pytest.raises(RuntimeError, match="No installed environment"):
            IsolatedAdapter("nonexistent", env_dir=tmp_path)

    def test_raises_when_env_status_error(self, tmp_path) -> None:
        """IsolatedAdapter raises RuntimeError if env has error status."""
        info = create_env_info("broken", base_dir=tmp_path, status="error")
        save_env_info(info)
        with pytest.raises(RuntimeError, match="No installed environment"):
            IsolatedAdapter("broken", env_dir=tmp_path)

    def test_init_with_installed_env(self, tmp_path) -> None:
        """IsolatedAdapter initializes successfully with installed env."""
        _make_installed_env(tmp_path)

        with patch("scmodelforge.zoo.registry.get_external_model") as mock_get:
            mock_get.return_value = _MockAdapter()
            adapter = IsolatedAdapter("mock_iso", env_dir=tmp_path)
            assert adapter.info.name == "mock_iso"
            assert adapter._adapter_module == _MockAdapter.__module__
            assert adapter._adapter_class == "_MockAdapter"

    def test_info_property(self, tmp_path) -> None:
        """IsolatedAdapter.info returns the wrapped adapter's info."""
        _make_installed_env(tmp_path)
        with patch("scmodelforge.zoo.registry.get_external_model") as mock_get:
            mock_get.return_value = _MockAdapter()
            adapter = IsolatedAdapter("mock_iso", env_dir=tmp_path)
            assert adapter.info.full_name == "Mock Isolated"
            assert adapter.info.hidden_dim == 16

    def test_noop_methods(self, tmp_path) -> None:
        """load_model, _require_package, _get_model_genes are no-ops."""
        _make_installed_env(tmp_path)
        with patch("scmodelforge.zoo.registry.get_external_model") as mock_get:
            mock_get.return_value = _MockAdapter()
            adapter = IsolatedAdapter("mock_iso", env_dir=tmp_path)
            adapter._require_package()
            adapter.load_model()
            assert adapter._get_model_genes() == []


class TestIsolatedAdapterExtraction:
    def test_extract_embeddings_mocked_subprocess(self, tmp_path) -> None:
        """extract_embeddings delegates to subprocess and reads output."""
        _make_installed_env(tmp_path)

        with patch("scmodelforge.zoo.registry.get_external_model") as mock_get:
            mock_get.return_value = _MockAdapter()
            adapter = IsolatedAdapter("mock_iso", env_dir=tmp_path)

        adata = _small_adata()
        expected = np.ones((5, 16), dtype=np.float32)

        def mock_subprocess_run(cmd, **kwargs):
            """Simulate the worker: read config, write output npy."""
            config_path = cmd[cmd.index("--config") + 1]
            with open(config_path) as f:
                config = json.load(f)
            np.save(config["output_npy"], expected)
            return MagicMock(returncode=0, stderr="")

        with patch("scmodelforge.zoo.isolation.subprocess.run", side_effect=mock_subprocess_run):
            result = adapter.extract_embeddings(adata)
            assert result.shape == (5, 16)
            assert np.allclose(result, 1.0)

    def test_extract_timeout_raises(self, tmp_path) -> None:
        """extract_embeddings raises TimeoutError on subprocess timeout."""
        _make_installed_env(tmp_path)

        with patch("scmodelforge.zoo.registry.get_external_model") as mock_get:
            mock_get.return_value = _MockAdapter()
            adapter = IsolatedAdapter("mock_iso", env_dir=tmp_path, timeout=1)

        adata = _small_adata()

        with (
            patch(
                "scmodelforge.zoo.isolation.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="test", timeout=1),
            ),
            pytest.raises(TimeoutError, match="timed out"),
        ):
            adapter.extract_embeddings(adata)

    def test_extract_nonzero_exit_raises(self, tmp_path) -> None:
        """extract_embeddings raises RuntimeError on non-zero exit."""
        _make_installed_env(tmp_path)

        with patch("scmodelforge.zoo.registry.get_external_model") as mock_get:
            mock_get.return_value = _MockAdapter()
            adapter = IsolatedAdapter("mock_iso", env_dir=tmp_path)

        adata = _small_adata()
        mock_result = MagicMock(returncode=1, stderr="ImportError: No module named 'geneformer'")

        with (
            patch("scmodelforge.zoo.isolation.subprocess.run", return_value=mock_result),
            pytest.raises(RuntimeError, match="failed"),
        ):
            adapter.extract_embeddings(adata)
