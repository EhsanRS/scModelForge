"""Subprocess isolation for external model adapters.

Provides ``install_env()`` to create isolated virtualenvs for external
models and ``IsolatedAdapter`` to transparently run embedding extraction
in a subprocess using that environment.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scmodelforge.zoo._env_registry import (
    create_env_info,
    load_env_info,
    save_env_info,
)
from scmodelforge.zoo.base import BaseModelAdapter, ExternalModelInfo

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData

logger = logging.getLogger(__name__)

# Base dependencies always installed in every isolated env
_BASE_DEPS = ["anndata>=0.10", "numpy>=1.23", "scipy>=1.9"]


def _find_uv() -> str:
    """Locate the ``uv`` binary.

    Returns
    -------
    str
        Path to the ``uv`` executable.

    Raises
    ------
    FileNotFoundError
        If ``uv`` is not found on ``$PATH``.
    """
    uv_path = shutil.which("uv")
    if uv_path is None:
        msg = (
            "Could not find 'uv' on $PATH. Install it via:\n"
            "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "Or: pip install uv"
        )
        raise FileNotFoundError(msg)
    return uv_path


def _get_scmodelforge_src_path() -> str:
    """Return the path to scModelForge's parent ``src/`` directory.

    This is the directory that should be added to ``sys.path`` in the
    worker subprocess so that ``import scmodelforge.zoo.*`` works.
    """
    # This file is at src/scmodelforge/zoo/isolation.py
    # We need src/ (the parent of scmodelforge/)
    return str(Path(__file__).resolve().parents[2])


def install_env(
    model_name: str,
    *,
    adapter_cls: type[BaseModelAdapter] | None = None,
    extra_deps: list[str] | None = None,
    env_dir: str | Path | None = None,
    python_version: str | None = None,
    uv_path: str | None = None,
) -> None:
    """Create an isolated virtualenv for a zoo model.

    Parameters
    ----------
    model_name
        Registered model name (e.g. ``"geneformer"``).
    adapter_cls
        Adapter class. If ``None``, resolved from the registry.
    extra_deps
        Additional pip packages to install.
    env_dir
        Base directory for environments. Defaults to
        ``~/.cache/scmodelforge/envs``.
    python_version
        Python version for the venv (e.g. ``"3.10"``). If ``None``, uses
        the current Python version.
    uv_path
        Path to the ``uv`` binary. Auto-detected if ``None``.

    Raises
    ------
    FileNotFoundError
        If ``uv`` is not found.
    RuntimeError
        If venv creation or dependency installation fails.
    """
    uv = uv_path or _find_uv()

    # Resolve adapter class
    if adapter_cls is None:
        from scmodelforge.zoo.registry import get_external_model

        # get_external_model instantiates — we need the class
        adapter_instance = get_external_model(model_name)
        adapter_cls = type(adapter_instance)

    # Determine deps
    deps = list(adapter_cls.isolation_deps())
    if not deps:
        # Fallback: instantiate to read info.pip_package
        try:
            info = adapter_cls().info
            if info.pip_package:
                deps = [info.pip_package]
        except Exception:
            pass
    deps = list(_BASE_DEPS) + deps + (extra_deps or [])

    # Create env info
    info = create_env_info(
        model_name,
        base_dir=env_dir,
        deps=deps,
        python_version=python_version or "",
        status="installing",
    )
    save_env_info(info)

    env_path = Path(info.env_path)
    venv_path = Path(info.venv_path)

    try:
        # Create venv
        venv_cmd: list[str] = [uv, "venv", str(venv_path)]
        if python_version:
            venv_cmd.extend(["--python", python_version])
        logger.info("Creating venv: %s", " ".join(venv_cmd))
        result = subprocess.run(venv_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create venv:\n{result.stderr}")

        # Install deps
        python_path = str(venv_path / "bin" / "python")
        install_cmd = [uv, "pip", "install", "--python", python_path, *deps]
        logger.info("Installing deps: %s", " ".join(install_cmd))
        result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install dependencies:\n{result.stderr}")

        # Copy worker script into env directory
        worker_src = Path(__file__).parent / "_worker.py"
        worker_dst = env_path / "worker.py"
        shutil.copy2(worker_src, worker_dst)

        # Update status
        info.status = "installed"
        info.python_path = python_path
        save_env_info(info)
        logger.info("Environment for '%s' installed at %s", model_name, env_path)

    except Exception:
        info.status = "error"
        save_env_info(info)
        raise


class IsolatedAdapter(BaseModelAdapter):
    """Adapter wrapper that runs embedding extraction in a subprocess.

    The actual adapter runs inside an isolated virtualenv created by
    :func:`install_env`. This class handles serialization of the input
    data and config, subprocess invocation, and deserialization of the
    output embeddings.

    Parameters
    ----------
    model_name
        Registered model name (e.g. ``"geneformer"``).
    env_dir
        Base directory for environments.
    timeout
        Subprocess timeout in seconds. 0 means no timeout.
    **kwargs
        Additional keyword arguments forwarded to the adapter constructor
        in the subprocess.
    """

    def __init__(
        self,
        model_name: str,
        env_dir: str | Path | None = None,
        timeout: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name
        self._env_dir = env_dir
        self._timeout = timeout if timeout > 0 else None
        self._adapter_kwargs = kwargs

        # Verify environment is installed
        env_info = load_env_info(model_name, base_dir=env_dir)
        if env_info is None or env_info.status != "installed":
            msg = (
                f"No installed environment found for '{model_name}'. "
                f"Install it first with:\n"
                f"  scmodelforge zoo install {model_name}"
            )
            raise RuntimeError(msg)
        self._env_info = env_info

        # Get adapter metadata by briefly importing the adapter class
        from scmodelforge.zoo.registry import get_external_model

        adapter_instance = get_external_model(model_name)
        self._adapter_info = adapter_instance.info
        self._adapter_module = type(adapter_instance).__module__
        self._adapter_class = type(adapter_instance).__name__

    @property
    def info(self) -> ExternalModelInfo:
        return self._adapter_info

    def _require_package(self) -> None:
        pass  # Handled in the subprocess

    def load_model(self) -> None:
        pass  # Handled in the subprocess

    def _get_model_genes(self) -> list[str]:
        return []  # Would require subprocess call; not needed for embedding extraction

    def extract_embeddings(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> np.ndarray:
        """Extract embeddings by delegating to the isolated subprocess.

        Parameters
        ----------
        adata
            Input AnnData with cells in rows.
        batch_size
            Override default batch size.
        device
            Override default device.

        Returns
        -------
        np.ndarray
            Cell embeddings of shape ``(n_cells, hidden_dim)``.

        Raises
        ------
        TimeoutError
            If the subprocess exceeds the timeout.
        RuntimeError
            If the subprocess exits with a non-zero code.
        """
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="scmf_iso_") as tmpdir:
            tmp = Path(tmpdir)
            input_h5ad = tmp / "input.h5ad"
            output_npy = tmp / "output.npy"
            config_path = tmp / "config.json"

            # Write input data
            adata.write_h5ad(input_h5ad)

            # Write config
            config = {
                "scmodelforge_src_path": _get_scmodelforge_src_path(),
                "adapter_module": self._adapter_module,
                "adapter_class": self._adapter_class,
                "adapter_kwargs": self._adapter_kwargs,
                "input_h5ad": str(input_h5ad),
                "output_npy": str(output_npy),
                "batch_size": batch_size or self._batch_size,
                "device": device or self._device,
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            # Find worker script
            worker_path = Path(self._env_info.env_path) / "worker.py"
            if not worker_path.exists():
                # Fallback to the bundled worker
                worker_path = Path(__file__).parent / "_worker.py"

            # Run subprocess
            python_path = self._env_info.python_path
            cmd = [python_path, str(worker_path), "--config", str(config_path)]
            logger.info("Running isolated extraction: %s", " ".join(cmd))

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired as exc:
                msg = f"Isolated extraction for '{self._model_name}' timed out after {self._timeout}s"
                raise TimeoutError(msg) from exc

            if result.returncode != 0:
                msg = (
                    f"Isolated extraction for '{self._model_name}' failed "
                    f"(exit code {result.returncode}):\n{result.stderr}"
                )
                raise RuntimeError(msg)

            if not output_npy.exists():
                msg = f"Worker did not produce output file: {output_npy}"
                raise RuntimeError(msg)

            return np.load(output_npy)
