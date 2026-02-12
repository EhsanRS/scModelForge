"""Cloud filesystem utilities for S3/GCS/Azure access via fsspec.

Provides cloud URL detection, dependency checking, and a cloud-aware
AnnData reader that handles both standard and backed read modes.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

CLOUD_SCHEMES = ("s3://", "gs://", "gcs://", "az://", "abfs://")

_SCHEME_TO_PACKAGE: dict[str, str] = {
    "s3://": "s3fs",
    "gs://": "gcsfs",
    "gcs://": "gcsfs",
    "az://": "adlfs",
    "abfs://": "adlfs",
}

_SCHEME_TO_INSTALL: dict[str, str] = {
    "s3://": 'pip install "scModelForge[cloud]"  # or: pip install s3fs',
    "gs://": 'pip install "scModelForge[cloud]"  # or: pip install gcsfs',
    "gcs://": 'pip install "scModelForge[cloud]"  # or: pip install gcsfs',
    "az://": "pip install adlfs",
    "abfs://": "pip install adlfs",
}


def is_cloud_path(path: str | Path) -> bool:
    """Return ``True`` if *path* starts with a cloud URL scheme.

    Recognised schemes: ``s3://``, ``gs://``, ``gcs://``, ``az://``,
    ``abfs://``.
    """
    s = str(path)
    return any(s.startswith(scheme) for scheme in CLOUD_SCHEMES)


def require_fsspec(path: str) -> None:
    """Raise :class:`ImportError` with install instructions if fsspec or the
    scheme-specific backend is missing.

    Parameters
    ----------
    path
        A cloud URL whose scheme determines which backend package to check.
    """
    import importlib.util

    if importlib.util.find_spec("fsspec") is None:
        msg = (
            "fsspec is required for cloud storage access. "
            'Install it with: pip install "scModelForge[cloud]"  # or: pip install fsspec'
        )
        raise ImportError(msg)

    for scheme, package in _SCHEME_TO_PACKAGE.items():
        if path.startswith(scheme):
            if importlib.util.find_spec(package) is None:
                install_hint = _SCHEME_TO_INSTALL.get(scheme, f"pip install {package}")
                msg = (
                    f"The {package!r} package is required for {scheme} URLs. "
                    f"Install it with: {install_hint}"
                )
                raise ImportError(msg)
            return


def read_h5ad(
    path: str,
    storage_options: dict | None = None,
    backed: str | None = None,
    cache_dir: str | None = None,
) -> ad.AnnData:
    """Cloud-aware AnnData reader.

    Downloads the remote file to a local cache directory (hash-based
    filename for deduplication), then reads using :func:`anndata.read_h5ad`.
    This ensures compatibility with all anndata versions and supports
    both standard and backed read modes.

    Parameters
    ----------
    path
        Cloud URL (``s3://…``, ``gs://…``, etc.) or any fsspec-compatible URL.
    storage_options
        Dict forwarded to the fsspec filesystem backend for
        authentication / configuration.
    backed
        If ``"r"``, open in backed (memory-mapped) mode from the local
        cached copy.  ``None`` reads the full file into memory.
    cache_dir
        Local directory for caching cloud files.
        ``None`` uses a temporary directory.

    Returns
    -------
    anndata.AnnData
    """
    import anndata as _ad

    require_fsspec(path)

    # Always download to local cache first — ensures compatibility with
    # all anndata versions and avoids h5py issues with remote files.
    local_path = _download_to_cache(path, storage_options=storage_options, cache_dir=cache_dir)

    if backed is not None:
        logger.info("Reading backed from local cache: %s", local_path)
        return _ad.read_h5ad(local_path, backed=backed)

    logger.info("Reading cloud file via cache: %s", local_path)
    return _ad.read_h5ad(local_path)


def _download_to_cache(
    path: str,
    storage_options: dict | None = None,
    cache_dir: str | None = None,
) -> str:
    """Download a remote file to a local cache directory.

    Uses a hash-based filename so the same URL is only downloaded once
    per cache directory.

    Returns
    -------
    str
        Path to the local cached file.
    """
    import fsspec

    # Deterministic cache filename from the URL
    url_hash = hashlib.sha256(path.encode()).hexdigest()[:16]
    suffix = ".h5ad"

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        local_path = cache_path / f"{url_hash}{suffix}"
    else:
        # Use a persistent temp directory so repeated calls in the same
        # session find the cached file
        tmp_dir = Path(tempfile.gettempdir()) / "scmodelforge_cloud_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local_path = tmp_dir / f"{url_hash}{suffix}"

    if local_path.exists():
        logger.info("Using cached file: %s", local_path)
        return str(local_path)

    logger.info("Downloading %s → %s", path, local_path)
    fs, fs_path = fsspec.core.url_to_fs(path, **(storage_options or {}))
    fs.get(fs_path, str(local_path))

    return str(local_path)
