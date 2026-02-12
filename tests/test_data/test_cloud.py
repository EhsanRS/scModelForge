"""Tests for cloud filesystem utilities (S3/GCS/Azure via fsspec)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scmodelforge.data.cloud import (
    CLOUD_SCHEMES,
    is_cloud_path,
    read_h5ad,
    require_fsspec,
)

# ---------------------------------------------------------------------------
# TestIsCloudPath
# ---------------------------------------------------------------------------


class TestIsCloudPath:
    def test_s3_url(self):
        assert is_cloud_path("s3://bucket/path/data.h5ad") is True

    def test_gs_url(self):
        assert is_cloud_path("gs://bucket/data.h5ad") is True

    def test_gcs_url(self):
        assert is_cloud_path("gcs://bucket/data.h5ad") is True

    def test_az_url(self):
        assert is_cloud_path("az://container/data.h5ad") is True

    def test_abfs_url(self):
        assert is_cloud_path("abfs://container/data.h5ad") is True

    def test_local_absolute_path(self):
        assert is_cloud_path("/home/user/data.h5ad") is False

    def test_local_relative_path(self):
        assert is_cloud_path("data/file.h5ad") is False

    def test_path_object(self):
        assert is_cloud_path(Path("/tmp/data.h5ad")) is False

    def test_all_schemes_recognised(self):
        for scheme in CLOUD_SCHEMES:
            assert is_cloud_path(f"{scheme}bucket/key") is True


# ---------------------------------------------------------------------------
# TestRequireFsspec
# ---------------------------------------------------------------------------


class TestRequireFsspec:
    def test_missing_fsspec_raises(self):
        with patch("importlib.util.find_spec", return_value=None), pytest.raises(ImportError, match="fsspec"):
            require_fsspec("s3://bucket/data.h5ad")

    def test_missing_s3fs_raises(self):
        def _find_spec(name):
            if name == "s3fs":
                return None
            # Return a truthy sentinel for fsspec
            return True

        with patch("importlib.util.find_spec", side_effect=_find_spec), pytest.raises(ImportError, match="s3fs"):
            require_fsspec("s3://bucket/data.h5ad")

    def test_missing_gcsfs_raises(self):
        def _find_spec(name):
            if name == "gcsfs":
                return None
            return True

        with patch("importlib.util.find_spec", side_effect=_find_spec), pytest.raises(ImportError, match="gcsfs"):
            require_fsspec("gs://bucket/data.h5ad")

    def test_all_present_passes(self):
        # Should not raise when all deps are available
        with patch("importlib.util.find_spec", return_value=True):
            require_fsspec("s3://bucket/data.h5ad")


# ---------------------------------------------------------------------------
# TestReadH5adCloud — uses fsspec memory:// filesystem
# ---------------------------------------------------------------------------


def _make_test_adata(n_cells: int = 10, n_genes: int = 5) -> ad.AnnData:
    """Create a small in-memory AnnData for testing."""
    rng = np.random.default_rng(42)
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    return ad.AnnData(
        X=X,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
    )


@pytest.fixture()
def memory_h5ad(tmp_path):
    """Write a small .h5ad to the fsspec memory filesystem and return its URL."""
    import fsspec

    adata = _make_test_adata(n_cells=10, n_genes=5)

    # Write to a local temp file first, then copy to memory://
    local_path = tmp_path / "test.h5ad"
    adata.write_h5ad(local_path)

    fs = fsspec.filesystem("memory")
    with open(local_path, "rb") as f:
        data = f.read()
    fs.pipe("memory://test_bucket/test.h5ad", data)

    yield "memory://test_bucket/test.h5ad"

    # Cleanup
    with contextlib.suppress(Exception):
        fs.rm("memory://test_bucket/test.h5ad")


class TestReadH5adCloud:
    def test_non_backed_read(self, memory_h5ad):
        """Non-backed read from memory:// should return valid AnnData."""
        # memory:// is not in CLOUD_SCHEMES, so we need to bypass require_fsspec
        with patch("scmodelforge.data.cloud.require_fsspec"):
            result = read_h5ad(memory_h5ad)
        assert isinstance(result, ad.AnnData)
        assert result.shape == (10, 5)

    def test_backed_read_with_cache_dir(self, memory_h5ad, tmp_path):
        """Backed read should download to cache_dir then read backed."""
        cache_dir = tmp_path / "cache"
        with patch("scmodelforge.data.cloud.require_fsspec"):
            result = read_h5ad(memory_h5ad, backed="r", cache_dir=str(cache_dir))
        assert isinstance(result, ad.AnnData)
        assert result.shape == (10, 5)
        # Verify a cached file was created
        cached_files = list(cache_dir.glob("*.h5ad"))
        assert len(cached_files) == 1

    def test_backed_read_without_cache_uses_tempdir(self, memory_h5ad):
        """Backed read without cache_dir should use temp directory."""
        with patch("scmodelforge.data.cloud.require_fsspec"):
            result = read_h5ad(memory_h5ad, backed="r")
        assert isinstance(result, ad.AnnData)
        assert result.shape == (10, 5)

    def test_cache_deduplication(self, memory_h5ad, tmp_path):
        """Same URL should reuse cached file on second read."""
        cache_dir = tmp_path / "cache"
        with patch("scmodelforge.data.cloud.require_fsspec"):
            read_h5ad(memory_h5ad, backed="r", cache_dir=str(cache_dir))
            # Second read should hit the cache
            result = read_h5ad(memory_h5ad, backed="r", cache_dir=str(cache_dir))
        assert isinstance(result, ad.AnnData)
        assert result.shape == (10, 5)
        # Still only one cached file
        cached_files = list(cache_dir.glob("*.h5ad"))
        assert len(cached_files) == 1

    def test_storage_options_forwarded(self, tmp_path):
        """storage_options should be passed to fsspec."""
        adata = _make_test_adata(n_cells=5, n_genes=3)
        local_path = tmp_path / "opts_test.h5ad"
        adata.write_h5ad(local_path)

        # Use local filesystem via file:// URL (with storage_options)
        file_url = f"file://{local_path}"
        with patch("scmodelforge.data.cloud.require_fsspec"):
            result = read_h5ad(file_url, storage_options={})
        assert isinstance(result, ad.AnnData)
        assert result.shape == (5, 3)


# ---------------------------------------------------------------------------
# TestIntegration — load_adata, AnnDataStore, StreamingCellDataset
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_load_adata_dispatches_cloud_path(self, memory_h5ad):
        """load_adata should detect cloud URLs and use cloud reader."""
        from scmodelforge.config.schema import DataConfig
        from scmodelforge.data._utils import load_adata

        config = DataConfig(source="local", paths=[memory_h5ad])

        # memory:// is not in CLOUD_SCHEMES, so mock is_cloud_path at source
        with (
            patch("scmodelforge.data.cloud.is_cloud_path", return_value=True),
            patch("scmodelforge.data.cloud.require_fsspec"),
        ):
            result = load_adata(config)
        assert isinstance(result, ad.AnnData)
        assert result.shape == (10, 5)

    def test_anndata_store_load_handles_cloud(self, memory_h5ad):
        """AnnDataStore._load should handle cloud URLs without Path() crash."""
        from scmodelforge.data.anndata_store import AnnDataStore

        # memory:// is not in CLOUD_SCHEMES, so mock is_cloud_path at source
        with (
            patch("scmodelforge.data.cloud.is_cloud_path", return_value=True),
            patch("scmodelforge.data.cloud.require_fsspec"),
        ):
            result = AnnDataStore._load(memory_h5ad)
        assert isinstance(result, ad.AnnData)
        assert result.shape == (10, 5)

    def test_streaming_dataset_accepts_storage_options(self):
        """StreamingCellDataset should accept storage_options and cache_dir."""
        from scmodelforge.data.gene_vocab import GeneVocab
        from scmodelforge.data.streaming import StreamingCellDataset

        vocab = GeneVocab.from_genes(["GENE_A", "GENE_B"])
        ds = StreamingCellDataset(
            file_paths=["s3://bucket/data.h5ad"],
            gene_vocab=vocab,
            storage_options={"key": "test"},
            cache_dir="/tmp/cache",
        )
        assert ds.storage_options == {"key": "test"}
        assert ds.cache_dir == "/tmp/cache"

    def test_cloud_storage_config_defaults(self):
        """CloudStorageConfig should have sensible defaults."""
        from scmodelforge.config.schema import CloudStorageConfig, DataConfig

        cloud = CloudStorageConfig()
        assert cloud.storage_options == {}
        assert cloud.cache_dir is None

        data = DataConfig()
        assert data.cloud.storage_options == {}
        assert data.cloud.cache_dir is None
