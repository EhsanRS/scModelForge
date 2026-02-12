"""Tests for cached offline preprocessing (data/preprocess.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import anndata as ad

if TYPE_CHECKING:
    from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scmodelforge.data.preprocess import preprocess_h5ad


@pytest.fixture()
def raw_adata(tmp_path: Path) -> str:
    """Create a small raw .h5ad file and return its path."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 50, 200
    X = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_type": rng.choice(["A", "B", "C"], size=n_cells)},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    path = str(tmp_path / "raw.h5ad")
    adata.write_h5ad(path)
    return path


class TestPreprocessH5ad:
    """Tests for preprocess_h5ad()."""

    def test_normalize_library_size(self, raw_adata: str, tmp_path: Path) -> None:
        """Output cell sums should approximately match target_sum."""
        out = str(tmp_path / "norm.h5ad")
        preprocess_h5ad(raw_adata, out, normalize="library_size", target_sum=1e4, log1p=False)
        result = ad.read_h5ad(out)
        sums = np.array(result.X.sum(axis=1)).flatten()
        np.testing.assert_allclose(sums, 1e4, rtol=1e-5)

    def test_log1p_applied(self, raw_adata: str, tmp_path: Path) -> None:
        """Values should be log-transformed (all non-negative, max < raw max)."""
        out = str(tmp_path / "log.h5ad")
        preprocess_h5ad(raw_adata, out, normalize=None, log1p=True)
        raw = ad.read_h5ad(raw_adata)
        result = ad.read_h5ad(out)
        # log1p values should be smaller than raw (for values > 0)
        assert result.X.max() < raw.X.max() or raw.X.max() <= 1.0
        assert result.X.min() >= 0.0

    def test_hvg_selection(self, raw_adata: str, tmp_path: Path) -> None:
        """Output should have fewer genes when HVG selection is applied."""
        out = str(tmp_path / "hvg.h5ad")
        n_hvg = 50
        preprocess_h5ad(raw_adata, out, normalize="library_size", log1p=True, hvg_n_top_genes=n_hvg)
        result = ad.read_h5ad(out)
        assert result.n_vars == n_hvg

    def test_no_normalize(self, raw_adata: str, tmp_path: Path) -> None:
        """Skipping normalization should leave raw counts (modulo log1p)."""
        out = str(tmp_path / "no_norm.h5ad")
        preprocess_h5ad(raw_adata, out, normalize=None, log1p=False)
        raw = ad.read_h5ad(raw_adata)
        result = ad.read_h5ad(out)
        np.testing.assert_array_equal(result.X, raw.X)

    def test_no_log1p(self, raw_adata: str, tmp_path: Path) -> None:
        """Skipping log1p should keep normalized values without log transform."""
        out = str(tmp_path / "no_log.h5ad")
        preprocess_h5ad(raw_adata, out, normalize="library_size", target_sum=1e4, log1p=False)
        result = ad.read_h5ad(out)
        sums = np.array(result.X.sum(axis=1)).flatten()
        np.testing.assert_allclose(sums, 1e4, rtol=1e-5)
        # Without log1p, some values can be > ln(1e4) ~ 9.2
        assert result.X.max() > 10.0

    def test_cloud_input(self, raw_adata: str, tmp_path: Path) -> None:
        """Cloud paths should be routed through cloud.read_h5ad."""
        out = str(tmp_path / "cloud_out.h5ad")
        mock_adata = ad.read_h5ad(raw_adata)

        with patch("scmodelforge.data.cloud.is_cloud_path", return_value=True) as _, patch(
            "scmodelforge.data.cloud.read_h5ad", return_value=mock_adata
        ) as mock_read:
            preprocess_h5ad("s3://bucket/data.h5ad", out, normalize=None, log1p=False)
            mock_read.assert_called_once_with("s3://bucket/data.h5ad")

        result = ad.read_h5ad(out)
        assert result.n_obs == mock_adata.n_obs

    def test_output_is_valid_h5ad(self, raw_adata: str, tmp_path: Path) -> None:
        """Output file should be a valid .h5ad that loads without error."""
        out = str(tmp_path / "valid.h5ad")
        preprocess_h5ad(raw_adata, out)
        result = ad.read_h5ad(out)
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert result.n_vars > 0

    def test_preserves_obs_metadata(self, raw_adata: str, tmp_path: Path) -> None:
        """obs columns should survive preprocessing."""
        out = str(tmp_path / "meta.h5ad")
        preprocess_h5ad(raw_adata, out)
        raw = ad.read_h5ad(raw_adata)
        result = ad.read_h5ad(out)
        assert list(result.obs.columns) == list(raw.obs.columns)
        pd.testing.assert_index_equal(result.obs.index, raw.obs.index)
        pd.testing.assert_series_equal(result.obs["cell_type"], raw.obs["cell_type"])
