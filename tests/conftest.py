"""Shared test fixtures for scModelForge."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture()
def mini_adata() -> ad.AnnData:
    """A small AnnData object for testing (100 cells x 200 genes).

    Contains sparse expression matrix with ~30% non-zero entries,
    cell type labels, and batch labels in obs, plus gene name and
    Ensembl ID annotations in var.
    """
    rng = np.random.default_rng(42)
    n_obs, n_vars = 100, 200

    X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=42)

    obs = pd.DataFrame(
        {
            "cell_type": rng.choice(["T cell", "B cell", "Monocyte"], n_obs),
            "batch": rng.choice(["batch1", "batch2"], n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    var = pd.DataFrame(
        {
            "gene_name": [f"GENE_{i}" for i in range(n_vars)],
            "ensembl_id": [f"ENSG{i:011d}" for i in range(n_vars)],
        },
        index=[f"GENE_{i}" for i in range(n_vars)],
    )

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture()
def gene_vocab(mini_adata: ad.AnnData) -> dict[str, int]:
    """A simple gene name to index mapping derived from mini_adata."""
    return {g: i for i, g in enumerate(mini_adata.var_names)}


@pytest.fixture()
def tmp_h5ad(mini_adata: ad.AnnData, tmp_path) -> str:
    """Write mini_adata to a temporary .h5ad file and return the path."""
    path = tmp_path / "test.h5ad"
    mini_adata.write_h5ad(path)
    return str(path)
