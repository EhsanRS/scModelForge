"""Shared fixtures for zoo module tests."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scmodelforge.zoo.base import BaseModelAdapter, ExternalModelInfo


class DummyAdapter(BaseModelAdapter):
    """Concrete adapter for testing — requires no external packages."""

    def __init__(
        self,
        model_name_or_path: str = "dummy/model",
        device: str = "cpu",
        batch_size: int = 32,
        hidden_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, device=device, batch_size=batch_size, **kwargs)
        self._hidden_dim = hidden_dim
        self._model_genes = [f"gene_{i}" for i in range(50)]

    @property
    def info(self) -> ExternalModelInfo:
        return ExternalModelInfo(
            name="dummy",
            full_name="Dummy Test Model",
            hidden_dim=self._hidden_dim,
            species=["human"],
            pip_package="",
            gene_id_format="symbol",
            supports_finetune=False,
        )

    def _require_package(self) -> None:
        pass  # No external package needed

    def _get_model_genes(self) -> list[str]:
        return list(self._model_genes)

    def load_model(self) -> None:
        pass  # Nothing to load

    def extract_embeddings(
        self,
        adata: ad.AnnData,
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> np.ndarray:
        self._ensure_loaded()
        rng = np.random.default_rng(42)
        return rng.standard_normal((adata.n_obs, self._hidden_dim)).astype(np.float32)


@pytest.fixture()
def dummy_adapter() -> DummyAdapter:
    """A DummyAdapter instance for testing."""
    return DummyAdapter()


@pytest.fixture()
def zoo_adata() -> ad.AnnData:
    """Small AnnData (20 cells x 50 genes) for zoo tests."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 20, 50

    X = sp.csr_matrix(rng.poisson(lam=3.0, size=(n_cells, n_genes)).astype(np.float32))
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    cell_types = ["type_A"] * 10 + ["type_B"] * 10
    batches = (["batch_0"] * 5 + ["batch_1"] * 5) * 2

    obs = pd.DataFrame(
        {"cell_type": cell_types, "batch": batches},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture()
def zoo_embeddings() -> np.ndarray:
    """20x64 numpy array for zoo integration tests."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((20, 64)).astype(np.float32)
    emb[:10, 0] += 5.0
    emb[10:, 0] -= 5.0
    return emb
