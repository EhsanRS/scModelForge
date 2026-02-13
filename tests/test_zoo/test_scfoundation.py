"""Tests for the scFoundation adapter.

scfoundation is an optional dependency, so we mock it entirely
using ``patch.dict("sys.modules", ...)``.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scmodelforge.zoo.base import ExternalModelInfo

# ---------------------------------------------------------------------------
# Mock scfoundation
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_scfoundation_packages():
    """Patch sys.modules so scfoundation imports succeed."""
    import torch

    mock_scfoundation = ModuleType("scfoundation")

    # Mock model instance
    mock_model_instance = MagicMock()

    def _mock_encode(batch_X):
        batch_size = batch_X.shape[0]
        # Return (batch, n_genes, hidden_dim) for max-pooling
        return torch.randn(batch_size, batch_X.shape[1], 768)

    mock_model_instance.encode = MagicMock(side_effect=_mock_encode)
    mock_model_instance.to = MagicMock(return_value=mock_model_instance)
    mock_model_instance.train = MagicMock(return_value=mock_model_instance)

    gene_list = [f"gene_{i}" for i in range(100)]
    mock_config = {"gene_list": gene_list}

    mock_scfoundation.load_model = MagicMock(return_value=(mock_model_instance, mock_config))
    mock_scfoundation.get_gene_list = MagicMock(return_value=gene_list)

    modules_patch = {
        "scfoundation": mock_scfoundation,
    }

    with patch.dict(sys.modules, modules_patch):
        yield {
            "model_instance": mock_model_instance,
            "load_model": mock_scfoundation.load_model,
            "gene_list": gene_list,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScFoundationAdapterInfo:
    def test_info_properties(self):
        from scmodelforge.zoo.scfoundation import ScFoundationAdapter

        adapter = ScFoundationAdapter()
        info = adapter.info
        assert isinstance(info, ExternalModelInfo)
        assert info.name == "scfoundation"
        assert info.gene_id_format == "symbol"
        assert info.hidden_dim == 768
        assert info.pip_package == "scfoundation"
        assert info.supports_finetune is False
        assert info.n_parameters == 100_000_000

    def test_default_model_path(self):
        from scmodelforge.zoo.scfoundation import ScFoundationAdapter

        adapter = ScFoundationAdapter()
        assert adapter._model_name_or_path == "genbio-ai/scFoundation"

    def test_custom_model_path(self):
        from scmodelforge.zoo.scfoundation import ScFoundationAdapter

        adapter = ScFoundationAdapter(model_name_or_path="/local/scfoundation")
        assert adapter._model_name_or_path == "/local/scfoundation"


class TestScFoundationAdapterImportGuard:
    def test_require_package_raises_without_scfoundation(self):
        from scmodelforge.zoo.scfoundation import ScFoundationAdapter

        adapter = ScFoundationAdapter()
        with patch.dict(sys.modules, {"scfoundation": None}), pytest.raises(ImportError, match="scfoundation"):
            adapter._require_package()


class TestScFoundationAdapterLoad:
    def test_load_model(self, _mock_scfoundation_packages):
        from scmodelforge.zoo.scfoundation import ScFoundationAdapter

        adapter = ScFoundationAdapter()
        adapter._require_package()
        adapter.load_model()
        assert adapter._model is not None
        assert len(adapter._gene_list) == 100

    def test_get_model_genes(self, _mock_scfoundation_packages):
        from scmodelforge.zoo.scfoundation import ScFoundationAdapter

        adapter = ScFoundationAdapter()
        adapter._ensure_loaded()
        genes = adapter._get_model_genes()
        assert len(genes) == 100


class TestScFoundationAdapterEmbeddings:
    def test_extract_embeddings_shape(self, _mock_scfoundation_packages, zoo_adata):
        from scmodelforge.zoo.scfoundation import ScFoundationAdapter

        adapter = ScFoundationAdapter()
        embeddings = adapter.extract_embeddings(zoo_adata)
        assert embeddings.shape == (zoo_adata.n_obs, 768)
        assert embeddings.dtype == np.float32

    def test_extract_embeddings_custom_batch_size(self, _mock_scfoundation_packages, zoo_adata):
        from scmodelforge.zoo.scfoundation import ScFoundationAdapter

        adapter = ScFoundationAdapter()
        embeddings = adapter.extract_embeddings(zoo_adata, batch_size=5)
        assert embeddings.shape[0] == zoo_adata.n_obs


class TestScFoundationRegistry:
    def test_registered(self):
        from scmodelforge.zoo.registry import list_external_models

        assert "scfoundation" in list_external_models()

    def test_get_by_name(self):
        from scmodelforge.zoo.registry import get_external_model

        adapter = get_external_model("scfoundation")
        assert adapter.info.name == "scfoundation"
