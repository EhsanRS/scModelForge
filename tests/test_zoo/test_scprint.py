"""Tests for the scPRINT adapter.

scprint is an optional dependency, so we mock it entirely
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
# Mock scprint
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_scprint_packages():
    """Patch sys.modules so scprint imports succeed."""
    import torch

    mock_scprint = ModuleType("scprint")
    mock_scprint_model = ModuleType("scprint.model")
    mock_scprint_tasks = ModuleType("scprint.tasks")

    # Mock scPrint model class
    mock_model_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.to = MagicMock(return_value=mock_model_instance)
    mock_model_instance.train = MagicMock(return_value=mock_model_instance)
    mock_model_instance.genes = [f"gene_{i}" for i in range(100)]

    mock_model_cls.from_pretrained = MagicMock(return_value=mock_model_instance)
    mock_scprint_model.scPrint = mock_model_cls

    # Mock Embedder
    mock_embedder_cls = MagicMock()
    mock_embedder_instance = MagicMock()

    def _mock_embed(model, adata):
        return torch.randn(adata.n_obs, 512)

    mock_embedder_instance.embed = MagicMock(side_effect=_mock_embed)
    mock_embedder_cls.return_value = mock_embedder_instance
    mock_scprint_tasks.Embedder = mock_embedder_cls

    modules_patch = {
        "scprint": mock_scprint,
        "scprint.model": mock_scprint_model,
        "scprint.tasks": mock_scprint_tasks,
    }

    with patch.dict(sys.modules, modules_patch):
        yield {
            "model_instance": mock_model_instance,
            "scPrint": mock_model_cls,
            "Embedder": mock_embedder_cls,
            "embedder_instance": mock_embedder_instance,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScPRINTAdapterInfo:
    def test_info_properties(self):
        from scmodelforge.zoo.scprint import ScPRINTAdapter

        adapter = ScPRINTAdapter()
        info = adapter.info
        assert isinstance(info, ExternalModelInfo)
        assert info.name == "scprint"
        assert info.gene_id_format == "symbol"
        assert info.hidden_dim == 512
        assert info.pip_package == "scprint"
        assert info.supports_finetune is False

    def test_default_model_path(self):
        from scmodelforge.zoo.scprint import ScPRINTAdapter

        adapter = ScPRINTAdapter()
        assert adapter._model_name_or_path == "jkobject/scPRINT"

    def test_custom_model_path(self):
        from scmodelforge.zoo.scprint import ScPRINTAdapter

        adapter = ScPRINTAdapter(model_name_or_path="/local/scprint")
        assert adapter._model_name_or_path == "/local/scprint"


class TestScPRINTAdapterImportGuard:
    def test_require_package_raises_without_scprint(self):
        from scmodelforge.zoo.scprint import ScPRINTAdapter

        adapter = ScPRINTAdapter()
        with patch.dict(sys.modules, {"scprint": None}), pytest.raises(ImportError, match="scprint"):
            adapter._require_package()


class TestScPRINTAdapterLoad:
    def test_load_model(self, _mock_scprint_packages):
        from scmodelforge.zoo.scprint import ScPRINTAdapter

        adapter = ScPRINTAdapter()
        adapter._require_package()
        adapter.load_model()
        assert adapter._model is not None
        assert len(adapter._gene_names) == 100

    def test_get_model_genes(self, _mock_scprint_packages):
        from scmodelforge.zoo.scprint import ScPRINTAdapter

        adapter = ScPRINTAdapter()
        adapter._ensure_loaded()
        genes = adapter._get_model_genes()
        assert len(genes) == 100


class TestScPRINTAdapterEmbeddings:
    def test_extract_embeddings_shape(self, _mock_scprint_packages, zoo_adata):
        from scmodelforge.zoo.scprint import ScPRINTAdapter

        adapter = ScPRINTAdapter()
        embeddings = adapter.extract_embeddings(zoo_adata)
        assert embeddings.shape == (zoo_adata.n_obs, 512)
        assert embeddings.dtype == np.float32

    def test_extract_embeddings_custom_batch_size(self, _mock_scprint_packages, zoo_adata):
        from scmodelforge.zoo.scprint import ScPRINTAdapter

        adapter = ScPRINTAdapter()
        embeddings = adapter.extract_embeddings(zoo_adata, batch_size=5)
        assert embeddings.shape[0] == zoo_adata.n_obs


class TestScPRINTRegistry:
    def test_registered(self):
        from scmodelforge.zoo.registry import list_external_models

        assert "scprint" in list_external_models()

    def test_get_by_name(self):
        from scmodelforge.zoo.registry import get_external_model

        adapter = get_external_model("scprint")
        assert adapter.info.name == "scprint"
