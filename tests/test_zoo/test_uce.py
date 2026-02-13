"""Tests for the UCE adapter.

uce is an optional dependency, so we mock it entirely
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
# Mock uce
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_uce_packages():
    """Patch sys.modules so uce imports succeed."""
    import torch

    mock_uce = ModuleType("uce")

    # Mock model instance
    mock_model_instance = MagicMock()

    def _mock_get_cell_embeddings(batch_adata):
        n_cells = batch_adata.n_obs if hasattr(batch_adata, "n_obs") else len(batch_adata)
        return torch.randn(n_cells, 1280)

    mock_model_instance.get_cell_embeddings = MagicMock(side_effect=_mock_get_cell_embeddings)
    mock_model_instance.to = MagicMock(return_value=mock_model_instance)
    mock_model_instance.train = MagicMock(return_value=mock_model_instance)

    mock_uce.get_pretrained = MagicMock(return_value=mock_model_instance)

    modules_patch = {
        "uce": mock_uce,
    }

    with patch.dict(sys.modules, modules_patch):
        yield {
            "model_instance": mock_model_instance,
            "get_pretrained": mock_uce.get_pretrained,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUCEAdapterInfo:
    def test_info_properties(self):
        from scmodelforge.zoo.uce import UCEAdapter

        adapter = UCEAdapter()
        info = adapter.info
        assert isinstance(info, ExternalModelInfo)
        assert info.name == "uce"
        assert info.gene_id_format == "symbol"
        assert info.hidden_dim == 1280
        assert info.pip_package == "uce-model"
        assert info.supports_finetune is False
        assert "human" in info.species
        assert "mouse" in info.species

    def test_default_variant(self):
        from scmodelforge.zoo.uce import UCEAdapter

        adapter = UCEAdapter()
        assert adapter._model_variant == "large"

    def test_custom_variant(self):
        from scmodelforge.zoo.uce import UCEAdapter

        adapter = UCEAdapter(model_variant="small")
        assert adapter._model_variant == "small"


class TestUCEAdapterImportGuard:
    def test_require_package_raises_without_uce(self):
        from scmodelforge.zoo.uce import UCEAdapter

        adapter = UCEAdapter()
        with patch.dict(sys.modules, {"uce": None}), pytest.raises(ImportError, match="uce"):
            adapter._require_package()


class TestUCEAdapterLoad:
    def test_load_model(self, _mock_uce_packages):
        from scmodelforge.zoo.uce import UCEAdapter

        adapter = UCEAdapter()
        adapter._require_package()
        adapter.load_model()
        assert adapter._model is not None
        _mock_uce_packages["get_pretrained"].assert_called_once_with("large")

    def test_load_model_small(self, _mock_uce_packages):
        from scmodelforge.zoo.uce import UCEAdapter

        adapter = UCEAdapter(model_variant="small")
        adapter._require_package()
        adapter.load_model()
        _mock_uce_packages["get_pretrained"].assert_called_with("small")

    def test_get_model_genes_empty(self, _mock_uce_packages):
        """UCE has no fixed gene vocabulary (uses ESM2)."""
        from scmodelforge.zoo.uce import UCEAdapter

        adapter = UCEAdapter()
        adapter._ensure_loaded()
        genes = adapter._get_model_genes()
        assert genes == []


class TestUCEAdapterEmbeddings:
    def test_extract_embeddings_shape(self, _mock_uce_packages, zoo_adata):
        from scmodelforge.zoo.uce import UCEAdapter

        adapter = UCEAdapter()
        embeddings = adapter.extract_embeddings(zoo_adata)
        assert embeddings.shape == (zoo_adata.n_obs, 1280)
        assert embeddings.dtype == np.float32

    def test_extract_embeddings_custom_batch_size(self, _mock_uce_packages, zoo_adata):
        from scmodelforge.zoo.uce import UCEAdapter

        adapter = UCEAdapter()
        embeddings = adapter.extract_embeddings(zoo_adata, batch_size=5)
        assert embeddings.shape[0] == zoo_adata.n_obs


class TestUCERegistry:
    def test_registered(self):
        from scmodelforge.zoo.registry import list_external_models

        assert "uce" in list_external_models()

    def test_get_by_name(self):
        from scmodelforge.zoo.registry import get_external_model

        adapter = get_external_model("uce")
        assert adapter.info.name == "uce"
