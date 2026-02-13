"""Tests for the STACK adapter.

arc-stack is an optional dependency, so we mock it entirely
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
# Mock stack (arc-stack)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_stack_packages():
    """Patch sys.modules so stack imports succeed."""
    import torch

    mock_stack = ModuleType("stack")

    # Mock model instance
    mock_model_instance = MagicMock()

    def _mock_encode(chunk_tensor):
        # chunk_tensor shape: (batch=1, n_cells, n_genes)
        n_cells = chunk_tensor.shape[1]
        # Return (batch=1, n_cells, hidden_dim=100)
        return torch.randn(1, n_cells, 100)

    mock_model_instance.encode = MagicMock(side_effect=_mock_encode)
    mock_model_instance.to = MagicMock(return_value=mock_model_instance)
    mock_model_instance.train = MagicMock(return_value=mock_model_instance)
    mock_model_instance.gene_list = [f"gene_{i}" for i in range(50)]

    mock_stack.load_model = MagicMock(return_value=mock_model_instance)

    modules_patch = {
        "stack": mock_stack,
    }

    with patch.dict(sys.modules, modules_patch):
        yield {
            "model_instance": mock_model_instance,
            "load_model": mock_stack.load_model,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStackAdapterInfo:
    def test_info_properties(self):
        from scmodelforge.zoo.stack import StackAdapter

        adapter = StackAdapter()
        info = adapter.info
        assert isinstance(info, ExternalModelInfo)
        assert info.name == "stack"
        assert info.gene_id_format == "symbol"
        assert info.hidden_dim == 100
        assert info.pip_package == "arc-stack"
        assert info.supports_finetune is False

    def test_default_model_path(self):
        from scmodelforge.zoo.stack import StackAdapter

        adapter = StackAdapter()
        assert adapter._model_name_or_path == "ArcInstitute/stack-pretrained"

    def test_custom_sample_size(self):
        from scmodelforge.zoo.stack import StackAdapter

        adapter = StackAdapter(sample_size=128)
        assert adapter._sample_size == 128


class TestStackAdapterImportGuard:
    def test_require_package_raises_without_stack(self):
        from scmodelforge.zoo.stack import StackAdapter

        adapter = StackAdapter()
        with patch.dict(sys.modules, {"stack": None}), pytest.raises(ImportError, match="stack"):
            adapter._require_package()


class TestStackAdapterLoad:
    def test_load_model(self, _mock_stack_packages):
        from scmodelforge.zoo.stack import StackAdapter

        adapter = StackAdapter()
        adapter._require_package()
        adapter.load_model()
        assert adapter._model is not None
        assert len(adapter._gene_list) == 50

    def test_get_model_genes(self, _mock_stack_packages):
        from scmodelforge.zoo.stack import StackAdapter

        adapter = StackAdapter()
        adapter._ensure_loaded()
        genes = adapter._get_model_genes()
        assert len(genes) == 50


class TestStackAdapterEmbeddings:
    def test_extract_embeddings_shape(self, _mock_stack_packages, zoo_adata):
        from scmodelforge.zoo.stack import StackAdapter

        adapter = StackAdapter(sample_size=10)
        embeddings = adapter.extract_embeddings(zoo_adata)
        assert embeddings.shape == (zoo_adata.n_obs, 100)
        assert embeddings.dtype == np.float32

    def test_extract_embeddings_with_padding(self, _mock_stack_packages, zoo_adata):
        """Chunk size larger than n_cells should pad correctly."""
        from scmodelforge.zoo.stack import StackAdapter

        # sample_size > n_cells forces padding
        adapter = StackAdapter(sample_size=256)
        embeddings = adapter.extract_embeddings(zoo_adata)
        assert embeddings.shape == (zoo_adata.n_obs, 100)


class TestStackRegistry:
    def test_registered(self):
        from scmodelforge.zoo.registry import list_external_models

        assert "stack" in list_external_models()

    def test_get_by_name(self):
        from scmodelforge.zoo.registry import get_external_model

        adapter = get_external_model("stack")
        assert adapter.info.name == "stack"
