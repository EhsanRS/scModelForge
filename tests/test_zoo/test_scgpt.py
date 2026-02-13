"""Tests for the scGPT adapter.

scgpt is an optional dependency, so we mock it entirely
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
# Mock scgpt
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_scgpt_packages():
    """Patch sys.modules so scgpt imports succeed."""
    import torch

    # --- Mock scgpt ---
    mock_scgpt = ModuleType("scgpt")
    mock_scgpt_model = ModuleType("scgpt.model")
    mock_scgpt_tokenizer = ModuleType("scgpt.tokenizer")

    # Mock GeneVocab
    mock_vocab_cls = MagicMock()
    mock_vocab_instance = MagicMock()

    # Create vocab with gene symbols
    gene_to_idx = {f"gene_{i}": i for i in range(100)}
    gene_to_idx["<pad>"] = 100
    gene_to_idx["<cls>"] = 101
    gene_to_idx["<eoc>"] = 102

    mock_vocab_instance.__contains__ = MagicMock(side_effect=lambda x: x in gene_to_idx)
    mock_vocab_instance.__getitem__ = MagicMock(side_effect=lambda x: gene_to_idx.get(x, -1))
    mock_vocab_instance.__len__ = MagicMock(return_value=len(gene_to_idx))
    mock_vocab_instance.get = MagicMock(side_effect=lambda x, default=-1: gene_to_idx.get(x, default))
    mock_vocab_instance.get_stoi = MagicMock(return_value=gene_to_idx)
    mock_vocab_instance.set_default_index = MagicMock()
    mock_vocab_instance.append_token = MagicMock()

    mock_vocab_cls.from_file = MagicMock(return_value=mock_vocab_instance)
    mock_scgpt_tokenizer.GeneVocab = mock_vocab_cls

    # Mock TransformerModel
    mock_model_cls = MagicMock()
    mock_model_instance = MagicMock()

    def _mock_encode(input_gene_ids, input_values, src_key_padding_mask=None, batch_labels=None):
        batch_size, seq_len = input_gene_ids.shape
        hidden = torch.randn(batch_size, seq_len, 512)
        return hidden

    mock_model_instance._encode = MagicMock(side_effect=_mock_encode)
    mock_model_instance.to = MagicMock(return_value=mock_model_instance)
    mock_model_instance.train = MagicMock(return_value=mock_model_instance)
    mock_model_instance.load_state_dict = MagicMock()
    mock_model_cls.return_value = mock_model_instance
    mock_scgpt_model.TransformerModel = mock_model_cls

    modules_patch = {
        "scgpt": mock_scgpt,
        "scgpt.model": mock_scgpt_model,
        "scgpt.tokenizer": mock_scgpt_tokenizer,
    }

    with patch.dict(sys.modules, modules_patch):
        yield {
            "model_instance": mock_model_instance,
            "vocab_instance": mock_vocab_instance,
            "TransformerModel": mock_model_cls,
            "GeneVocab": mock_vocab_cls,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScGPTAdapterInfo:
    def test_info_properties(self):
        from scmodelforge.zoo.scgpt import ScGPTAdapter

        adapter = ScGPTAdapter()
        info = adapter.info
        assert isinstance(info, ExternalModelInfo)
        assert info.name == "scgpt"
        assert info.gene_id_format == "symbol"
        assert info.hidden_dim == 512
        assert info.pip_package == "scgpt"
        assert info.supports_finetune is False

    def test_default_model_path(self):
        from scmodelforge.zoo.scgpt import ScGPTAdapter

        adapter = ScGPTAdapter()
        assert adapter._model_name_or_path == "scGPT_human"

    def test_custom_model_path(self):
        from scmodelforge.zoo.scgpt import ScGPTAdapter

        adapter = ScGPTAdapter(model_name_or_path="/local/scgpt")
        assert adapter._model_name_or_path == "/local/scgpt"


class TestScGPTAdapterImportGuard:
    def test_require_package_raises_without_scgpt(self):
        from scmodelforge.zoo.scgpt import ScGPTAdapter

        adapter = ScGPTAdapter()
        with patch.dict(sys.modules, {"scgpt": None}), pytest.raises(ImportError, match="scgpt"):
            adapter._require_package()


def _make_scgpt_model_dir(tmp_path):
    """Create a mock scGPT model directory with vocab, config, and model files."""
    import json

    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text("{}")
    config_file = tmp_path / "args.json"
    config_file.write_text(json.dumps({
        "embsize": 512,
        "nheads": 8,
        "d_hid": 512,
        "nlayers": 12,
        "n_layers_cls": 3,
        "dropout": 0.2,
        "pad_token": "<pad>",
        "pad_value": -2,
    }))
    model_file = tmp_path / "best_model.pt"
    model_file.write_text("")
    return tmp_path


class TestScGPTAdapterLoad:
    def test_load_model(self, _mock_scgpt_packages, tmp_path):
        from scmodelforge.zoo.scgpt import ScGPTAdapter

        model_dir = _make_scgpt_model_dir(tmp_path)
        adapter = ScGPTAdapter(model_name_or_path=str(model_dir))
        adapter._require_package()

        with patch("torch.load", return_value={}):
            adapter.load_model()
        assert adapter._model is not None
        assert adapter._vocab is not None

    def test_get_model_genes(self, _mock_scgpt_packages, tmp_path):
        from scmodelforge.zoo.scgpt import ScGPTAdapter

        model_dir = _make_scgpt_model_dir(tmp_path)
        adapter = ScGPTAdapter(model_name_or_path=str(model_dir))

        with patch("torch.load", return_value={}):
            adapter._ensure_loaded()
        genes = adapter._get_model_genes()
        assert len(genes) > 0


class TestScGPTAdapterEmbeddings:
    def test_extract_embeddings_shape(self, _mock_scgpt_packages, zoo_adata, tmp_path):
        from scmodelforge.zoo.scgpt import ScGPTAdapter

        model_dir = _make_scgpt_model_dir(tmp_path)
        adapter = ScGPTAdapter(model_name_or_path=str(model_dir))

        with patch("torch.load", return_value={}):
            embeddings = adapter.extract_embeddings(zoo_adata)
        assert embeddings.shape == (zoo_adata.n_obs, 512)
        assert embeddings.dtype == np.float32


class TestScGPTRegistry:
    def test_registered(self):
        from scmodelforge.zoo.registry import list_external_models

        assert "scgpt" in list_external_models()

    def test_get_by_name(self):
        from scmodelforge.zoo.registry import get_external_model

        adapter = get_external_model("scgpt")
        assert adapter.info.name == "scgpt"
