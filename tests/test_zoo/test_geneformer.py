"""Tests for the Geneformer adapter.

geneformer and transformers are optional dependencies, so we mock them
entirely using ``patch.dict("sys.modules", ...)``.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scmodelforge.zoo.base import ExternalModelInfo, GeneOverlapReport

# ---------------------------------------------------------------------------
# Mock geneformer + transformers
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_geneformer_packages():
    """Patch sys.modules so geneformer and transformers imports succeed."""
    # --- Mock transformers ---
    mock_transformers = ModuleType("transformers")

    mock_bert_model = MagicMock()
    # Model instance returned by BertModel.from_pretrained()
    mock_model_instance = MagicMock()
    mock_config = MagicMock()
    mock_config.max_position_embeddings = 2048
    mock_model_instance.config = mock_config

    # Mock forward pass: return object with last_hidden_state
    import torch

    def _mock_forward(input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, 256)
        result = MagicMock()
        result.last_hidden_state = hidden
        return result

    mock_model_instance.side_effect = _mock_forward
    mock_model_instance.to = MagicMock(return_value=mock_model_instance)
    mock_bert_model.from_pretrained = MagicMock(return_value=mock_model_instance)
    mock_transformers.BertModel = mock_bert_model

    # --- Mock geneformer ---
    mock_geneformer = ModuleType("geneformer")

    mock_tokenizer_cls = MagicMock()
    mock_tokenizer_instance = MagicMock()
    # Create gene token dict with Ensembl-style IDs
    mock_tokenizer_instance.gene_token_dict = {
        f"ENSG{i:011d}": i for i in range(100)
    }
    mock_tokenizer_instance.gene_median_dict = {
        f"ENSG{i:011d}": float(i + 1) for i in range(100)
    }
    mock_tokenizer_cls.return_value = mock_tokenizer_instance
    mock_geneformer.TranscriptomeTokenizer = mock_tokenizer_cls

    modules_patch = {
        "transformers": mock_transformers,
        "geneformer": mock_geneformer,
    }

    with patch.dict(sys.modules, modules_patch):
        yield {
            "BertModel": mock_bert_model,
            "model_instance": mock_model_instance,
            "tokenizer_instance": mock_tokenizer_instance,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGeneformerAdapterInfo:
    def test_info_properties(self):
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        adapter = GeneformerAdapter()
        info = adapter.info
        assert isinstance(info, ExternalModelInfo)
        assert info.name == "geneformer"
        assert info.gene_id_format == "ensembl"
        assert info.hidden_dim == 256
        assert info.pip_package == "geneformer"
        assert info.supports_finetune is False

    def test_default_model_path(self):
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        adapter = GeneformerAdapter()
        assert adapter._model_name_or_path == "ctheodoris/Geneformer"

    def test_custom_model_path(self):
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        adapter = GeneformerAdapter(model_name_or_path="/local/geneformer")
        assert adapter._model_name_or_path == "/local/geneformer"


class TestGeneformerAdapterImportGuard:
    def test_require_package_raises_without_geneformer(self):
        """extract_embeddings raises ImportError when geneformer is not installed."""
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        adapter = GeneformerAdapter()
        with patch.dict(sys.modules, {"geneformer": None}), pytest.raises(ImportError, match="geneformer"):
            adapter._require_package()

    def test_require_package_raises_without_transformers(self):
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        adapter = GeneformerAdapter()
        # geneformer exists but transformers doesn't
        mock_gf = ModuleType("geneformer")
        with patch.dict(sys.modules, {"geneformer": mock_gf, "transformers": None}), pytest.raises(
            ImportError, match="transformers"
        ):
            adapter._require_package()


class TestGeneformerAdapterLoad:
    def test_load_model(self, _mock_geneformer_packages):
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        adapter = GeneformerAdapter()
        adapter._require_package()
        adapter.load_model()
        assert adapter._model is not None
        assert adapter._token_dict is not None
        assert len(adapter._token_dict) == 100

    def test_get_model_genes(self, _mock_geneformer_packages):
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        adapter = GeneformerAdapter()
        adapter._ensure_loaded()
        genes = adapter._get_model_genes()
        assert len(genes) == 100
        assert all(g.startswith("ENSG") for g in genes)


class TestGeneformerAdapterEmbeddings:
    def test_extract_embeddings_shape(self, _mock_geneformer_packages, zoo_adata):
        """Embeddings have correct shape (n_cells, hidden_dim)."""
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        # Give adata Ensembl-style gene names matching the mock vocab
        zoo_adata.var_names = [f"ENSG{i:011d}" for i in range(zoo_adata.n_vars)]

        adapter = GeneformerAdapter()
        embeddings = adapter.extract_embeddings(zoo_adata)
        assert embeddings.shape == (zoo_adata.n_obs, 256)
        assert embeddings.dtype == np.float32

    def test_extract_embeddings_custom_batch_size(self, _mock_geneformer_packages, zoo_adata):
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        zoo_adata.var_names = [f"ENSG{i:011d}" for i in range(zoo_adata.n_vars)]

        adapter = GeneformerAdapter()
        embeddings = adapter.extract_embeddings(zoo_adata, batch_size=5)
        assert embeddings.shape[0] == zoo_adata.n_obs

    def test_gene_overlap_report(self, _mock_geneformer_packages, zoo_adata):
        from scmodelforge.zoo.geneformer import GeneformerAdapter

        # Only 50 of the 100 model genes are in adata
        zoo_adata.var_names = [f"ENSG{i:011d}" for i in range(zoo_adata.n_vars)]

        adapter = GeneformerAdapter()
        adapter._ensure_loaded()
        report = adapter.gene_overlap_report(zoo_adata)
        assert isinstance(report, GeneOverlapReport)
        assert report.matched == 50
        assert report.model_vocab_size == 100
        assert report.coverage == pytest.approx(0.5)


class TestGeneformerRegistry:
    def test_registered(self):
        from scmodelforge.zoo.registry import list_external_models

        assert "geneformer" in list_external_models()

    def test_get_by_name(self):
        from scmodelforge.zoo.registry import get_external_model

        adapter = get_external_model("geneformer")
        assert adapter.info.name == "geneformer"
