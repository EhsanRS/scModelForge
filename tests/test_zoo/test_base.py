"""Tests for BaseModelAdapter ABC and data types."""

from __future__ import annotations

import pytest

from scmodelforge.zoo.base import BaseModelAdapter, ExternalModelInfo, GeneOverlapReport


class TestExternalModelInfo:
    def test_defaults(self):
        info = ExternalModelInfo(name="test")
        assert info.name == "test"
        assert info.full_name == ""
        assert info.hidden_dim == 0
        assert info.species == ["human"]
        assert info.gene_id_format == "symbol"
        assert info.supports_finetune is False

    def test_custom_fields(self):
        info = ExternalModelInfo(
            name="test",
            full_name="Test Model",
            hidden_dim=256,
            species=["human", "mouse"],
            gene_id_format="ensembl",
            supports_finetune=True,
        )
        assert info.full_name == "Test Model"
        assert info.hidden_dim == 256
        assert info.species == ["human", "mouse"]
        assert info.gene_id_format == "ensembl"
        assert info.supports_finetune is True


class TestGeneOverlapReport:
    def test_fields(self):
        report = GeneOverlapReport(
            matched=80,
            missing=20,
            extra=50,
            coverage=0.8,
            model_vocab_size=100,
            adata_n_genes=130,
        )
        assert report.matched == 80
        assert report.missing == 20
        assert report.extra == 50
        assert report.coverage == pytest.approx(0.8)
        assert report.model_vocab_size == 100
        assert report.adata_n_genes == 130


class TestBaseModelAdapterABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseModelAdapter()  # type: ignore[abstract]

    def test_abstract_methods_enforced(self):
        """Subclass missing any abstract method cannot be instantiated."""

        class IncompleteAdapter(BaseModelAdapter):
            @property
            def info(self):
                return ExternalModelInfo(name="incomplete")

            def _require_package(self):
                pass

            # Missing: load_model, extract_embeddings, _get_model_genes

        with pytest.raises(TypeError, match="abstract"):
            IncompleteAdapter()  # type: ignore[abstract]

    def test_concrete_subclass_works(self, dummy_adapter):
        assert dummy_adapter.info.name == "dummy"

    def test_get_backbone_raises_not_implemented(self, dummy_adapter):
        with pytest.raises(NotImplementedError, match="does not support fine-tuning"):
            dummy_adapter.get_backbone()

    def test_ensure_loaded_calls_once(self, dummy_adapter):
        """_ensure_loaded() calls _require_package and load_model only once."""
        import unittest.mock as mock

        with mock.patch.object(dummy_adapter, "_require_package") as m_req, mock.patch.object(
            dummy_adapter, "load_model"
        ) as m_load:
            dummy_adapter._loaded = False
            dummy_adapter._ensure_loaded()
            m_req.assert_called_once()
            m_load.assert_called_once()

            # Second call should be no-op
            dummy_adapter._ensure_loaded()
            m_req.assert_called_once()
            m_load.assert_called_once()

    def test_gene_overlap_report(self, dummy_adapter, zoo_adata):
        report = dummy_adapter.gene_overlap_report(zoo_adata)
        assert isinstance(report, GeneOverlapReport)
        assert report.matched == 50  # All genes overlap with dummy's vocab
        assert report.coverage == pytest.approx(1.0)

    def test_gene_overlap_report_partial(self, dummy_adapter, zoo_adata):
        """When adata has different genes, overlap is partial."""
        # Rename half the genes in adata to create partial overlap
        new_var_names = [f"gene_{i}" for i in range(25)] + [f"other_{i}" for i in range(25)]
        zoo_adata.var_names = new_var_names
        report = dummy_adapter.gene_overlap_report(zoo_adata)
        assert report.matched == 25
        assert report.missing == 25
        assert report.extra == 25
        assert report.coverage == pytest.approx(0.5)
