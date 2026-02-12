"""Tests for OrthologMapper and multi-species gene vocabulary."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scmodelforge._constants import NUM_SPECIAL_TOKENS
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.data.ortholog_mapper import OrthologMapper

# ------------------------------------------------------------------
# OrthologMapper construction
# ------------------------------------------------------------------


class TestOrthologMapperConstruction:
    def test_default_construction(self):
        mapper = OrthologMapper()
        assert mapper.organisms == ["human", "mouse"]
        assert repr(mapper).startswith("OrthologMapper(")

    def test_lazy_loading(self):
        mapper = OrthologMapper()
        # Table should not be loaded until first access
        assert mapper._table is None
        # Accessing n_mapped triggers loading
        n = mapper.n_mapped
        assert n > 0
        assert mapper._table is not None

    def test_from_config(self):
        from scmodelforge.config.schema import MultiSpeciesConfig

        config = MultiSpeciesConfig(
            enabled=True,
            organisms=["human", "mouse"],
            canonical_organism="human",
            include_one2many=True,
        )
        mapper = OrthologMapper.from_config(config)
        assert mapper.organisms == ["human", "mouse"]
        assert mapper._include_one2many is True

    def test_custom_table_path(self, tmp_path):
        # Create a minimal ortholog table
        tsv = tmp_path / "custom.tsv"
        tsv.write_text(
            "human_gene_symbol\tmouse_gene_symbol\thuman_ensembl_id\tmouse_ensembl_id\torthology_type\n"
            "TP53\tTrp53\tENSG00000141510\tENSMUSG00000059552\tone2one\n"
            "MYC\tMyc\tENSG00000136997\tENSMUSG00000022346\tone2one\n"
        )
        mapper = OrthologMapper(ortholog_table_path=tsv)
        assert mapper.n_mapped == 2

    def test_missing_table_raises(self, tmp_path):
        mapper = OrthologMapper(ortholog_table_path=tmp_path / "nonexistent.tsv")
        with pytest.raises(FileNotFoundError, match="Ortholog table not found"):
            _ = mapper.n_mapped

    def test_repr(self):
        mapper = OrthologMapper()
        r = repr(mapper)
        assert "organisms=" in r
        assert "canonical=" in r


# ------------------------------------------------------------------
# Gene name translation
# ------------------------------------------------------------------


class TestTranslation:
    def test_human_identity(self):
        mapper = OrthologMapper()
        names = ["TP53", "BRCA1", "EGFR"]
        result = mapper.translate_gene_names(names, source_organism="human")
        assert result == names

    def test_mouse_to_human(self):
        mapper = OrthologMapper()
        result = mapper.translate_gene_names(["Trp53"], source_organism="mouse")
        assert result == ["TP53"]

    def test_mouse_to_human_batch(self):
        mapper = OrthologMapper()
        result = mapper.translate_gene_names(
            ["Trp53", "Brca1", "Egfr"], source_organism="mouse"
        )
        assert result == ["TP53", "BRCA1", "EGFR"]

    def test_unknown_mouse_gene_passes_through(self):
        mapper = OrthologMapper()
        result = mapper.translate_gene_names(
            ["Trp53", "NotARealGene123"], source_organism="mouse"
        )
        assert result[0] == "TP53"
        assert result[1] == "NotARealGene123"

    def test_unsupported_organism_raises(self):
        mapper = OrthologMapper()
        with pytest.raises(ValueError, match="Unsupported source organism"):
            mapper.translate_gene_names(["X"], source_organism="zebrafish")

    def test_one2many_filtering(self):
        """Without one2many, only one2one orthologs are available."""
        mapper_strict = OrthologMapper(include_one2many=False)
        mapper_loose = OrthologMapper(include_one2many=True)
        assert mapper_loose.n_mapped >= mapper_strict.n_mapped


# ------------------------------------------------------------------
# Canonical gene list
# ------------------------------------------------------------------


class TestCanonicalGenes:
    def test_get_all_canonical_genes(self):
        mapper = OrthologMapper()
        genes = mapper.get_all_canonical_genes()
        assert len(genes) > 0
        # Should be sorted
        assert genes == sorted(genes)
        # Should contain known human genes
        assert "TP53" in genes
        assert "BRCA1" in genes

    def test_canonical_genes_no_duplicates(self):
        mapper = OrthologMapper()
        genes = mapper.get_all_canonical_genes()
        assert len(genes) == len(set(genes))


# ------------------------------------------------------------------
# GeneVocab.multi_species()
# ------------------------------------------------------------------


class TestGeneVocabMultiSpecies:
    def test_multi_species_default(self):
        vocab = GeneVocab.multi_species()
        assert len(vocab) > NUM_SPECIAL_TOKENS
        assert "TP53" in vocab
        assert "BRCA1" in vocab

    def test_multi_species_with_base_genes(self):
        vocab = GeneVocab.multi_species(base_genes=["TP53", "BRCA1", "MYC"])
        assert len(vocab) == 3 + NUM_SPECIAL_TOKENS
        assert "TP53" in vocab
        assert "BRCA1" in vocab
        assert "MYC" in vocab

    def test_multi_species_contains_human_genes(self):
        vocab = GeneVocab.multi_species()
        # Should contain genes from the ortholog table
        assert "EGFR" in vocab
        assert "GAPDH" in vocab

    def test_multi_species_with_one2many(self):
        vocab_strict = GeneVocab.multi_species(include_one2many=False)
        vocab_loose = GeneVocab.multi_species(include_one2many=True)
        # one2many may add more canonical genes
        assert len(vocab_loose) >= len(vocab_strict)


# ------------------------------------------------------------------
# AnnDataStore with ortholog_mapper
# ------------------------------------------------------------------


class TestAnnDataStoreOrthologs:
    def _make_mouse_adata(self) -> ad.AnnData:
        """Create a small AnnData with mouse gene names."""
        rng = np.random.default_rng(42)
        n_obs, n_vars = 20, 10
        X = sp.random(n_obs, n_vars, density=0.5, format="csr", random_state=42)
        # Use real mouse gene names from the ortholog table
        mouse_genes = ["Trp53", "Brca1", "Egfr", "Gapdh", "Actb",
                        "Myc", "Kras", "Rb1", "Pten", "Cdkn2a"]
        obs = pd.DataFrame(
            {"cell_type": rng.choice(["A", "B"], n_obs)},
            index=[f"cell_{i}" for i in range(n_obs)],
        )
        var = pd.DataFrame(index=mouse_genes)
        return ad.AnnData(X=X, obs=obs, var=var)

    def test_ortholog_mapper_translates_genes(self):
        from scmodelforge.data.anndata_store import AnnDataStore

        mouse_adata = self._make_mouse_adata()
        mapper = OrthologMapper()

        # Build vocab with human gene names
        human_genes = ["TP53", "BRCA1", "EGFR", "GAPDH", "ACTB",
                        "MYC", "KRAS", "RB1", "PTEN", "CDKN2A"]
        vocab = GeneVocab.from_genes(human_genes)

        store = AnnDataStore(
            [mouse_adata],
            gene_vocab=vocab,
            ortholog_mapper=mapper,
            source_organism="mouse",
        )
        assert len(store) == 20

        # Should be able to get cells with mapped gene indices
        expr, gene_idx, meta = store.get_cell(0)
        assert len(expr) > 0
        # All gene indices should be in the human vocab range
        assert all(idx >= NUM_SPECIAL_TOKENS for idx in gene_idx)
        assert all(idx < len(vocab) for idx in gene_idx)

    def test_without_ortholog_mapper_no_overlap(self):
        """Mouse genes don't match human vocab without mapping."""
        from scmodelforge.data.anndata_store import AnnDataStore

        mouse_adata = self._make_mouse_adata()
        human_genes = ["TP53", "BRCA1", "EGFR", "GAPDH", "ACTB",
                        "MYC", "KRAS", "RB1", "PTEN", "CDKN2A"]
        vocab = GeneVocab.from_genes(human_genes)

        store = AnnDataStore([mouse_adata], gene_vocab=vocab)
        # Without mapping, mouse names (Trp53) won't match human names (TP53)
        expr, gene_idx, meta = store.get_cell(0)
        assert len(expr) == 0
