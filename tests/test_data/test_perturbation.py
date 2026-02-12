"""Tests for perturbation-aware data handling."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.data.perturbation import (
    PerturbationDataset,
    PerturbationMetadata,
    collate_perturbation_cells,
    detect_perturbation_columns,
    parse_perturbation_metadata,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def crispr_adata() -> ad.AnnData:
    """AnnData with CRISPR perturbation metadata."""
    rng = np.random.default_rng(42)
    n_obs, n_vars = 30, 50
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=42)
    obs = pd.DataFrame(
        {
            "guide_identity": (
                ["control"] * 10
                + ["TP53"] * 10
                + ["BRCA1"] * 10
            ),
            "cell_type": rng.choice(["T cell", "B cell"], n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"GENE_{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture()
def chemical_adata() -> ad.AnnData:
    """AnnData with chemical perturbation + dose metadata."""
    n_obs, n_vars = 20, 50
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=42)
    obs = pd.DataFrame(
        {
            "treatment": (
                ["control"] * 5
                + ["Dexamethasone"] * 5
                + ["Rapamycin"] * 5
                + ["control"] * 5
            ),
            "dose": (
                [0.0] * 5
                + [1.0] * 5
                + [10.0] * 5
                + [0.0] * 5
            ),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"GENE_{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture()
def adata_no_pert() -> ad.AnnData:
    """AnnData with no perturbation columns."""
    n_obs, n_vars = 10, 20
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=42)
    obs = pd.DataFrame(
        {"cell_type": ["A"] * n_obs},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"GENE_{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


# ------------------------------------------------------------------
# PerturbationMetadata
# ------------------------------------------------------------------


class TestPerturbationMetadata:
    def test_default_construction(self):
        pm = PerturbationMetadata()
        assert pm.perturbation_type == "unknown"
        assert pm.perturbation_name == ""
        assert pm.dose is None
        assert pm.dose_unit is None
        assert pm.is_control is False

    def test_full_construction(self):
        pm = PerturbationMetadata(
            perturbation_type="crispr",
            perturbation_name="TP53",
            dose=1.0,
            dose_unit="uM",
            is_control=False,
        )
        assert pm.perturbation_type == "crispr"
        assert pm.perturbation_name == "TP53"
        assert pm.dose == 1.0
        assert pm.dose_unit == "uM"

    def test_control_flag(self):
        pm = PerturbationMetadata(
            perturbation_name="control",
            is_control=True,
        )
        assert pm.is_control is True


# ------------------------------------------------------------------
# detect_perturbation_columns
# ------------------------------------------------------------------


class TestDetectColumns:
    def test_detects_guide_identity(self, crispr_adata):
        result = detect_perturbation_columns(crispr_adata)
        assert result["perturbation"] == "guide_identity"
        assert "dose" not in result

    def test_detects_treatment_and_dose(self, chemical_adata):
        result = detect_perturbation_columns(chemical_adata)
        assert result["perturbation"] == "treatment"
        assert result["dose"] == "dose"

    def test_no_match(self, adata_no_pert):
        result = detect_perturbation_columns(adata_no_pert)
        assert result == {}

    def test_perturbation_column_name(self):
        """Column named exactly 'perturbation' is detected first."""
        obs = pd.DataFrame(
            {"perturbation": ["A", "B"], "treatment": ["X", "Y"]},
            index=["c0", "c1"],
        )
        adata = ad.AnnData(
            X=sp.csr_matrix(np.ones((2, 3))),
            obs=obs,
            var=pd.DataFrame(index=["G0", "G1", "G2"]),
        )
        result = detect_perturbation_columns(adata)
        assert result["perturbation"] == "perturbation"


# ------------------------------------------------------------------
# parse_perturbation_metadata
# ------------------------------------------------------------------


class TestParsePerturbationMetadata:
    def test_crispr_metadata(self, crispr_adata):
        metadata = parse_perturbation_metadata(crispr_adata)
        assert len(metadata) == 30
        # First 10 are controls
        for pm in metadata[:10]:
            assert pm.is_control is True
            assert pm.perturbation_name == "control"
        # Next 10 are TP53
        for pm in metadata[10:20]:
            assert pm.is_control is False
            assert pm.perturbation_name == "TP53"
        # Perturbation type inferred from column name
        assert metadata[0].perturbation_type == "crispr"

    def test_chemical_with_dose(self, chemical_adata):
        metadata = parse_perturbation_metadata(chemical_adata, dose_unit="uM")
        assert len(metadata) == 20
        # Controls have dose 0
        assert metadata[0].dose == 0.0
        assert metadata[0].is_control is True
        # Dexamethasone has dose 1.0
        assert metadata[5].dose == 1.0
        assert metadata[5].perturbation_name == "Dexamethasone"
        assert metadata[5].dose_unit == "uM"

    def test_auto_detect(self, crispr_adata):
        """Auto-detection finds the right column."""
        metadata = parse_perturbation_metadata(crispr_adata)
        assert metadata[10].perturbation_name == "TP53"

    def test_explicit_key(self, chemical_adata):
        metadata = parse_perturbation_metadata(
            chemical_adata, perturbation_key="treatment"
        )
        assert metadata[5].perturbation_name == "Dexamethasone"

    def test_no_perturbation_columns(self, adata_no_pert):
        metadata = parse_perturbation_metadata(adata_no_pert)
        assert len(metadata) == 10
        # All should be default (unknown) metadata
        for pm in metadata:
            assert pm.perturbation_type == "unknown"
            assert pm.perturbation_name == ""

    def test_control_label_case_insensitive(self, crispr_adata):
        metadata = parse_perturbation_metadata(
            crispr_adata, control_label="CONTROL"
        )
        assert metadata[0].is_control is True


# ------------------------------------------------------------------
# PerturbationDataset
# ------------------------------------------------------------------


class TestPerturbationDataset:
    def test_getitem_has_perturbation_key(self, crispr_adata):
        vocab = GeneVocab.from_adata(crispr_adata)
        ds = PerturbationDataset(crispr_adata, gene_vocab=vocab)
        item = ds[0]
        assert "perturbation" in item
        assert isinstance(item["perturbation"], PerturbationMetadata)

    def test_getitem_standard_keys(self, crispr_adata):
        vocab = GeneVocab.from_adata(crispr_adata)
        ds = PerturbationDataset(crispr_adata, gene_vocab=vocab)
        item = ds[0]
        assert "expression" in item
        assert "gene_indices" in item
        assert "n_genes" in item
        assert "metadata" in item

    def test_control_cells(self, crispr_adata):
        vocab = GeneVocab.from_adata(crispr_adata)
        ds = PerturbationDataset(crispr_adata, gene_vocab=vocab)
        item = ds[0]
        assert item["perturbation"].is_control is True

    def test_perturbed_cells(self, crispr_adata):
        vocab = GeneVocab.from_adata(crispr_adata)
        ds = PerturbationDataset(crispr_adata, gene_vocab=vocab)
        item = ds[15]
        assert item["perturbation"].is_control is False
        assert item["perturbation"].perturbation_name == "TP53"

    def test_length(self, crispr_adata):
        vocab = GeneVocab.from_adata(crispr_adata)
        ds = PerturbationDataset(crispr_adata, gene_vocab=vocab)
        assert len(ds) == 30

    def test_repr(self, crispr_adata):
        vocab = GeneVocab.from_adata(crispr_adata)
        ds = PerturbationDataset(crispr_adata, gene_vocab=vocab)
        r = repr(ds)
        assert "PerturbationDataset" in r
        assert "n_cells=30" in r


# ------------------------------------------------------------------
# Collation
# ------------------------------------------------------------------


class TestCollation:
    def _make_batch(self, crispr_adata):
        vocab = GeneVocab.from_adata(crispr_adata)
        ds = PerturbationDataset(crispr_adata, gene_vocab=vocab)
        return [ds[i] for i in range(4)]

    def test_collation_shape(self, crispr_adata):
        batch = self._make_batch(crispr_adata)
        collated = collate_perturbation_cells(batch)
        assert collated["expression"].shape[0] == 4
        assert collated["gene_indices"].shape[0] == 4
        assert collated["attention_mask"].shape[0] == 4

    def test_perturbation_names(self, crispr_adata):
        batch = self._make_batch(crispr_adata)
        collated = collate_perturbation_cells(batch)
        assert isinstance(collated["perturbation_names"], list)
        assert len(collated["perturbation_names"]) == 4

    def test_is_control_tensor(self, crispr_adata):
        batch = self._make_batch(crispr_adata)
        collated = collate_perturbation_cells(batch)
        assert collated["is_control"].dtype == torch.bool
        assert collated["is_control"].shape == (4,)
        # First 4 cells are controls (indices 0-9 are control)
        assert collated["is_control"].all()

    def test_doses_tensor(self, chemical_adata):
        vocab = GeneVocab.from_adata(chemical_adata)
        ds = PerturbationDataset(chemical_adata, gene_vocab=vocab)
        batch = [ds[i] for i in range(4)]
        collated = collate_perturbation_cells(batch)
        assert collated["doses"].dtype == torch.float32
        assert collated["doses"].shape == (4,)

    def test_missing_dose_is_nan(self, crispr_adata):
        """CRISPR data has no dose column, so doses should be NaN."""
        batch = self._make_batch(crispr_adata)
        collated = collate_perturbation_cells(batch)
        assert torch.isnan(collated["doses"]).all()
