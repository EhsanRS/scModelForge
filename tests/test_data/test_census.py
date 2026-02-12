"""Tests for CELLxGENE Census integration.

All Census API calls are mocked — no actual Census access is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import anndata as ad
import pandas as pd
import pytest
import scipy.sparse as sp

from scmodelforge.config.schema import CensusConfig, DataConfig
from scmodelforge.data._utils import load_adata
from scmodelforge.data.census import build_obs_value_filter, load_census_adata

# ------------------------------------------------------------------
# TestBuildObsValueFilter
# ------------------------------------------------------------------


class TestBuildObsValueFilter:
    def test_single_string_value(self):
        result = build_obs_value_filter({"tissue": "brain"})
        assert result == "tissue == 'brain'"

    def test_list_values(self):
        result = build_obs_value_filter({"tissue": ["brain", "lung"]})
        assert result == "tissue in ['brain', 'lung']"

    def test_bool_value(self):
        result = build_obs_value_filter({"is_primary_data": True})
        assert result == "is_primary_data == True"

    def test_bool_false_value(self):
        result = build_obs_value_filter({"is_primary_data": False})
        assert result == "is_primary_data == False"

    def test_numeric_int_value(self):
        result = build_obs_value_filter({"n_genes": 500})
        assert result == "n_genes == 500"

    def test_numeric_float_value(self):
        result = build_obs_value_filter({"score": 0.95})
        assert result == "score == 0.95"

    def test_multiple_filters_combined(self):
        result = build_obs_value_filter({
            "tissue": "brain",
            "is_primary_data": True,
        })
        assert "tissue == 'brain'" in result
        assert "is_primary_data == True" in result
        assert " and " in result

    def test_empty_dict_returns_none(self):
        result = build_obs_value_filter({})
        assert result is None

    def test_list_with_mixed_types(self):
        result = build_obs_value_filter({"col": [1, 2, 3]})
        assert result == "col in [1, 2, 3]"


# ------------------------------------------------------------------
# TestLoadCensusAdata
# ------------------------------------------------------------------


def _make_mock_adata(n_obs: int = 50, n_vars: int = 20) -> ad.AnnData:
    """Create a small AnnData for mocking Census responses."""
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=0)
    obs = pd.DataFrame(
        {"cell_type": ["T cell"] * n_obs, "tissue": ["brain"] * n_obs},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"GENE_{i}" for i in range(n_vars)]},
        index=[f"GENE_{i}" for i in range(n_vars)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_mock_census_module(mock_adata: ad.AnnData) -> MagicMock:
    """Build a mock ``cellxgene_census`` module for sys.modules injection."""
    mock_mod = MagicMock()
    mock_soma = MagicMock()
    mock_mod.open_soma.return_value.__enter__ = MagicMock(return_value=mock_soma)
    mock_mod.open_soma.return_value.__exit__ = MagicMock(return_value=False)
    mock_mod.get_anndata.return_value = mock_adata
    return mock_mod


class TestLoadCensusAdata:
    def test_calls_get_anndata_with_correct_args(self):
        mock_adata = _make_mock_adata()
        mock_mod = _make_mock_census_module(mock_adata)
        mock_soma = mock_mod.open_soma.return_value.__enter__.return_value

        with patch.dict("sys.modules", {"cellxgene_census": mock_mod}):
            cfg = CensusConfig(
                organism="Mus musculus",
                census_version="2024-07-01",
                obs_value_filter="tissue == 'brain'",
            )
            result = load_census_adata(cfg)

        mock_mod.open_soma.assert_called_once_with(census_version="2024-07-01")
        mock_mod.get_anndata.assert_called_once_with(
            mock_soma,
            organism="Mus musculus",
            obs_value_filter="tissue == 'brain'",
            var_value_filter=None,
            column_names=None,
        )
        assert result.n_obs == 50

    def test_structured_filters_converted(self):
        mock_adata = _make_mock_adata()
        mock_mod = _make_mock_census_module(mock_adata)

        with patch.dict("sys.modules", {"cellxgene_census": mock_mod}):
            cfg = CensusConfig(filters={"tissue": ["brain", "lung"]})
            load_census_adata(cfg)

        call_kwargs = mock_mod.get_anndata.call_args[1]
        assert call_kwargs["obs_value_filter"] == "tissue in ['brain', 'lung']"

    def test_raw_filter_takes_precedence(self):
        mock_adata = _make_mock_adata()
        mock_mod = _make_mock_census_module(mock_adata)

        with patch.dict("sys.modules", {"cellxgene_census": mock_mod}):
            cfg = CensusConfig(
                obs_value_filter="tissue == 'custom'",
                filters={"tissue": "brain"},
            )
            load_census_adata(cfg)

        call_kwargs = mock_mod.get_anndata.call_args[1]
        assert call_kwargs["obs_value_filter"] == "tissue == 'custom'"

    def test_obs_columns_merged_with_obs_keys(self):
        mock_adata = _make_mock_adata()
        mock_mod = _make_mock_census_module(mock_adata)

        with patch.dict("sys.modules", {"cellxgene_census": mock_mod}):
            cfg = CensusConfig(obs_columns=["cell_type", "tissue"])
            load_census_adata(cfg, obs_keys=["donor_id"])

        call_kwargs = mock_mod.get_anndata.call_args[1]
        col_names = call_kwargs["column_names"]["obs"]
        assert set(col_names) == {"cell_type", "tissue", "donor_id"}

    def test_import_error_message(self):
        with patch.dict("sys.modules", {"cellxgene_census": None}), pytest.raises(
            ImportError, match="cellxgene-census is required"
        ):
            load_census_adata(CensusConfig())

    def test_var_value_filter_forwarded(self):
        mock_adata = _make_mock_adata()
        mock_mod = _make_mock_census_module(mock_adata)

        with patch.dict("sys.modules", {"cellxgene_census": mock_mod}):
            cfg = CensusConfig(var_value_filter="feature_name in ['TP53', 'BRCA1']")
            load_census_adata(cfg)

        call_kwargs = mock_mod.get_anndata.call_args[1]
        assert call_kwargs["var_value_filter"] == "feature_name in ['TP53', 'BRCA1']"


# ------------------------------------------------------------------
# TestLoadAdata
# ------------------------------------------------------------------


class TestLoadAdata:
    def test_adata_passthrough(self, mini_adata):
        cfg = DataConfig(source="local")
        result = load_adata(cfg, adata=mini_adata)
        assert result is mini_adata

    def test_local_source_loads_from_paths(self, tmp_h5ad):
        cfg = DataConfig(source="local", paths=[tmp_h5ad])
        result = load_adata(cfg)
        assert result.n_obs == 100

    def test_local_source_concats_multiple(self, mini_adata, tmp_path):
        p1 = tmp_path / "a.h5ad"
        p2 = tmp_path / "b.h5ad"
        mini_adata.write_h5ad(p1)
        mini_adata.write_h5ad(p2)
        cfg = DataConfig(source="local", paths=[str(p1), str(p2)])
        result = load_adata(cfg)
        assert result.n_obs == 200

    @patch("scmodelforge.data.census.load_census_adata")
    def test_census_source_dispatches(self, mock_load):
        mock_load.return_value = _make_mock_adata()
        cfg = DataConfig(source="cellxgene_census")
        result = load_adata(cfg, obs_keys=["cell_type"])
        mock_load.assert_called_once_with(cfg.census, obs_keys=["cell_type"])
        assert result.n_obs == 50

    def test_unknown_source_raises(self):
        cfg = DataConfig(source="unknown_db")
        with pytest.raises(ValueError, match="Unknown data source"):
            load_adata(cfg)


# ------------------------------------------------------------------
# TestCensusConfig
# ------------------------------------------------------------------


class TestCensusConfig:
    def test_census_config_defaults(self):
        cfg = CensusConfig()
        assert cfg.organism == "Homo sapiens"
        assert cfg.census_version == "latest"
        assert cfg.obs_value_filter is None
        assert cfg.var_value_filter is None
        assert cfg.filters is None
        assert cfg.obs_columns is None

    def test_data_config_has_census(self):
        cfg = DataConfig()
        assert isinstance(cfg.census, CensusConfig)

    def test_census_config_in_yaml(self):
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(DataConfig)
        override = OmegaConf.create({
            "source": "cellxgene_census",
            "census": {
                "organism": "Mus musculus",
                "census_version": "2024-07-01",
            },
        })
        merged = OmegaConf.merge(schema, override)
        cfg = OmegaConf.to_object(merged)
        assert cfg.source == "cellxgene_census"
        assert cfg.census.organism == "Mus musculus"
        assert cfg.census.census_version == "2024-07-01"


# ------------------------------------------------------------------
# TestCensusIntegration
# ------------------------------------------------------------------


class TestCensusIntegration:
    @patch("scmodelforge.data._utils.load_adata")
    def test_cell_data_module_with_census_source(self, mock_load, mini_adata):
        """Mocked Census → CellDataModule.setup() succeeds."""
        mock_load.return_value = mini_adata

        from scmodelforge.config.schema import DataConfig, TokenizerConfig
        from scmodelforge.training.data_module import CellDataModule

        data_cfg = DataConfig(source="cellxgene_census")
        tok_cfg = TokenizerConfig()
        dm = CellDataModule(data_cfg, tok_cfg, training_batch_size=8)
        dm.setup()

        assert dm._is_setup
        assert dm._train_dataset is not None

    @patch("scmodelforge.data._utils.load_adata")
    def test_finetune_data_module_with_census_source(self, mock_load, mini_adata):
        """Mocked Census → FineTuneDataModule.setup() succeeds."""
        mock_load.return_value = mini_adata

        from scmodelforge.config.schema import DataConfig, FinetuneConfig, TokenizerConfig
        from scmodelforge.finetuning.data_module import FineTuneDataModule

        data_cfg = DataConfig(source="cellxgene_census")
        tok_cfg = TokenizerConfig()
        ft_cfg = FinetuneConfig(label_key="cell_type")
        dm = FineTuneDataModule(data_cfg, tok_cfg, ft_cfg, training_batch_size=8)
        dm.setup()

        assert dm._is_setup
        assert dm._train_dataset is not None
