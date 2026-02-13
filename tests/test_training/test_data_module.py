"""Tests for TokenizedCellDataset and CellDataModule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scmodelforge.config.schema import GeneSelectionConfig, SamplingConfig
from scmodelforge.training.data_module import CellDataModule, TokenizedCellDataset

if TYPE_CHECKING:
    from anndata import AnnData

    from scmodelforge.config.schema import DataConfig, TokenizerConfig


# ------------------------------------------------------------------
# TokenizedCellDataset
# ------------------------------------------------------------------


class TestTokenizedCellDataset:
    """Tests for TokenizedCellDataset."""

    def test_len_matches_underlying(self, tiny_adata: AnnData, tiny_vocab, tiny_tokenizer_config) -> None:
        from scmodelforge.data.dataset import CellDataset
        from scmodelforge.tokenizers.registry import get_tokenizer

        ds = CellDataset(tiny_adata, tiny_vocab)
        tok = get_tokenizer(
            tiny_tokenizer_config.strategy,
            gene_vocab=tiny_vocab,
            max_len=tiny_tokenizer_config.max_genes,
            prepend_cls=tiny_tokenizer_config.prepend_cls,
        )
        wrapped = TokenizedCellDataset(ds, tok)
        assert len(wrapped) == len(ds)

    def test_getitem_returns_tokenized_cell(self, tiny_adata: AnnData, tiny_vocab, tiny_tokenizer_config) -> None:
        from scmodelforge.data.dataset import CellDataset
        from scmodelforge.tokenizers.base import TokenizedCell
        from scmodelforge.tokenizers.registry import get_tokenizer

        ds = CellDataset(tiny_adata, tiny_vocab)
        tok = get_tokenizer(
            tiny_tokenizer_config.strategy,
            gene_vocab=tiny_vocab,
            max_len=tiny_tokenizer_config.max_genes,
            prepend_cls=tiny_tokenizer_config.prepend_cls,
        )
        wrapped = TokenizedCellDataset(ds, tok)
        cell = wrapped[0]
        assert isinstance(cell, TokenizedCell)
        assert cell.input_ids.ndim == 1
        assert cell.attention_mask.ndim == 1

    def test_getitem_with_masking(self, tiny_adata: AnnData, tiny_vocab, tiny_tokenizer_config) -> None:
        from scmodelforge.data.dataset import CellDataset
        from scmodelforge.tokenizers.base import MaskedTokenizedCell
        from scmodelforge.tokenizers.masking import MaskingStrategy
        from scmodelforge.tokenizers.registry import get_tokenizer

        ds = CellDataset(tiny_adata, tiny_vocab)
        tok = get_tokenizer(
            tiny_tokenizer_config.strategy,
            gene_vocab=tiny_vocab,
            max_len=tiny_tokenizer_config.max_genes,
            prepend_cls=tiny_tokenizer_config.prepend_cls,
        )
        masking = MaskingStrategy(mask_ratio=0.15, vocab_size=len(tiny_vocab))
        wrapped = TokenizedCellDataset(ds, tok, masking=masking)
        cell = wrapped[0]
        assert isinstance(cell, MaskedTokenizedCell)
        assert cell.labels.ndim == 1
        assert cell.masked_positions.ndim == 1


# ------------------------------------------------------------------
# CellDataModule
# ------------------------------------------------------------------


class TestCellDataModule:
    """Tests for CellDataModule."""

    def test_setup_creates_datasets(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            seed=42,
            adata=tiny_adata,
        )
        dm.setup()
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert dm._is_setup is True

    def test_setup_is_idempotent(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            adata=tiny_adata,
        )
        dm.setup()
        train_ds = dm._train_dataset
        dm.setup()  # second call
        assert dm._train_dataset is train_ds  # same object

    def test_gene_vocab_available_after_setup(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            adata=tiny_adata,
        )
        dm.setup()
        assert dm.gene_vocab is not None
        assert len(dm.gene_vocab) > 0

    def test_gene_vocab_before_setup_raises(
        self, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
        )
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.gene_vocab

    def test_tokenizer_before_setup_raises(
        self, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
        )
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.tokenizer

    def test_train_dataloader(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
        )
        dm.setup()
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].ndim == 2

    def test_val_dataloader(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
        )
        dm.setup()
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "labels" in batch

    def test_train_val_split_sizes(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            seed=42,
            adata=tiny_adata,
        )
        dm.setup()
        n_train = len(dm._train_dataset)  # type: ignore[arg-type]
        n_val = len(dm._val_dataset)  # type: ignore[arg-type]
        assert n_train + n_val == 20
        assert n_val >= 1

    def test_dataloader_before_setup_raises(
        self, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
        )
        with pytest.raises(RuntimeError, match="setup"):
            dm.train_dataloader()
        with pytest.raises(RuntimeError, match="setup"):
            dm.val_dataloader()

    def test_masking_property(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            adata=tiny_adata,
        )
        dm.setup()
        assert dm.masking is not None

    def test_weighted_sampling_creates_sampler(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        samp_cfg = SamplingConfig(strategy="weighted", label_key="cell_type")
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
            sampling_config=samp_cfg,
        )
        dm.setup()
        assert dm._sampler is not None
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch

    def test_gene_selection_most_expressed_uses_collator(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        from scmodelforge.data.gene_selection import GeneSelectionCollator

        gs_cfg = GeneSelectionConfig(strategy="most_expressed", n_genes=5)
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
            gene_selection_config=gs_cfg,
        )
        dm.setup()
        assert dm._use_gene_selection is True
        assert isinstance(dm._train_collate_fn, GeneSelectionCollator)
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch
        # Sequence length should be limited
        assert batch["input_ids"].shape[1] <= 5 + 1  # n_genes + CLS

    def test_weighted_sampling_passes_obs_keys_to_load_adata(
        self, tiny_adata: AnnData, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        """Weighted sampling label_key must be forwarded to load_adata as obs_keys."""
        from unittest.mock import patch

        from scmodelforge.config.schema import DataConfig as _DataConfig, PreprocessingConfig

        census_data_config = _DataConfig(
            source="cellxgene_census",
            preprocessing=PreprocessingConfig(normalize="library_size", target_sum=1e4, log1p=True),
        )
        samp_cfg = SamplingConfig(strategy="weighted", label_key="cell_type")
        dm = CellDataModule(
            data_config=census_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            sampling_config=samp_cfg,
        )
        # Mock load_adata at the source module (locally imported in setup())
        with patch("scmodelforge.data._utils.load_adata", return_value=tiny_adata) as mock_load:
            dm.setup()
            mock_load.assert_called_once()
            _, kwargs = mock_load.call_args
            assert kwargs.get("obs_keys") == ["cell_type"]

    def test_defaults_unchanged_behaviour(
        self, tiny_adata: AnnData, tiny_data_config: DataConfig, tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        """Default configs: no sampler, no gene selection — same as before."""
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
        )
        dm.setup()
        assert dm._sampler is None
        assert dm._use_gene_selection is False
        assert isinstance(dm._train_dataset, TokenizedCellDataset)


# ------------------------------------------------------------------
# Streaming mode
# ------------------------------------------------------------------


class TestStreamingSetup:
    """Tests that streaming mode avoids full data materialization."""

    @pytest.fixture()
    def h5ad_file(self, tmp_path, tiny_adata: AnnData) -> str:
        """Write tiny_adata to an H5AD file."""
        path = tmp_path / "test.h5ad"
        tiny_adata.write_h5ad(path)
        return str(path)

    @pytest.fixture()
    def two_h5ad_files(self, tmp_path, tiny_adata: AnnData) -> list[str]:
        """Write tiny_adata to two separate H5AD files."""
        paths = []
        for i in range(2):
            path = tmp_path / f"test_{i}.h5ad"
            tiny_adata.write_h5ad(path)
            paths.append(str(path))
        return paths

    @pytest.fixture()
    def streaming_data_config(self, h5ad_file) -> DataConfig:
        from scmodelforge.config.schema import DataConfig as _DataConfig, PreprocessingConfig

        return _DataConfig(
            paths=[h5ad_file],
            streaming=True,
            streaming_chunk_size=10,
            streaming_shuffle_buffer=0,
            preprocessing=PreprocessingConfig(
                normalize="library_size",
                target_sum=1e4,
                log1p=True,
            ),
        )

    def test_streaming_setup_from_files(
        self,
        streaming_data_config: DataConfig,
        tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        """Streaming setup from file paths should produce valid datasets."""
        dm = CellDataModule(
            data_config=streaming_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
        )
        dm.setup()
        assert dm._streaming is True
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert dm._gene_vocab is not None
        assert dm._loaded_adata is not None

    def test_streaming_val_dataloader_yields_batches(
        self,
        streaming_data_config: DataConfig,
        tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        """Val dataloader in streaming mode should yield valid batches."""
        dm = CellDataModule(
            data_config=streaming_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
        )
        dm.setup()
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "labels" in batch

    def test_streaming_train_dataloader_yields_batches(
        self,
        streaming_data_config: DataConfig,
        tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        """Train dataloader in streaming mode should yield valid batches."""
        dm = CellDataModule(
            data_config=streaming_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
        )
        dm.setup()
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "attention_mask" in batch

    def test_streaming_with_injected_adata(
        self,
        tiny_adata: AnnData,
        streaming_data_config: DataConfig,
        tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        """Streaming mode with injected adata should still work (testing shortcut)."""
        dm = CellDataModule(
            data_config=streaming_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
        )
        dm.setup()
        assert dm._streaming is True
        assert len(dm.gene_vocab) > 0
        assert dm.adata is not None

    def test_scan_var_names_backed(self, h5ad_file, tiny_adata: AnnData) -> None:
        """_scan_var_names_backed should return gene names without loading X."""
        from scmodelforge.config.schema import CloudStorageConfig

        cloud_cfg = CloudStorageConfig()
        names = CellDataModule._scan_var_names_backed([h5ad_file], cloud_cfg)
        expected = list(tiny_adata.var_names)
        assert names == expected

    def test_scan_var_names_backed_multi_file(self, two_h5ad_files, tiny_adata: AnnData) -> None:
        """_scan_var_names_backed with identical files returns deduplicated names."""
        from scmodelforge.config.schema import CloudStorageConfig

        cloud_cfg = CloudStorageConfig()
        names = CellDataModule._scan_var_names_backed(two_h5ad_files, cloud_cfg)
        expected = list(tiny_adata.var_names)
        assert names == expected  # deduped — same genes in both files

    def test_read_val_subset_backed(self, h5ad_file, tiny_adata: AnnData) -> None:
        """_read_val_subset_backed should return in-memory AnnData."""
        from scmodelforge.config.schema import CloudStorageConfig

        cloud_cfg = CloudStorageConfig()
        val_adata = CellDataModule._read_val_subset_backed([h5ad_file], cloud_cfg)
        # Should be in-memory (not backed)
        assert val_adata.n_obs <= tiny_adata.n_obs
        assert val_adata.n_obs > 0
