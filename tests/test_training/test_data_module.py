"""Tests for TokenizedCellDataset and CellDataModule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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
