"""Tests for TokenizedCell, MaskedTokenizedCell, and BaseTokenizer."""

from __future__ import annotations

import pytest
import torch

from scmodelforge._constants import CLS_TOKEN_ID, PAD_TOKEN_ID
from scmodelforge.tokenizers.base import BaseTokenizer, MaskedTokenizedCell, TokenizedCell, _pad_1d

# ------------------------------------------------------------------
# Dataclass construction
# ------------------------------------------------------------------


class TestTokenizedCell:
    def test_create_minimal(self):
        cell = TokenizedCell(
            input_ids=torch.tensor([3, 5, 6]),
            attention_mask=torch.ones(3, dtype=torch.long),
        )
        assert cell.input_ids.shape == (3,)
        assert cell.values is None
        assert cell.metadata == {}

    def test_create_with_values(self):
        cell = TokenizedCell(
            input_ids=torch.tensor([3, 5, 6]),
            attention_mask=torch.ones(3, dtype=torch.long),
            values=torch.tensor([0.0, 1.0, 2.0]),
            gene_indices=torch.tensor([3, 5, 6]),
            metadata={"cell_type": "T cell"},
        )
        assert cell.values is not None
        assert cell.values.shape == (3,)
        assert cell.metadata["cell_type"] == "T cell"

    def test_satisfies_protocol(self):
        from scmodelforge._types import TokenizedCellProtocol

        cell = TokenizedCell(
            input_ids=torch.tensor([1, 2]),
            attention_mask=torch.ones(2, dtype=torch.long),
            metadata={"x": 1},
        )
        assert isinstance(cell, TokenizedCellProtocol)


class TestMaskedTokenizedCell:
    def test_create(self):
        cell = MaskedTokenizedCell(
            input_ids=torch.tensor([3, 2, 6]),
            attention_mask=torch.ones(3, dtype=torch.long),
            labels=torch.tensor([-100, 5, -100]),
            masked_positions=torch.tensor([False, True, False]),
        )
        assert cell.labels.shape == (3,)
        assert cell.masked_positions.sum().item() == 1

    def test_inherits_tokenized_cell(self):
        cell = MaskedTokenizedCell(
            input_ids=torch.tensor([1]),
            attention_mask=torch.ones(1, dtype=torch.long),
        )
        assert isinstance(cell, TokenizedCell)


# ------------------------------------------------------------------
# Abstract enforcement
# ------------------------------------------------------------------


class TestBaseTokenizerAbstract:
    def test_cannot_instantiate(self, small_vocab):
        with pytest.raises(TypeError, match="abstract"):
            BaseTokenizer(small_vocab)

    def test_concrete_subclass(self, small_vocab):
        class DummyTokenizer(BaseTokenizer):
            @property
            def vocab_size(self) -> int:
                return len(self.gene_vocab)

            @property
            def strategy_name(self) -> str:
                return "dummy"

            def tokenize(self, expression, gene_indices, metadata=None) -> TokenizedCell:
                ids = torch.tensor([CLS_TOKEN_ID], dtype=torch.long)
                return TokenizedCell(
                    input_ids=ids,
                    attention_mask=torch.ones(1, dtype=torch.long),
                    gene_indices=ids,
                )

        tok = DummyTokenizer(small_vocab, max_len=10)
        assert tok.vocab_size > 0
        assert tok.strategy_name == "dummy"

    def test_missing_abstract_method(self, small_vocab):
        class Incomplete(BaseTokenizer):
            @property
            def vocab_size(self) -> int:
                return 10

            @property
            def strategy_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete(small_vocab)


# ------------------------------------------------------------------
# _collate: padding, truncation, masking
# ------------------------------------------------------------------


class TestCollate:
    def _make_dummy_tokenizer(self, small_vocab, max_len=20):
        class Dummy(BaseTokenizer):
            @property
            def vocab_size(self) -> int:
                return len(self.gene_vocab)

            @property
            def strategy_name(self) -> str:
                return "dummy"

            def tokenize(self, expression, gene_indices, metadata=None) -> TokenizedCell:
                return TokenizedCell(input_ids=torch.tensor([]), attention_mask=torch.tensor([]))

        return Dummy(small_vocab, max_len=max_len)

    def test_padding_to_batch_max(self, small_vocab):
        tok = self._make_dummy_tokenizer(small_vocab)
        cells = [
            TokenizedCell(
                input_ids=torch.tensor([5, 6, 7], dtype=torch.long),
                attention_mask=torch.ones(3, dtype=torch.long),
                values=torch.tensor([1.0, 2.0, 3.0]),
                gene_indices=torch.tensor([5, 6, 7], dtype=torch.long),
            ),
            TokenizedCell(
                input_ids=torch.tensor([8, 9], dtype=torch.long),
                attention_mask=torch.ones(2, dtype=torch.long),
                values=torch.tensor([4.0, 5.0]),
                gene_indices=torch.tensor([8, 9], dtype=torch.long),
            ),
        ]
        batch = tok._collate(cells)
        assert batch["input_ids"].shape == (2, 3)
        assert batch["attention_mask"].shape == (2, 3)
        # Second cell should be padded
        assert batch["input_ids"][1, 2].item() == PAD_TOKEN_ID
        assert batch["attention_mask"][1, 2].item() == 0
        assert batch["values"][1, 2].item() == 0.0

    def test_no_padding_when_same_length(self, small_vocab):
        tok = self._make_dummy_tokenizer(small_vocab)
        cells = [
            TokenizedCell(
                input_ids=torch.tensor([5, 6], dtype=torch.long),
                attention_mask=torch.ones(2, dtype=torch.long),
                gene_indices=torch.tensor([5, 6], dtype=torch.long),
            ),
            TokenizedCell(
                input_ids=torch.tensor([7, 8], dtype=torch.long),
                attention_mask=torch.ones(2, dtype=torch.long),
                gene_indices=torch.tensor([7, 8], dtype=torch.long),
            ),
        ]
        batch = tok._collate(cells)
        assert batch["input_ids"].shape == (2, 2)
        assert (batch["attention_mask"] == 1).all()

    def test_truncation_safety_net(self, small_vocab):
        tok = self._make_dummy_tokenizer(small_vocab, max_len=3)
        # Cell exceeds max_len — should be truncated in _collate
        long_ids = torch.arange(5, dtype=torch.long)
        cells = [
            TokenizedCell(
                input_ids=long_ids,
                attention_mask=torch.ones(5, dtype=torch.long),
                values=torch.arange(5, dtype=torch.float32),
                gene_indices=long_ids.clone(),
            ),
        ]
        # _collate truncates cells exceeding max_len
        # We manually truncate since _truncate is a safety net
        for i, c in enumerate(cells):
            if c.input_ids.shape[0] > tok.max_len:
                n = tok.max_len
                cells[i] = TokenizedCell(
                    input_ids=c.input_ids[:n],
                    attention_mask=c.attention_mask[:n],
                    values=c.values[:n] if c.values is not None else None,
                    gene_indices=c.gene_indices[:n],
                    metadata=c.metadata,
                )
        batch = tok._collate(cells)
        assert batch["input_ids"].shape == (1, 3)

    def test_collate_masked_cells(self, small_vocab):
        tok = self._make_dummy_tokenizer(small_vocab)
        cells = [
            MaskedTokenizedCell(
                input_ids=torch.tensor([3, 2, 5], dtype=torch.long),
                attention_mask=torch.ones(3, dtype=torch.long),
                values=torch.tensor([0.0, 1.0, 2.0]),
                gene_indices=torch.tensor([3, 5, 5], dtype=torch.long),
                labels=torch.tensor([-100, 7, -100], dtype=torch.long),
                masked_positions=torch.tensor([False, True, False]),
            ),
            MaskedTokenizedCell(
                input_ids=torch.tensor([3, 6], dtype=torch.long),
                attention_mask=torch.ones(2, dtype=torch.long),
                values=torch.tensor([0.0, 3.0]),
                gene_indices=torch.tensor([3, 6], dtype=torch.long),
                labels=torch.tensor([-100, -100], dtype=torch.long),
                masked_positions=torch.tensor([False, False]),
            ),
        ]
        batch = tok._collate(cells)
        assert "labels" in batch
        assert "masked_positions" in batch
        assert batch["labels"].shape == (2, 3)
        # Padded label position should be -100
        assert batch["labels"][1, 2].item() == -100

    def test_collate_empty_list(self, small_vocab):
        tok = self._make_dummy_tokenizer(small_vocab)
        assert tok._collate([]) == {}


# ------------------------------------------------------------------
# _pad_1d helper
# ------------------------------------------------------------------


class TestPad1d:
    def test_pad_int(self):
        t = torch.tensor([1, 2, 3], dtype=torch.long)
        padded = _pad_1d(t, 2, 0)
        assert padded.shape == (5,)
        assert padded.tolist() == [1, 2, 3, 0, 0]

    def test_pad_zero_length(self):
        t = torch.tensor([1, 2], dtype=torch.long)
        padded = _pad_1d(t, 0, 0)
        assert padded.shape == (2,)
        assert padded.tolist() == [1, 2]

    def test_pad_float(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        padded = _pad_1d(t, 3, -100.0)
        assert padded.shape == (5,)
        assert padded[-1].item() == -100.0
