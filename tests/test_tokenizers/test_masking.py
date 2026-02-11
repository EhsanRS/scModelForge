"""Tests for MaskingStrategy."""

from __future__ import annotations

import pytest
import torch

from scmodelforge._constants import CLS_TOKEN_ID, MASK_TOKEN_ID, PAD_TOKEN_ID
from scmodelforge.tokenizers.base import MaskedTokenizedCell, TokenizedCell
from scmodelforge.tokenizers.masking import MaskingStrategy

# ------------------------------------------------------------------
# Basic masking
# ------------------------------------------------------------------


class TestBasicMasking:
    def test_returns_masked_cell(self, tokenized_cell_for_masking):
        strategy = MaskingStrategy(mask_ratio=0.15, vocab_size=20)
        result = strategy.apply(tokenized_cell_for_masking, seed=42)
        assert isinstance(result, MaskedTokenizedCell)

    def test_labels_at_masked_positions(self, tokenized_cell_for_masking):
        strategy = MaskingStrategy(mask_ratio=0.5, vocab_size=20)
        result = strategy.apply(tokenized_cell_for_masking, seed=42)
        # Labels should have original IDs at masked positions, -100 elsewhere
        masked_idx = result.masked_positions.nonzero(as_tuple=True)[0]
        for i in masked_idx:
            assert result.labels[i].item() != -100
        unmasked_idx = (~result.masked_positions).nonzero(as_tuple=True)[0]
        for i in unmasked_idx:
            assert result.labels[i].item() == -100

    def test_at_least_one_masked(self, tokenized_cell_for_masking):
        """Even with low mask_ratio, at least 1 token is masked."""
        strategy = MaskingStrategy(mask_ratio=0.01, vocab_size=20)
        result = strategy.apply(tokenized_cell_for_masking, seed=42)
        assert result.masked_positions.sum().item() >= 1

    def test_mask_count_approximate(self, tokenized_cell_for_masking):
        """Number of masked tokens should be close to mask_ratio * n_maskable."""
        strategy = MaskingStrategy(mask_ratio=0.5, vocab_size=20)
        result = strategy.apply(tokenized_cell_for_masking, seed=42)
        # Cell has CLS + 7 genes = 8 tokens. CLS not maskable → 7 maskable
        n_maskable = 7
        expected = max(1, round(n_maskable * 0.5))
        actual = result.masked_positions.sum().item()
        assert actual == expected


# ------------------------------------------------------------------
# CLS / PAD protection
# ------------------------------------------------------------------


class TestProtection:
    def test_cls_never_masked(self, tokenized_cell_for_masking):
        strategy = MaskingStrategy(mask_ratio=0.99, vocab_size=20)
        # Run multiple times to be sure
        for seed in range(10):
            result = strategy.apply(tokenized_cell_for_masking, seed=seed)
            cls_positions = (tokenized_cell_for_masking.input_ids == CLS_TOKEN_ID)
            assert not result.masked_positions[cls_positions].any()

    def test_pad_never_masked(self, small_vocab):
        """Pad tokens (from collation) should never be masked."""
        from scmodelforge.tokenizers import RankValueTokenizer

        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        # Tokenize and collate a batch — shorter cell gets padding
        import numpy as np

        expr1 = np.array([1.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        expr2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0, 0, 0, 0, 0], dtype=np.float32)
        genes = np.arange(4, 14, dtype=np.int64)

        cell1 = tok.tokenize(expr1, genes)
        cell2 = tok.tokenize(expr2, genes)

        # Manually pad cell1 to match cell2 length
        pad_len = cell2.input_ids.shape[0] - cell1.input_ids.shape[0]
        padded_cell = TokenizedCell(
            input_ids=torch.cat([cell1.input_ids, torch.full((pad_len,), PAD_TOKEN_ID, dtype=torch.long)]),
            attention_mask=torch.cat([cell1.attention_mask, torch.zeros(pad_len, dtype=torch.long)]),
            values=torch.cat([cell1.values, torch.zeros(pad_len)]) if cell1.values is not None else None,
            gene_indices=torch.cat(
                [cell1.gene_indices, torch.full((pad_len,), PAD_TOKEN_ID, dtype=torch.long)]
            ),
        )
        strategy = MaskingStrategy(mask_ratio=0.99, vocab_size=20)
        result = strategy.apply(padded_cell, seed=42)
        pad_positions = padded_cell.input_ids == PAD_TOKEN_ID
        assert not result.masked_positions[pad_positions].any()


# ------------------------------------------------------------------
# Action ratios (mask / random / keep)
# ------------------------------------------------------------------


class TestActionRatios:
    def test_mask_token_applied(self, tokenized_cell_for_masking):
        strategy = MaskingStrategy(
            mask_ratio=0.99,
            mask_action_ratio=1.0,
            random_replace_ratio=0.0,
            vocab_size=20,
        )
        result = strategy.apply(tokenized_cell_for_masking, seed=42)
        masked_idx = result.masked_positions.nonzero(as_tuple=True)[0]
        for i in masked_idx:
            assert result.input_ids[i].item() == MASK_TOKEN_ID

    def test_keep_action(self, tokenized_cell_for_masking):
        """With keep_ratio=1.0, masked positions keep original token."""
        strategy = MaskingStrategy(
            mask_ratio=0.99,
            mask_action_ratio=0.0,
            random_replace_ratio=0.0,
            vocab_size=20,
        )
        original_ids = tokenized_cell_for_masking.input_ids.clone()
        result = strategy.apply(tokenized_cell_for_masking, seed=42)
        masked_idx = result.masked_positions.nonzero(as_tuple=True)[0]
        for i in masked_idx:
            assert result.input_ids[i].item() == original_ids[i].item()

    def test_random_replace(self):
        """With random_replace_ratio=1.0, masked positions get random tokens."""
        cell = TokenizedCell(
            input_ids=torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=torch.long),
            attention_mask=torch.ones(10, dtype=torch.long),
            gene_indices=torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=torch.long),
        )
        strategy = MaskingStrategy(
            mask_ratio=0.99,
            mask_action_ratio=0.0,
            random_replace_ratio=1.0,
            vocab_size=100,
        )
        result = strategy.apply(cell, seed=42)
        # At least some tokens should differ from original
        masked_idx = result.masked_positions.nonzero(as_tuple=True)[0]
        original = cell.input_ids.clone()
        changed = sum(1 for i in masked_idx if result.input_ids[i] != original[i])
        # With random from 0-99, very unlikely all 10 stay the same
        assert changed > 0


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self, tokenized_cell_for_masking):
        strategy = MaskingStrategy(mask_ratio=0.5, vocab_size=20)
        r1 = strategy.apply(tokenized_cell_for_masking, seed=123)
        r2 = strategy.apply(tokenized_cell_for_masking, seed=123)
        assert torch.equal(r1.input_ids, r2.input_ids)
        assert torch.equal(r1.labels, r2.labels)
        assert torch.equal(r1.masked_positions, r2.masked_positions)

    def test_different_seed_different_result(self, tokenized_cell_for_masking):
        strategy = MaskingStrategy(mask_ratio=0.5, vocab_size=20)
        r1 = strategy.apply(tokenized_cell_for_masking, seed=1)
        r2 = strategy.apply(tokenized_cell_for_masking, seed=2)
        # Very unlikely to be identical with different seeds
        assert not torch.equal(r1.masked_positions, r2.masked_positions)


# ------------------------------------------------------------------
# No aliasing
# ------------------------------------------------------------------


class TestNoAliasing:
    def test_original_not_mutated(self, tokenized_cell_for_masking):
        original_ids = tokenized_cell_for_masking.input_ids.clone()
        strategy = MaskingStrategy(mask_ratio=0.99, vocab_size=20)
        strategy.apply(tokenized_cell_for_masking, seed=42)
        assert torch.equal(tokenized_cell_for_masking.input_ids, original_ids)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_single_maskable_token(self):
        """Cell with only CLS + 1 gene: should mask the 1 gene."""
        cell = TokenizedCell(
            input_ids=torch.tensor([CLS_TOKEN_ID, 10], dtype=torch.long),
            attention_mask=torch.ones(2, dtype=torch.long),
            gene_indices=torch.tensor([CLS_TOKEN_ID, 10], dtype=torch.long),
        )
        strategy = MaskingStrategy(mask_ratio=0.5, vocab_size=20)
        result = strategy.apply(cell, seed=42)
        assert result.masked_positions.sum().item() == 1
        assert result.masked_positions[1].item() is True

    def test_no_maskable_tokens(self):
        """Cell with only CLS — nothing to mask."""
        cell = TokenizedCell(
            input_ids=torch.tensor([CLS_TOKEN_ID], dtype=torch.long),
            attention_mask=torch.ones(1, dtype=torch.long),
            gene_indices=torch.tensor([CLS_TOKEN_ID], dtype=torch.long),
        )
        strategy = MaskingStrategy(mask_ratio=0.5, vocab_size=20)
        result = strategy.apply(cell, seed=42)
        assert result.masked_positions.sum().item() == 0

    def test_cell_without_values(self):
        """Masking works even if cell has no values tensor."""
        cell = TokenizedCell(
            input_ids=torch.tensor([5, 6, 7, 8], dtype=torch.long),
            attention_mask=torch.ones(4, dtype=torch.long),
            gene_indices=torch.tensor([5, 6, 7, 8], dtype=torch.long),
        )
        strategy = MaskingStrategy(mask_ratio=0.5, vocab_size=20)
        result = strategy.apply(cell, seed=42)
        assert result.values is None
        assert result.masked_positions.sum().item() >= 1


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


class TestValidation:
    def test_mask_ratio_zero(self):
        with pytest.raises(ValueError, match="mask_ratio"):
            MaskingStrategy(mask_ratio=0.0)

    def test_mask_ratio_negative(self):
        with pytest.raises(ValueError, match="mask_ratio"):
            MaskingStrategy(mask_ratio=-0.1)

    def test_mask_ratio_above_one(self):
        with pytest.raises(ValueError, match="mask_ratio"):
            MaskingStrategy(mask_ratio=1.5)

    def test_action_ratios_exceed_one(self):
        with pytest.raises(ValueError, match="must be <= 1.0"):
            MaskingStrategy(mask_ratio=0.15, mask_action_ratio=0.8, random_replace_ratio=0.3)

    def test_negative_action_ratio(self):
        with pytest.raises(ValueError):
            MaskingStrategy(mask_ratio=0.15, mask_action_ratio=-0.1)

    def test_missing_vocab_size_for_random(self):
        with pytest.raises(ValueError, match="vocab_size"):
            MaskingStrategy(mask_ratio=0.15, random_replace_ratio=0.1, vocab_size=None)

    def test_no_vocab_size_ok_when_no_random(self):
        """vocab_size not required when random_replace_ratio == 0."""
        strategy = MaskingStrategy(mask_ratio=0.15, random_replace_ratio=0.0, vocab_size=None)
        assert strategy.vocab_size is None
