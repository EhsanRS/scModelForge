"""Tests for RankValueTokenizer."""

from __future__ import annotations

import numpy as np
import torch

from scmodelforge._constants import CLS_TOKEN_ID, NUM_SPECIAL_TOKENS
from scmodelforge.tokenizers import RankValueTokenizer

# ------------------------------------------------------------------
# Ranking correctness
# ------------------------------------------------------------------


class TestRanking:
    def test_descending_order(self, small_vocab, sample_expression, sample_gene_indices):
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        # Non-zero values: idx1=5, idx2=3, idx4=8, idx5=1, idx7=2, idx8=8, idx9=4
        # Descending: 8,8,5,4,3,2,1 → genes 4,8,1,9,2,7,5 (offset by special tokens)
        vals = cell.values
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1], f"values not descending at position {i}"

    def test_nonzero_only(self, small_vocab, sample_expression, sample_gene_indices):
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        # 7 non-zero values in sample_expression
        assert cell.input_ids.shape[0] == 7

    def test_gene_indices_are_input_ids(self, small_vocab, sample_expression, sample_gene_indices):
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert torch.equal(cell.input_ids, cell.gene_indices)

    def test_stable_sort_for_ties(self, small_vocab, sample_expression, sample_gene_indices):
        """Genes with tied expression (idx4=8, idx8=8) should keep original order."""
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        # First two should be gene indices 4 and 8 (offset by NUM_SPECIAL_TOKENS)
        assert cell.input_ids[0].item() == NUM_SPECIAL_TOKENS + 4
        assert cell.input_ids[1].item() == NUM_SPECIAL_TOKENS + 8


# ------------------------------------------------------------------
# CLS token
# ------------------------------------------------------------------


class TestCLS:
    def test_prepend_cls(self, small_vocab, sample_expression, sample_gene_indices):
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids[0].item() == CLS_TOKEN_ID
        assert cell.values[0].item() == 0.0
        # 7 non-zero + 1 CLS = 8
        assert cell.input_ids.shape[0] == 8

    def test_no_cls(self, small_vocab, sample_expression, sample_gene_indices):
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids[0].item() != CLS_TOKEN_ID
        assert cell.input_ids.shape[0] == 7

    def test_cls_value_is_zero(self, small_vocab, sample_expression, sample_gene_indices):
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.values[0].item() == 0.0


# ------------------------------------------------------------------
# Truncation
# ------------------------------------------------------------------


class TestTruncation:
    def test_truncation_with_cls(self, large_vocab, large_expression, large_gene_indices):
        max_len = 20
        tok = RankValueTokenizer(gene_vocab=large_vocab, max_len=max_len, prepend_cls=True)
        cell = tok.tokenize(large_expression, large_gene_indices)
        assert cell.input_ids.shape[0] == max_len
        assert cell.input_ids[0].item() == CLS_TOKEN_ID
        # 19 genes + 1 CLS = 20
        assert cell.values.shape[0] == max_len

    def test_truncation_without_cls(self, large_vocab, large_expression, large_gene_indices):
        max_len = 20
        tok = RankValueTokenizer(gene_vocab=large_vocab, max_len=max_len, prepend_cls=False)
        cell = tok.tokenize(large_expression, large_gene_indices)
        assert cell.input_ids.shape[0] == max_len

    def test_no_truncation_when_short(self, small_vocab, sample_expression, sample_gene_indices):
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=100, prepend_cls=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        # 7 non-zero + CLS = 8, well under 100
        assert cell.input_ids.shape[0] == 8


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_cell_with_cls(self, small_vocab):
        """All-zero expression with CLS → length-1 sequence [CLS]."""
        expr = np.zeros(10, dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(expr, genes)
        assert cell.input_ids.shape[0] == 1
        assert cell.input_ids[0].item() == CLS_TOKEN_ID

    def test_empty_cell_without_cls(self, small_vocab):
        """All-zero expression without CLS → length-0 sequence."""
        expr = np.zeros(10, dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(expr, genes)
        assert cell.input_ids.shape[0] == 0

    def test_single_gene(self, small_vocab):
        """Only one non-zero gene."""
        expr = np.array([0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(expr, genes)
        # CLS + 1 gene = 2
        assert cell.input_ids.shape[0] == 2
        assert cell.input_ids[0].item() == CLS_TOKEN_ID
        assert cell.input_ids[1].item() == NUM_SPECIAL_TOKENS + 2
        assert cell.values[1].item() == 5.0

    def test_all_same_expression(self, small_vocab):
        """All genes have the same expression — stable sort preserves order."""
        expr = np.full(10, 3.0, dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(expr, genes)
        # Stable sort should preserve original order
        expected = torch.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=torch.long)
        assert torch.equal(cell.input_ids, expected)

    def test_torch_input(self, small_vocab):
        """Accept torch tensors as input."""
        expr = torch.tensor([0.0, 5.0, 3.0, 0.0], dtype=torch.float32)
        genes = torch.tensor([4, 5, 6, 7], dtype=torch.long)
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(expr, genes)
        assert cell.input_ids.shape[0] == 2


# ------------------------------------------------------------------
# Attention mask
# ------------------------------------------------------------------


class TestAttentionMask:
    def test_all_ones(self, small_vocab, sample_expression, sample_gene_indices):
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert (cell.attention_mask == 1).all()
        assert cell.attention_mask.shape == cell.input_ids.shape


# ------------------------------------------------------------------
# Batch tokenization
# ------------------------------------------------------------------


class TestBatch:
    def test_batch_padding(self, small_vocab):
        """Batch of cells with different lengths should be padded to batch max."""
        # Cell 1: 3 non-zero genes
        expr1 = np.array([0, 1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        # Cell 2: 5 non-zero genes
        expr2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0, 0, 0, 0, 0], dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)

        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        batch = tok.tokenize_batch([expr1, expr2], [genes, genes])

        # Cell 1: CLS + 3 = 4; Cell 2: CLS + 5 = 6 → pad to 6
        assert batch["input_ids"].shape == (2, 6)
        assert batch["attention_mask"].shape == (2, 6)
        # Cell 1 should have 2 padding tokens
        assert batch["attention_mask"][0].sum().item() == 4
        assert batch["attention_mask"][1].sum().item() == 6

    def test_batch_single_cell(self, small_vocab, sample_expression, sample_gene_indices):
        tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        batch = tok.tokenize_batch([sample_expression], [sample_gene_indices])
        assert batch["input_ids"].shape[0] == 1


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------


class TestProperties:
    def test_vocab_size(self, small_vocab):
        tok = RankValueTokenizer(gene_vocab=small_vocab)
        assert tok.vocab_size == len(small_vocab)

    def test_strategy_name(self, small_vocab):
        tok = RankValueTokenizer(gene_vocab=small_vocab)
        assert tok.strategy_name == "rank_value"
