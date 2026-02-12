"""Tests for ContinuousProjectionTokenizer."""

from __future__ import annotations

import numpy as np
import torch

from scmodelforge._constants import CLS_TOKEN_ID, NUM_SPECIAL_TOKENS, PAD_TOKEN_ID
from scmodelforge.tokenizers.continuous_projection import ContinuousProjectionTokenizer


class TestContinuousProjectionTokenizer:
    """Core functionality tests."""

    def test_basic_tokenize(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20)
        cell = tok.tokenize(sample_expression, sample_gene_indices)

        # CLS + 10 genes = 11
        assert cell.input_ids.shape[0] == 11
        assert cell.attention_mask.shape[0] == 11
        assert cell.values.shape[0] == 11

    def test_no_bin_ids(self, small_vocab, sample_expression, sample_gene_indices):
        """ContinuousProjectionTokenizer should NOT populate bin_ids."""
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.bin_ids is None

    def test_cls_token_prepended(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids[0].item() == CLS_TOKEN_ID
        assert cell.values[0].item() == 0.0

    def test_no_cls(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids[0].item() != CLS_TOKEN_ID
        assert cell.input_ids.shape[0] == 10

    def test_fixed_gene_order(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        expected = torch.as_tensor(sample_gene_indices, dtype=torch.long)
        torch.testing.assert_close(cell.input_ids, expected)

    def test_values_are_expression(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        expected = torch.as_tensor(sample_expression, dtype=torch.float32)
        torch.testing.assert_close(cell.values, expected)

    def test_include_zero_genes_true(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20, include_zero_genes=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids.shape[0] == 11  # CLS + 10

    def test_include_zero_genes_false(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20, include_zero_genes=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        # 3 zeros in sample_expression, so 7 non-zero + CLS = 8
        assert cell.input_ids.shape[0] == 8


class TestContinuousProjectionLogTransform:
    def test_log_transform_applied(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(
            gene_vocab=small_vocab, max_len=20, prepend_cls=False, log_transform=True
        )
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        expected = torch.log1p(torch.as_tensor(sample_expression, dtype=torch.float32))
        torch.testing.assert_close(cell.values, expected)

    def test_log_transform_off(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(
            gene_vocab=small_vocab, max_len=20, prepend_cls=False, log_transform=False
        )
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        expected = torch.as_tensor(sample_expression, dtype=torch.float32)
        torch.testing.assert_close(cell.values, expected)


class TestContinuousProjectionTruncation:
    def test_truncation(self, large_vocab, large_expression, large_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=large_vocab, max_len=10, prepend_cls=True)
        cell = tok.tokenize(large_expression, large_gene_indices)
        assert cell.input_ids.shape[0] == 10

    def test_truncation_no_cls(self, large_vocab, large_expression, large_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=large_vocab, max_len=10, prepend_cls=False)
        cell = tok.tokenize(large_expression, large_gene_indices)
        assert cell.input_ids.shape[0] == 10


class TestContinuousProjectionBatch:
    def test_batch_tokenize(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20)
        batch = tok.tokenize_batch(
            [sample_expression, sample_expression[:5]],
            [sample_gene_indices, sample_gene_indices[:5]],
        )
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "values" in batch
        assert "bin_ids" not in batch  # continuous tokenizer has no bin_ids
        assert batch["input_ids"].shape[0] == 2

    def test_batch_padding(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20)
        short_expr = sample_expression[:3]
        short_genes = sample_gene_indices[:3]
        batch = tok.tokenize_batch(
            [sample_expression, short_expr],
            [sample_gene_indices, short_genes],
        )
        assert batch["attention_mask"][1, -1].item() == 0
        assert batch["input_ids"][1, -1].item() == PAD_TOKEN_ID


class TestContinuousProjectionProperties:
    def test_strategy_name(self, small_vocab):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab)
        assert tok.strategy_name == "continuous_projection"

    def test_vocab_size(self, small_vocab):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab)
        assert tok.vocab_size == len(small_vocab)


class TestContinuousProjectionEdgeCases:
    def test_all_zeros(self, small_vocab):
        expr = np.zeros(10, dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(expr, genes)
        assert (cell.values == 0.0).all()

    def test_all_zeros_exclude_zero(self, small_vocab):
        expr = np.zeros(10, dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)
        tok = ContinuousProjectionTokenizer(
            gene_vocab=small_vocab, max_len=20, prepend_cls=False, include_zero_genes=False
        )
        cell = tok.tokenize(expr, genes)
        assert cell.input_ids.shape[0] == 0

    def test_metadata_passthrough(self, small_vocab, sample_expression, sample_gene_indices):
        tok = ContinuousProjectionTokenizer(gene_vocab=small_vocab, max_len=20)
        meta = {"cell_type": "neuron"}
        cell = tok.tokenize(sample_expression, sample_gene_indices, metadata=meta)
        assert cell.metadata == meta
