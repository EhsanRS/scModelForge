"""Tests for BinnedExpressionTokenizer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scmodelforge._constants import CLS_TOKEN_ID, NUM_SPECIAL_TOKENS, PAD_TOKEN_ID
from scmodelforge.tokenizers.binned_expression import BinnedExpressionTokenizer


class TestBinnedExpressionTokenizer:
    """Core functionality tests."""

    def test_basic_tokenize(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20)
        cell = tok.tokenize(sample_expression, sample_gene_indices)

        # Should have CLS + 10 genes = 11 tokens
        assert cell.input_ids.shape[0] == 11
        assert cell.attention_mask.shape[0] == 11
        assert cell.values.shape[0] == 11
        assert cell.bin_ids.shape[0] == 11
        assert cell.gene_indices.shape[0] == 11

    def test_cls_token_prepended(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)

        assert cell.input_ids[0].item() == CLS_TOKEN_ID
        assert cell.values[0].item() == 0.0
        assert cell.bin_ids[0].item() == 0

    def test_no_cls(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)

        assert cell.input_ids[0].item() != CLS_TOKEN_ID
        assert cell.input_ids.shape[0] == 10  # all 10 genes, no CLS

    def test_fixed_gene_order(self, small_vocab, sample_expression, sample_gene_indices):
        """Genes should maintain their original order (not sorted by expression)."""
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)

        expected_genes = torch.as_tensor(sample_gene_indices, dtype=torch.long)
        torch.testing.assert_close(cell.input_ids, expected_genes)

    def test_bin_ids_populated(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)

        assert cell.bin_ids is not None
        assert cell.bin_ids.dtype == torch.long
        # Zero expression genes should have bin_id=0
        zero_mask = torch.as_tensor(sample_expression, dtype=torch.float32) == 0
        assert (cell.bin_ids[zero_mask] == 0).all()

    def test_values_preserved(self, small_vocab, sample_expression, sample_gene_indices):
        """Original continuous expression values should be preserved."""
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)

        expected = torch.as_tensor(sample_expression, dtype=torch.float32)
        torch.testing.assert_close(cell.values, expected)

    def test_include_zero_genes_true(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20, include_zero_genes=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        # All 10 genes + CLS
        assert cell.input_ids.shape[0] == 11

    def test_include_zero_genes_false(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20, include_zero_genes=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        # sample_expression has 3 zeros, so 7 non-zero + CLS = 8
        assert cell.input_ids.shape[0] == 8

    def test_attention_mask_all_ones(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert (cell.attention_mask == 1).all()


class TestBinnedExpressionTruncation:
    def test_truncation(self, large_vocab, large_expression, large_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=large_vocab, max_len=10, prepend_cls=True)
        cell = tok.tokenize(large_expression, large_gene_indices)
        # max_len=10, CLS + 9 genes
        assert cell.input_ids.shape[0] == 10

    def test_truncation_no_cls(self, large_vocab, large_expression, large_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=large_vocab, max_len=10, prepend_cls=False)
        cell = tok.tokenize(large_expression, large_gene_indices)
        assert cell.input_ids.shape[0] == 10


class TestBinnedExpressionBatch:
    def test_batch_tokenize(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20)
        batch = tok.tokenize_batch(
            [sample_expression, sample_expression[:5]],
            [sample_gene_indices, sample_gene_indices[:5]],
        )

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "values" in batch
        assert "bin_ids" in batch
        assert batch["input_ids"].shape[0] == 2
        # Padded to batch max length
        assert batch["input_ids"].shape[1] == batch["attention_mask"].shape[1]

    def test_batch_padding(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20)
        short_expr = sample_expression[:3]
        short_genes = sample_gene_indices[:3]
        batch = tok.tokenize_batch(
            [sample_expression, short_expr],
            [sample_gene_indices, short_genes],
        )
        # Second cell is shorter so padding should be applied
        assert batch["attention_mask"][1, -1].item() == 0
        assert batch["input_ids"][1, -1].item() == PAD_TOKEN_ID


class TestBinnedExpressionFit:
    def test_fit_quantile(self, small_vocab):
        tok = BinnedExpressionTokenizer(
            gene_vocab=small_vocab,
            n_bins=10,
            binning_method="quantile",
        )
        assert tok.bin_edges is None

        rng = np.random.default_rng(42)
        values = rng.uniform(0.1, 10.0, size=1000).astype(np.float32)
        result = tok.fit(values)

        assert result is tok  # returns self
        assert tok.bin_edges is not None

    def test_tokenize_before_fit_raises(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(
            gene_vocab=small_vocab,
            n_bins=10,
            binning_method="quantile",
        )
        with pytest.raises(RuntimeError, match="Bin edges not set"):
            tok.tokenize(sample_expression, sample_gene_indices)

    def test_custom_bin_edges(self, small_vocab, sample_expression, sample_gene_indices):
        edges = np.linspace(0, 10, 6)  # 5 bins
        tok = BinnedExpressionTokenizer(
            gene_vocab=small_vocab,
            bin_edges=edges,
            prepend_cls=False,
        )
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.bin_ids is not None
        assert all(0 <= b < 5 for b in cell.bin_ids)


class TestBinnedExpressionProperties:
    def test_strategy_name(self, small_vocab):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab)
        assert tok.strategy_name == "binned_expression"

    def test_vocab_size(self, small_vocab):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab)
        assert tok.vocab_size == len(small_vocab)

    def test_n_bin_tokens(self, small_vocab):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, n_bins=51)
        assert tok.n_bin_tokens == 51


class TestBinnedExpressionEdgeCases:
    def test_all_zeros(self, small_vocab):
        expr = np.zeros(10, dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(expr, genes)
        # All bin_ids should be 0
        assert (cell.bin_ids == 0).all()

    def test_all_zeros_exclude_zero(self, small_vocab):
        expr = np.zeros(10, dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)
        tok = BinnedExpressionTokenizer(
            gene_vocab=small_vocab, max_len=20, prepend_cls=False, include_zero_genes=False
        )
        cell = tok.tokenize(expr, genes)
        # No genes to tokenize
        assert cell.input_ids.shape[0] == 0

    def test_single_gene(self, small_vocab):
        expr = np.array([5.0], dtype=np.float32)
        genes = np.array([NUM_SPECIAL_TOKENS], dtype=np.int64)
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(expr, genes)
        assert cell.input_ids.shape[0] == 2  # CLS + 1 gene

    def test_metadata_passthrough(self, small_vocab, sample_expression, sample_gene_indices):
        tok = BinnedExpressionTokenizer(gene_vocab=small_vocab, max_len=20)
        meta = {"cell_type": "T-cell", "batch": 0}
        cell = tok.tokenize(sample_expression, sample_gene_indices, metadata=meta)
        assert cell.metadata == meta
