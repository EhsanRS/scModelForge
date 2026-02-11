"""Shared fixtures for tokenizer tests."""

from __future__ import annotations

import numpy as np
import pytest

from scmodelforge._constants import NUM_SPECIAL_TOKENS
from scmodelforge.data.gene_vocab import GeneVocab


@pytest.fixture()
def small_vocab() -> GeneVocab:
    """A small gene vocabulary with 10 genes (total size = 10 + special tokens)."""
    genes = [f"GENE_{i}" for i in range(10)]
    return GeneVocab.from_genes(genes)


@pytest.fixture()
def sample_expression() -> np.ndarray:
    """Expression vector for 10 genes with a mix of zero and non-zero values.

    Indices (relative to gene list):
        0: 0.0, 1: 5.0, 2: 3.0, 3: 0.0, 4: 8.0, 5: 1.0, 6: 0.0, 7: 2.0, 8: 8.0, 9: 4.0
    Non-zero genes sorted descending by expression: 4(8), 8(8), 1(5), 9(4), 2(3), 7(2), 5(1)
    """
    return np.array([0.0, 5.0, 3.0, 0.0, 8.0, 1.0, 0.0, 2.0, 8.0, 4.0], dtype=np.float32)


@pytest.fixture()
def sample_gene_indices() -> np.ndarray:
    """Gene vocab indices for 10 genes (offset by NUM_SPECIAL_TOKENS)."""
    return np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)


@pytest.fixture()
def large_expression() -> np.ndarray:
    """Large expression vector with 100 non-zero genes for truncation tests."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 10.0, size=100).astype(np.float32)


@pytest.fixture()
def large_gene_indices() -> np.ndarray:
    """Gene vocab indices for 100 genes."""
    return np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 100, dtype=np.int64)


@pytest.fixture()
def large_vocab() -> GeneVocab:
    """A larger vocabulary with 100 genes."""
    genes = [f"GENE_{i}" for i in range(100)]
    return GeneVocab.from_genes(genes)


@pytest.fixture()
def tokenized_cell_for_masking(small_vocab, sample_expression, sample_gene_indices):
    """A pre-tokenized cell ready for masking tests."""
    from scmodelforge.tokenizers import RankValueTokenizer

    tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
    return tok.tokenize(sample_expression, sample_gene_indices)


@pytest.fixture()
def tokenized_cell_no_cls(small_vocab, sample_expression, sample_gene_indices):
    """A pre-tokenized cell without CLS for masking tests."""
    from scmodelforge.tokenizers import RankValueTokenizer

    tok = RankValueTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
    return tok.tokenize(sample_expression, sample_gene_indices)
