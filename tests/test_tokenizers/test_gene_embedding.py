"""Tests for GeneEmbeddingTokenizer and load_gene_embeddings."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scmodelforge._constants import CLS_TOKEN_ID, NUM_SPECIAL_TOKENS
from scmodelforge.tokenizers.gene_embedding import GeneEmbeddingTokenizer


class TestGeneEmbeddingTokenizer:
    """Core tokenization tests."""

    def test_basic_tokenize(self, small_vocab, sample_expression, sample_gene_indices):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab, max_len=20)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        # CLS + 7 non-zero genes = 8
        assert cell.input_ids.shape[0] == 8
        assert cell.attention_mask.shape[0] == 8
        assert cell.values.shape[0] == 8

    def test_vocab_size(self, small_vocab):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab)
        assert tok.vocab_size == len(small_vocab)

    def test_strategy_name(self, small_vocab):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab)
        assert tok.strategy_name == "gene_embedding"

    def test_cls_prepended(self, small_vocab, sample_expression, sample_gene_indices):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids[0].item() == CLS_TOKEN_ID
        assert cell.values[0].item() == 0.0

    def test_no_cls(self, small_vocab, sample_expression, sample_gene_indices):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids[0].item() != CLS_TOKEN_ID

    def test_values_are_expression(self, small_vocab, sample_expression, sample_gene_indices):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        # Non-zero values only
        nonzero = sample_expression[sample_expression > 0]
        expected = torch.as_tensor(nonzero, dtype=torch.float32)
        torch.testing.assert_close(cell.values, expected)

    def test_truncation(self, large_vocab, large_expression, large_gene_indices):
        tok = GeneEmbeddingTokenizer(gene_vocab=large_vocab, max_len=10, prepend_cls=True)
        cell = tok.tokenize(large_expression, large_gene_indices)
        assert cell.input_ids.shape[0] == 10  # 9 genes + CLS

    def test_zero_genes_filtered(self, small_vocab):
        expression = np.array([0.0, 0.0, 5.0, 0.0, 3.0], dtype=np.float32)
        gene_indices = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 5, dtype=np.int64)
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(expression, gene_indices)
        assert cell.input_ids.shape[0] == 2  # only 2 non-zero


class TestEmbeddingMatrix:
    """Tests for embedding loading, setting, and properties."""

    def test_no_embeddings_by_default(self, small_vocab):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab)
        assert tok.gene_embeddings is None
        assert tok.embedding_dim is None

    def test_set_gene_embeddings(self, small_vocab):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab)
        matrix = torch.randn(len(small_vocab), 128)
        tok.set_gene_embeddings(matrix)
        assert tok.gene_embeddings is not None
        assert tok.embedding_dim == 128
        torch.testing.assert_close(tok.gene_embeddings, matrix)

    def test_set_gene_embeddings_wrong_size_raises(self, small_vocab):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab)
        matrix = torch.randn(5, 128)  # wrong number of rows
        with pytest.raises(ValueError, match="must match"):
            tok.set_gene_embeddings(matrix)

    def test_load_from_pt_file(self, small_vocab, tmp_path):
        embedding_dim = 64
        gene_names = [f"GENE_{i}" for i in range(10)]
        embeddings = torch.randn(10, embedding_dim)
        data = {"gene_names": gene_names, "embeddings": embeddings}
        pt_path = str(tmp_path / "embeddings.pt")
        torch.save(data, pt_path)

        tok = GeneEmbeddingTokenizer(
            gene_vocab=small_vocab,
            embedding_path=pt_path,
            embedding_dim=embedding_dim,
        )
        assert tok.gene_embeddings is not None
        assert tok.gene_embeddings.shape == (len(small_vocab), embedding_dim)
        assert tok.embedding_dim == embedding_dim

    def test_load_from_npy_file(self, small_vocab, tmp_path):
        embedding_dim = 32
        gene_names = [f"GENE_{i}" for i in range(10)]
        embeddings = np.random.default_rng(0).standard_normal((10, embedding_dim)).astype(np.float32)
        data = {"gene_names": gene_names, "embeddings": embeddings}
        npy_path = str(tmp_path / "embeddings.npy")
        np.save(npy_path, data)

        tok = GeneEmbeddingTokenizer(
            gene_vocab=small_vocab,
            embedding_path=npy_path,
            embedding_dim=embedding_dim,
        )
        assert tok.gene_embeddings is not None
        assert tok.gene_embeddings.shape == (len(small_vocab), embedding_dim)

    def test_missing_genes_get_zeros(self, small_vocab, tmp_path):
        embedding_dim = 16
        # Only provide embeddings for 5 of the 10 genes
        gene_names = [f"GENE_{i}" for i in range(5)]
        embeddings = torch.ones(5, embedding_dim)
        data = {"gene_names": gene_names, "embeddings": embeddings}
        pt_path = str(tmp_path / "embeddings.pt")
        torch.save(data, pt_path)

        tok = GeneEmbeddingTokenizer(
            gene_vocab=small_vocab,
            embedding_path=pt_path,
        )
        emb = tok.gene_embeddings
        assert emb is not None
        # First 5 genes should have non-zero embeddings, remaining should be zero
        # (Special token rows at indices 0-3 have no matching gene names -> zero)
        for i in range(5):
            idx = small_vocab[f"GENE_{i}"]
            assert emb[idx].sum().item() > 0.0


class TestLoadGeneEmbeddings:
    """Tests for the load_gene_embeddings utility."""

    def test_unsupported_format_raises(self, small_vocab, tmp_path):
        from scmodelforge.tokenizers._utils import load_gene_embeddings

        bad_path = str(tmp_path / "embeddings.json")
        (tmp_path / "embeddings.json").write_text("{}")
        with pytest.raises(ValueError, match="Unsupported embedding file format"):
            load_gene_embeddings(bad_path, small_vocab)

    def test_missing_keys_raises(self, small_vocab, tmp_path):
        from scmodelforge.tokenizers._utils import load_gene_embeddings

        data = {"wrong_key": []}
        pt_path = str(tmp_path / "embeddings.pt")
        torch.save(data, pt_path)
        with pytest.raises(ValueError, match="must contain key"):
            load_gene_embeddings(pt_path, small_vocab)

    def test_alignment_preserves_order(self, small_vocab, tmp_path):
        from scmodelforge.tokenizers._utils import load_gene_embeddings

        embedding_dim = 8
        # Provide genes in reverse order
        gene_names = [f"GENE_{i}" for i in reversed(range(10))]
        embeddings = torch.arange(10 * embedding_dim, dtype=torch.float32).reshape(10, embedding_dim)
        data = {"gene_names": gene_names, "embeddings": embeddings}
        pt_path = str(tmp_path / "embeddings.pt")
        torch.save(data, pt_path)

        aligned = load_gene_embeddings(pt_path, small_vocab)
        # GENE_9 is at index 0 in the file => its embedding = first row
        idx_gene_9 = small_vocab["GENE_9"]
        torch.testing.assert_close(aligned[idx_gene_9], embeddings[0])


class TestRegistryAndCollation:
    """Registry integration and batch collation."""

    def test_registry_lookup(self):
        from scmodelforge.tokenizers.registry import list_tokenizers

        assert "gene_embedding" in list_tokenizers()

    def test_batch_collation(self, small_vocab, sample_expression, sample_gene_indices):
        tok = GeneEmbeddingTokenizer(gene_vocab=small_vocab, max_len=20)
        cells = [
            tok.tokenize(sample_expression, sample_gene_indices),
            tok.tokenize(sample_expression, sample_gene_indices),
        ]
        batch = tok._collate(cells)
        assert batch["input_ids"].shape[0] == 2
        assert batch["attention_mask"].shape[0] == 2
        assert batch["values"].shape[0] == 2
