"""Tests for GeneExpressionEmbedding component."""

from __future__ import annotations

import torch

from scmodelforge._constants import PAD_TOKEN_ID
from scmodelforge.models.components.embeddings import GeneExpressionEmbedding


class TestGeneExpressionEmbedding:
    """Tests for GeneExpressionEmbedding."""

    def test_output_shape(self):
        emb = GeneExpressionEmbedding(vocab_size=100, hidden_dim=64, max_seq_len=32)
        ids = torch.randint(1, 100, (2, 10))
        out = emb(ids)
        assert out.shape == (2, 10, 64)

    def test_with_expression_values(self):
        emb = GeneExpressionEmbedding(vocab_size=100, hidden_dim=64, use_expression_values=True)
        ids = torch.randint(1, 100, (2, 10))
        vals = torch.rand(2, 10)
        out = emb(ids, values=vals)
        assert out.shape == (2, 10, 64)

    def test_without_expression_values(self):
        emb = GeneExpressionEmbedding(vocab_size=100, hidden_dim=64, use_expression_values=False)
        ids = torch.randint(1, 100, (2, 10))
        out = emb(ids)
        assert out.shape == (2, 10, 64)
        assert not hasattr(emb, "expression_proj")

    def test_no_expression_proj_ignores_values(self):
        """When use_expression_values=False, passing values should have no effect."""
        emb = GeneExpressionEmbedding(vocab_size=100, hidden_dim=64, dropout=0.0, use_expression_values=False)
        ids = torch.randint(1, 100, (2, 10))
        vals = torch.rand(2, 10)
        out_no_vals = emb(ids)
        out_with_vals = emb(ids, values=vals)
        assert torch.equal(out_no_vals, out_with_vals)

    def test_padding_idx_zero_gradient(self):
        emb = GeneExpressionEmbedding(vocab_size=100, hidden_dim=64)
        assert emb.gene_embedding.padding_idx == PAD_TOKEN_ID
        # Embedding at padding idx should be zeros after init
        assert torch.all(emb.gene_embedding.weight[PAD_TOKEN_ID] == 0)

    def test_position_embedding_max_seq_len(self):
        emb = GeneExpressionEmbedding(vocab_size=100, hidden_dim=64, max_seq_len=16)
        assert emb.position_embedding.num_embeddings == 16
        ids = torch.randint(1, 100, (1, 16))
        out = emb(ids)
        assert out.shape == (1, 16, 64)

    def test_dropout_applied(self):
        emb = GeneExpressionEmbedding(vocab_size=100, hidden_dim=64, dropout=0.5)
        emb.train()
        ids = torch.randint(1, 100, (4, 10))
        # With high dropout, outputs should differ across calls (stochastic)
        out1 = emb(ids)
        out2 = emb(ids)
        # Very unlikely to be exactly equal with 50% dropout
        assert not torch.equal(out1, out2)

    def test_no_dropout_deterministic(self):
        emb = GeneExpressionEmbedding(vocab_size=100, hidden_dim=64, dropout=0.0)
        ids = torch.randint(1, 100, (2, 10))
        out1 = emb(ids)
        out2 = emb(ids)
        assert torch.equal(out1, out2)
