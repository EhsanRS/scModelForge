"""Tests for eval._utils — embedding extraction."""

from __future__ import annotations

import numpy as np

from scmodelforge.eval._utils import extract_embeddings


class TestExtractEmbeddings:
    """Tests for extract_embeddings()."""

    def test_returns_correct_shape(self, tiny_model, tiny_adata, tiny_tokenizer):
        emb = extract_embeddings(tiny_model, tiny_adata, tiny_tokenizer, batch_size=16)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (tiny_adata.n_obs, 32)  # 40 cells, hidden_dim=32

    def test_returns_float_array(self, tiny_model, tiny_adata, tiny_tokenizer):
        emb = extract_embeddings(tiny_model, tiny_adata, tiny_tokenizer, batch_size=16)
        assert emb.dtype == np.float32

    def test_deterministic(self, tiny_model, tiny_adata, tiny_tokenizer):
        emb1 = extract_embeddings(tiny_model, tiny_adata, tiny_tokenizer, batch_size=16)
        emb2 = extract_embeddings(tiny_model, tiny_adata, tiny_tokenizer, batch_size=16)
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_batch_sizes(self, tiny_model, tiny_adata, tiny_tokenizer):
        emb_small = extract_embeddings(tiny_model, tiny_adata, tiny_tokenizer, batch_size=8)
        emb_large = extract_embeddings(tiny_model, tiny_adata, tiny_tokenizer, batch_size=64)
        np.testing.assert_allclose(emb_small, emb_large, atol=1e-5)

    def test_preserves_training_state(self, tiny_model, tiny_adata, tiny_tokenizer):
        tiny_model.train()
        assert tiny_model.training
        extract_embeddings(tiny_model, tiny_adata, tiny_tokenizer, batch_size=16)
        assert tiny_model.training

    def test_preserves_eval_state(self, tiny_model, tiny_adata, tiny_tokenizer):
        tiny_model.eval()
        assert not tiny_model.training
        extract_embeddings(tiny_model, tiny_adata, tiny_tokenizer, batch_size=16)
        assert not tiny_model.training

    def test_embeddings_vary_across_cells(self, tiny_model, tiny_adata, tiny_tokenizer):
        emb = extract_embeddings(tiny_model, tiny_adata, tiny_tokenizer, batch_size=16)
        # Not all cells should have identical embeddings
        assert not np.allclose(emb[0], emb[-1])
