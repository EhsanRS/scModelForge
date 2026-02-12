"""Tests for the tokenizer registry."""

from __future__ import annotations

import pytest

from scmodelforge.tokenizers import get_tokenizer, list_tokenizers, register_tokenizer
from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell
from scmodelforge.tokenizers.registry import _REGISTRY

# ------------------------------------------------------------------
# Registration and lookup
# ------------------------------------------------------------------


class TestRegistration:
    def test_rank_value_registered(self):
        assert "rank_value" in _REGISTRY

    def test_binned_expression_registered(self):
        assert "binned_expression" in _REGISTRY

    def test_continuous_projection_registered(self):
        assert "continuous_projection" in _REGISTRY

    def test_list_tokenizers(self):
        names = list_tokenizers()
        assert "rank_value" in names
        assert "binned_expression" in names
        assert "continuous_projection" in names
        assert names == sorted(names)

    def test_get_tokenizer(self, small_vocab):
        tok = get_tokenizer("rank_value", gene_vocab=small_vocab)
        assert tok.strategy_name == "rank_value"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown tokenizer 'nonexistent'"):
            get_tokenizer("nonexistent")

    def test_unknown_shows_available(self):
        with pytest.raises(ValueError, match="rank_value"):
            get_tokenizer("nonexistent")

    def test_custom_registration(self, small_vocab):
        """Register a custom tokenizer and look it up."""

        @register_tokenizer("test_custom")
        class CustomTokenizer(BaseTokenizer):
            @property
            def vocab_size(self) -> int:
                return len(self.gene_vocab)

            @property
            def strategy_name(self) -> str:
                return "test_custom"

            def tokenize(self, expression, gene_indices, metadata=None) -> TokenizedCell:
                import torch

                return TokenizedCell(
                    input_ids=torch.tensor([0], dtype=torch.long),
                    attention_mask=torch.ones(1, dtype=torch.long),
                )

        try:
            assert "test_custom" in list_tokenizers()
            tok = get_tokenizer("test_custom", gene_vocab=small_vocab)
            assert tok.strategy_name == "test_custom"
        finally:
            # Clean up registry to not affect other tests
            _REGISTRY.pop("test_custom", None)

    def test_duplicate_registration_raises(self):
        """Re-registering the same name should raise."""
        with pytest.raises(ValueError, match="already registered"):

            @register_tokenizer("rank_value")
            class DuplicateTokenizer(BaseTokenizer):
                @property
                def vocab_size(self) -> int:
                    return 1

                @property
                def strategy_name(self) -> str:
                    return "rank_value"

                def tokenize(self, expression, gene_indices, metadata=None):
                    pass
