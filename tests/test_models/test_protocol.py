"""Tests for ModelOutput dataclass and ModelProtocol compliance."""

from __future__ import annotations

import pytest
import torch

from scmodelforge._types import ModelProtocol
from scmodelforge.models.protocol import ModelOutput


class TestModelOutput:
    """Tests for ModelOutput dataclass."""

    def test_create_empty(self):
        out = ModelOutput()
        assert out.loss is None
        assert out.logits is None
        assert out.embeddings is None
        assert out.hidden_states is None

    def test_create_with_tensors(self):
        loss = torch.tensor(1.5)
        logits = torch.randn(2, 10, 100)
        embeddings = torch.randn(2, 64)
        out = ModelOutput(loss=loss, logits=logits, embeddings=embeddings)
        assert torch.equal(out.loss, loss)
        assert out.logits.shape == (2, 10, 100)
        assert out.embeddings.shape == (2, 64)
        assert out.hidden_states is None

    def test_create_with_hidden_states(self):
        h1 = torch.randn(2, 10, 64)
        h2 = torch.randn(2, 10, 64)
        out = ModelOutput(hidden_states=(h1, h2))
        assert len(out.hidden_states) == 2
        assert out.hidden_states[0].shape == (2, 10, 64)

    def test_frozen(self):
        out = ModelOutput()
        with pytest.raises(AttributeError):
            out.loss = torch.tensor(0.0)


class TestModelProtocol:
    """Test that TransformerEncoder satisfies ModelProtocol."""

    def test_transformer_encoder_is_model_protocol(self):
        from scmodelforge.models import TransformerEncoder

        model = TransformerEncoder(vocab_size=50, hidden_dim=32, num_layers=1, num_heads=2)
        assert isinstance(model, ModelProtocol)

    def test_protocol_forward_signature(self):
        from scmodelforge.models import TransformerEncoder

        model = TransformerEncoder(vocab_size=50, hidden_dim=32, num_layers=1, num_heads=2)
        ids = torch.randint(0, 50, (1, 5))
        mask = torch.ones(1, 5, dtype=torch.long)
        out = model.forward(ids, mask)
        assert isinstance(out, ModelOutput)

    def test_protocol_encode_signature(self):
        from scmodelforge.models import TransformerEncoder

        model = TransformerEncoder(vocab_size=50, hidden_dim=32, num_layers=1, num_heads=2)
        ids = torch.randint(0, 50, (1, 5))
        mask = torch.ones(1, 5, dtype=torch.long)
        emb = model.encode(ids, mask)
        assert emb.shape == (1, 32)
