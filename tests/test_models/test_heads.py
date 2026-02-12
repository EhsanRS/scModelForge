"""Tests for prediction heads."""

from __future__ import annotations

import torch

from scmodelforge.models.components.heads import ExpressionPredictionHead, MaskedGenePredictionHead


class TestMaskedGenePredictionHead:
    """Tests for MaskedGenePredictionHead."""

    def test_output_shape(self):
        head = MaskedGenePredictionHead(hidden_dim=64, vocab_size=100)
        hidden = torch.randn(2, 10, 64)
        logits = head(hidden)
        assert logits.shape == (2, 10, 100)

    def test_different_vocab_sizes(self):
        for v in [50, 200, 1000]:
            head = MaskedGenePredictionHead(hidden_dim=32, vocab_size=v)
            hidden = torch.randn(1, 5, 32)
            assert head(hidden).shape == (1, 5, v)

    def test_gradient_flows(self):
        head = MaskedGenePredictionHead(hidden_dim=64, vocab_size=100)
        hidden = torch.randn(2, 10, 64, requires_grad=True)
        logits = head(hidden)
        loss = logits.sum()
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.shape == (2, 10, 64)

    def test_loss_computation(self):
        head = MaskedGenePredictionHead(hidden_dim=64, vocab_size=100)
        hidden = torch.randn(2, 10, 64)
        logits = head(hidden)
        labels = torch.randint(0, 100, (2, 10))
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, 100), labels.view(-1))
        assert loss.shape == ()
        assert loss.item() > 0


class TestExpressionPredictionHead:
    """Tests for ExpressionPredictionHead."""

    def test_output_shape(self):
        head = ExpressionPredictionHead(hidden_dim=64)
        hidden = torch.randn(2, 10, 64)
        out = head(hidden)
        assert out.shape == (2, 10)

    def test_single_sample(self):
        head = ExpressionPredictionHead(hidden_dim=32)
        hidden = torch.randn(1, 5, 32)
        out = head(hidden)
        assert out.shape == (1, 5)

    def test_gradient_flows(self):
        head = ExpressionPredictionHead(hidden_dim=64)
        hidden = torch.randn(2, 10, 64, requires_grad=True)
        out = head(hidden)
        loss = out.sum()
        loss.backward()
        assert hidden.grad is not None

    def test_mse_loss_computation(self):
        head = ExpressionPredictionHead(hidden_dim=64)
        hidden = torch.randn(2, 10, 64)
        preds = head(hidden)
        targets = torch.rand(2, 10)
        loss = torch.nn.MSELoss()(preds, targets)
        assert loss.shape == ()
        assert loss.item() > 0
