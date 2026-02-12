"""Pretraining prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedGenePredictionHead(nn.Module):
    """Predict masked gene tokens from hidden states.

    Architecture: ``Linear → GELU → LayerNorm → Linear(vocab_size)``.

    Parameters
    ----------
    hidden_dim
        Input hidden dimension.
    vocab_size
        Output vocabulary size (number of gene tokens).
    layer_norm_eps
        Epsilon for LayerNorm.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, layer_norm_eps: float = 1e-12) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict token logits from hidden states.

        Parameters
        ----------
        hidden_states
            Shape ``(B, S, H)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, S, V)``.
        """
        x = self.act(self.dense(hidden_states))
        x = self.layer_norm(x)
        return self.decoder(x)


class BinPredictionHead(nn.Module):
    """Predict expression bin IDs from hidden states.

    Architecture: ``Linear → GELU → LayerNorm → Linear(n_bins)``.

    Parameters
    ----------
    hidden_dim
        Input hidden dimension.
    n_bins
        Number of expression bins (output classes).
    layer_norm_eps
        Epsilon for LayerNorm.
    """

    def __init__(self, hidden_dim: int, n_bins: int, layer_norm_eps: float = 1e-12) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_dim, n_bins)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict bin logits from hidden states.

        Parameters
        ----------
        hidden_states
            Shape ``(B, S, H)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, S, n_bins)``.
        """
        x = self.act(self.dense(hidden_states))
        x = self.layer_norm(x)
        return self.decoder(x)


class ExpressionPredictionHead(nn.Module):
    """Predict continuous expression values from hidden states (MAE-style).

    Architecture: ``Linear → GELU → Linear(1)``.

    Parameters
    ----------
    hidden_dim
        Input hidden dimension.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict expression values.

        Parameters
        ----------
        hidden_states
            Shape ``(B, S, H)``.

        Returns
        -------
        torch.Tensor
            Predicted values of shape ``(B, S)``.
        """
        x = self.act(self.dense(hidden_states))
        return self.output(x).squeeze(-1)
