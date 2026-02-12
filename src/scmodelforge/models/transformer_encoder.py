"""BERT-style TransformerEncoder model for single-cell gene expression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from scmodelforge.models._utils import count_parameters, init_weights
from scmodelforge.models.components.embeddings import GeneExpressionEmbedding
from scmodelforge.models.components.heads import MaskedGenePredictionHead
from scmodelforge.models.components.pooling import cls_pool, mean_pool
from scmodelforge.models.protocol import ModelOutput
from scmodelforge.models.registry import register_model

if TYPE_CHECKING:
    from scmodelforge.config.schema import ModelConfig


@register_model("transformer_encoder")
class TransformerEncoder(nn.Module):
    """BERT-style transformer encoder for single-cell gene expression.

    Parameters
    ----------
    vocab_size
        Gene vocabulary size (including special tokens).
    hidden_dim
        Hidden dimension of the transformer.
    num_layers
        Number of transformer encoder layers.
    num_heads
        Number of attention heads.
    ffn_dim
        Feed-forward intermediate dimension. Defaults to ``4 * hidden_dim``.
    dropout
        Dropout probability.
    max_seq_len
        Maximum sequence length.
    pooling
        Pooling strategy: ``"cls"`` or ``"mean"``.
    activation
        Activation function name for the FFN (e.g. ``"gelu"``).
    use_expression_values
        Whether to use expression value projection in embeddings.
    layer_norm_eps
        Epsilon for LayerNorm layers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pooling: str = "cls",
        activation: str = "gelu",
        *,
        use_expression_values: bool = True,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self._pooling_strategy = pooling

        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim

        self.embedding = GeneExpressionEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_expression_values=use_expression_values,
            layer_norm_eps=layer_norm_eps,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = MaskedGenePredictionHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps,
        )

        self.apply(init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Forward pass with optional masked gene prediction loss.

        Parameters
        ----------
        input_ids
            Token IDs of shape ``(B, S)``.
        attention_mask
            Mask of shape ``(B, S)`` — 1 for real tokens, 0 for padding.
        values
            Optional expression values of shape ``(B, S)``.
        labels
            Optional target token IDs of shape ``(B, S)`` for computing loss.
            Positions with value ``-100`` are ignored.

        Returns
        -------
        ModelOutput
        """
        emb = self.embedding(input_ids, values=values)

        # nn.TransformerEncoder expects src_key_padding_mask where True = ignore
        padding_mask = attention_mask == 0
        hidden = self.encoder(emb, src_key_padding_mask=padding_mask)

        # Pooling for cell embeddings
        embeddings = self._pool(hidden, attention_mask)

        # Prediction head
        logits = self.head(hidden)

        # Loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

        return ModelOutput(loss=loss, logits=logits, embeddings=embeddings)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract cell embeddings of shape ``(B, hidden_dim)``.

        Parameters
        ----------
        input_ids
            Token IDs of shape ``(B, S)``.
        attention_mask
            Mask of shape ``(B, S)``.
        values
            Optional expression values of shape ``(B, S)``.

        Returns
        -------
        torch.Tensor
            Cell embeddings of shape ``(B, H)``.
        """
        emb = self.embedding(input_ids, values=values)
        padding_mask = attention_mask == 0
        hidden = self.encoder(emb, src_key_padding_mask=padding_mask)
        return self._pool(hidden, attention_mask)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply the configured pooling strategy."""
        if self._pooling_strategy == "cls":
            return cls_pool(hidden, attention_mask)
        if self._pooling_strategy == "mean":
            return mean_pool(hidden, attention_mask)
        msg = f"Unknown pooling strategy: {self._pooling_strategy!r}"
        raise ValueError(msg)

    @classmethod
    def from_config(cls, config: ModelConfig) -> TransformerEncoder:
        """Create a :class:`TransformerEncoder` from a :class:`ModelConfig`.

        Parameters
        ----------
        config
            Model configuration. ``vocab_size`` must be set (not ``None``).

        Returns
        -------
        TransformerEncoder
        """
        if config.vocab_size is None:
            msg = "ModelConfig.vocab_size must be set before constructing a model."
            raise ValueError(msg)
        return cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            pooling=config.pooling,
            activation=config.activation,
            use_expression_values=config.use_expression_values,
        )

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count the number of parameters in this model.

        Parameters
        ----------
        trainable_only
            If ``True`` (default), count only trainable parameters.

        Returns
        -------
        int
        """
        return count_parameters(self, trainable_only=trainable_only)
