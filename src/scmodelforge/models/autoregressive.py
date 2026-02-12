"""scGPT-style autoregressive transformer for single-cell gene expression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from scmodelforge.models._utils import count_parameters, init_weights
from scmodelforge.models.components.attention import generate_causal_mask
from scmodelforge.models.components.embeddings import GeneExpressionEmbedding
from scmodelforge.models.components.heads import BinPredictionHead, MaskedGenePredictionHead
from scmodelforge.models.components.pooling import cls_pool, mean_pool
from scmodelforge.models.protocol import ModelOutput
from scmodelforge.models.registry import register_model

if TYPE_CHECKING:
    from scmodelforge.config.schema import ModelConfig


@register_model("autoregressive_transformer")
class AutoregressiveTransformer(nn.Module):
    """Autoregressive (causal) transformer for single-cell gene expression.

    Uses ``nn.TransformerEncoder`` with a causal mask for left-to-right
    attention. Has dual prediction heads: gene identity (cross-entropy)
    and expression bins (cross-entropy). The combined loss is a weighted
    sum of both.

    Parameters
    ----------
    vocab_size
        Gene vocabulary size (including special tokens).
    n_bins
        Number of expression bins for the bin prediction head.
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
        Activation function name for the FFN.
    use_expression_values
        Whether to use expression value projection in embeddings.
    layer_norm_eps
        Epsilon for LayerNorm layers.
    gene_loss_weight
        Weight for the gene prediction loss component.
    expression_loss_weight
        Weight for the expression bin prediction loss component.
    """

    def __init__(
        self,
        vocab_size: int,
        n_bins: int = 51,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pooling: str = "cls",
        activation: str = "gelu",
        *,
        use_expression_values: bool = True,
        layer_norm_eps: float = 1e-12,
        gene_loss_weight: float = 1.0,
        expression_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.hidden_dim = hidden_dim
        self._pooling_strategy = pooling
        self._gene_loss_weight = gene_loss_weight
        self._expression_loss_weight = expression_loss_weight

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

        self.gene_head = MaskedGenePredictionHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps,
        )
        self.expression_head = BinPredictionHead(
            hidden_dim=hidden_dim,
            n_bins=n_bins,
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
        """Forward pass with causal attention and dual prediction heads.

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
        **kwargs
            Extra batch keys. ``bin_ids`` of shape ``(B, S)`` used as targets
            for the expression bin prediction head.

        Returns
        -------
        ModelOutput
        """
        emb = self.embedding(input_ids, values=values)

        seq_len = input_ids.size(1)
        causal_mask = generate_causal_mask(seq_len, device=input_ids.device)
        padding_mask = attention_mask == 0

        hidden = self.encoder(emb, mask=causal_mask, src_key_padding_mask=padding_mask)

        embeddings = self._pool(hidden, attention_mask)

        gene_logits = self.gene_head(hidden)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            gene_loss = loss_fn(gene_logits.view(-1, self.vocab_size), labels.view(-1))
            loss = self._gene_loss_weight * gene_loss

            bin_ids = kwargs.get("bin_ids")
            if bin_ids is not None:
                bin_logits = self.expression_head(hidden)
                # Mask bin labels where gene labels are -100
                bin_labels = bin_ids.clone()
                bin_labels[labels == -100] = -100
                bin_loss = loss_fn(bin_logits.view(-1, self.n_bins), bin_labels.view(-1))
                loss = loss + self._expression_loss_weight * bin_loss

        return ModelOutput(loss=loss, logits=gene_logits, embeddings=embeddings)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract cell embeddings without masking.

        Runs the full encoder on all tokens (no causal mask) and pools
        to produce cell embeddings of shape ``(B, hidden_dim)``.

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
    def from_config(cls, config: ModelConfig) -> AutoregressiveTransformer:
        """Create an :class:`AutoregressiveTransformer` from a :class:`ModelConfig`.

        Parameters
        ----------
        config
            Model configuration. ``vocab_size`` must be set (not ``None``).

        Returns
        -------
        AutoregressiveTransformer
        """
        if config.vocab_size is None:
            msg = "ModelConfig.vocab_size must be set before constructing a model."
            raise ValueError(msg)
        return cls(
            vocab_size=config.vocab_size,
            n_bins=config.n_bins,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            pooling=config.pooling,
            activation=config.activation,
            use_expression_values=config.use_expression_values,
            gene_loss_weight=config.gene_loss_weight,
            expression_loss_weight=config.expression_loss_weight,
        )

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count the number of parameters in this model."""
        return count_parameters(self, trainable_only=trainable_only)
