"""scFoundation-style masked autoencoder for single-cell gene expression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from scmodelforge.models._utils import count_parameters, init_weights
from scmodelforge.models.components.embeddings import GeneExpressionEmbedding
from scmodelforge.models.components.heads import ExpressionPredictionHead
from scmodelforge.models.components.pooling import cls_pool, mean_pool
from scmodelforge.models.protocol import ModelOutput
from scmodelforge.models.registry import register_model

if TYPE_CHECKING:
    from scmodelforge.config.schema import ModelConfig


@register_model("masked_autoencoder")
class MaskedAutoencoder(nn.Module):
    """Asymmetric encoder-decoder masked autoencoder for single-cell expression.

    The encoder processes only **unmasked** tokens (dense representation).
    The decoder receives encoder outputs at unmasked positions and a
    learnable mask token at masked positions, then predicts expression
    values with an MSE loss at masked positions.

    Parameters
    ----------
    vocab_size
        Gene vocabulary size (including special tokens).
    encoder_dim
        Encoder hidden dimension.
    decoder_dim
        Decoder hidden dimension. Defaults to ``encoder_dim // 2``.
    encoder_layers
        Number of encoder transformer layers.
    decoder_layers
        Number of decoder transformer layers.
    encoder_heads
        Number of encoder attention heads.
    decoder_heads
        Number of decoder attention heads. Defaults to ``encoder_heads``.
    ffn_dim
        Encoder feed-forward intermediate dimension. Defaults to ``4 * encoder_dim``.
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
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 512,
        decoder_dim: int | None = None,
        encoder_layers: int = 12,
        decoder_layers: int = 4,
        encoder_heads: int = 8,
        decoder_heads: int | None = None,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pooling: str = "mean",
        activation: str = "gelu",
        *,
        use_expression_values: bool = True,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self._pooling_strategy = pooling

        if decoder_dim is None:
            decoder_dim = encoder_dim // 2
        self.decoder_dim = decoder_dim

        if decoder_heads is None:
            decoder_heads = encoder_heads

        if ffn_dim is None:
            ffn_dim = 4 * encoder_dim

        # --- Encoder ---
        self.embedding = GeneExpressionEmbedding(
            vocab_size=vocab_size,
            hidden_dim=encoder_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_expression_values=use_expression_values,
            layer_norm_eps=layer_norm_eps,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        # --- Decoder ---
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_position_embedding = nn.Embedding(max_seq_len, decoder_dim)

        decoder_ffn_dim = 4 * decoder_dim
        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_layers)

        self.expression_head = ExpressionPredictionHead(hidden_dim=decoder_dim)

        self.apply(init_weights)
        # Re-init mask_token after apply (since apply would not have touched it)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Forward pass with asymmetric encode-decode and MSE loss.

        Parameters
        ----------
        input_ids
            Token IDs of shape ``(B, S)``.
        attention_mask
            Mask of shape ``(B, S)`` — 1 for real tokens, 0 for padding.
        values
            Optional expression values of shape ``(B, S)``.
        labels
            Not used directly for MAE loss. Presence triggers training mode;
            the loss target comes from ``values`` at masked positions.
        **kwargs
            Extra batch keys. ``masked_positions`` of shape ``(B, S)`` (bool
            or 0/1) indicates which positions are masked. If absent, inferred
            from ``labels != -100``.

        Returns
        -------
        ModelOutput
        """
        batch_size, seq_len = input_ids.shape

        # Determine masked positions
        masked_positions = kwargs.get("masked_positions")
        if masked_positions is None and labels is not None:
            masked_positions = labels != -100
        if masked_positions is not None:
            masked_positions = masked_positions.bool()

        # Full embeddings
        emb = self.embedding(input_ids, values=values)

        # If no masking, run full encoder and return embeddings only
        if masked_positions is None or not masked_positions.any():
            padding_mask = attention_mask == 0
            hidden = self.encoder(emb, src_key_padding_mask=padding_mask)
            embeddings = self._pool(hidden, attention_mask)
            return ModelOutput(loss=None, logits=None, embeddings=embeddings)

        # --- Encoder: unmasked tokens only ---
        unmasked = ~masked_positions & (attention_mask.bool())
        # Count unmasked tokens per sample
        unmasked_counts = unmasked.sum(dim=1)  # (B,)
        max_unmasked = int(unmasked_counts.max().item())

        # Start with mask tokens everywhere for decoder input
        dec_input = self.mask_token.expand(batch_size, seq_len, -1).clone()

        # Only run encoder if there are unmasked tokens
        if max_unmasked > 0:
            # Gather unmasked embeddings into dense tensor
            enc_input = torch.zeros(batch_size, max_unmasked, self.encoder_dim, device=emb.device, dtype=emb.dtype)
            enc_padding_mask = torch.ones(batch_size, max_unmasked, dtype=torch.bool, device=emb.device)
            for i in range(batch_size):
                idx = unmasked[i].nonzero(as_tuple=True)[0]
                n = idx.size(0)
                enc_input[i, :n] = emb[i, idx]
                enc_padding_mask[i, :n] = False  # False = attend

            enc_hidden = self.encoder(enc_input, src_key_padding_mask=enc_padding_mask)

            # Project encoder outputs to decoder dim
            enc_projected = self.enc_to_dec(enc_hidden)  # (B, max_unmasked, decoder_dim)

            # Place encoder outputs at unmasked positions
            for i in range(batch_size):
                idx = unmasked[i].nonzero(as_tuple=True)[0]
                n = idx.size(0)
                dec_input[i, idx] = enc_projected[i, :n]

        # Add decoder positional embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        dec_input = dec_input + self.decoder_position_embedding(position_ids)

        # Decoder padding mask
        dec_padding_mask = attention_mask == 0
        dec_hidden = self.decoder(dec_input, src_key_padding_mask=dec_padding_mask)

        # Prediction head
        pred_values = self.expression_head(dec_hidden)  # (B, S)

        # MSE loss at masked positions only
        loss = None
        if values is not None:
            mask_flat = masked_positions.view(-1)
            pred_flat = pred_values.view(-1)
            target_flat = values.view(-1)
            if mask_flat.any():
                loss = nn.functional.mse_loss(pred_flat[mask_flat], target_flat[mask_flat])

        embeddings = self._pool(
            # For embeddings during training, use encoder on full sequence
            self.encoder(emb, src_key_padding_mask=(attention_mask == 0)),
            attention_mask,
        )

        return ModelOutput(loss=loss, logits=pred_values, embeddings=embeddings)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract cell embeddings using the full encoder.

        Processes all tokens (no masking) and pools to produce
        cell embeddings of shape ``(B, encoder_dim)``.

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
            Cell embeddings of shape ``(B, encoder_dim)``.
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
    def from_config(cls, config: ModelConfig) -> MaskedAutoencoder:
        """Create a :class:`MaskedAutoencoder` from a :class:`ModelConfig`.

        Parameters
        ----------
        config
            Model configuration. ``vocab_size`` must be set (not ``None``).

        Returns
        -------
        MaskedAutoencoder
        """
        if config.vocab_size is None:
            msg = "ModelConfig.vocab_size must be set before constructing a model."
            raise ValueError(msg)
        return cls(
            vocab_size=config.vocab_size,
            encoder_dim=config.hidden_dim,
            decoder_dim=config.decoder_dim,
            encoder_layers=config.num_layers,
            decoder_layers=config.decoder_layers,
            encoder_heads=config.num_heads,
            decoder_heads=config.decoder_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            pooling=config.pooling,
            activation=config.activation,
            use_expression_values=config.use_expression_values,
        )

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count the number of parameters in this model."""
        return count_parameters(self, trainable_only=trainable_only)
