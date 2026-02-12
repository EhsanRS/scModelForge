"""Reference model architectures for single-cell foundation models."""

from scmodelforge.models.autoregressive import AutoregressiveTransformer
from scmodelforge.models.components import (
    BinPredictionHead,
    ExpressionPredictionHead,
    GeneExpressionEmbedding,
    MaskedGenePredictionHead,
    cls_pool,
    generate_causal_mask,
    mean_pool,
)
from scmodelforge.models.masked_autoencoder import MaskedAutoencoder
from scmodelforge.models.protocol import ModelOutput
from scmodelforge.models.registry import get_model, list_models, register_model
from scmodelforge.models.transformer_encoder import TransformerEncoder

__all__ = [
    "AutoregressiveTransformer",
    "BinPredictionHead",
    "ExpressionPredictionHead",
    "GeneExpressionEmbedding",
    "MaskedAutoencoder",
    "MaskedGenePredictionHead",
    "ModelOutput",
    "TransformerEncoder",
    "cls_pool",
    "generate_causal_mask",
    "get_model",
    "list_models",
    "mean_pool",
    "register_model",
]
