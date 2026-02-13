"""Reusable model components: embeddings, heads, pooling, attention."""

from scmodelforge.models.components.attention import build_encoder, build_encoder_layer, generate_causal_mask
from scmodelforge.models.components.custom_attention import (
    FlashSelfAttention,
    GeneGeneAttention,
    LinearAttention,
    build_attention,
)
from scmodelforge.models.components.embeddings import GeneExpressionEmbedding
from scmodelforge.models.components.encoder import ScModelForgeEncoder
from scmodelforge.models.components.encoder_layer import ScModelForgeEncoderLayer
from scmodelforge.models.components.heads import BinPredictionHead, ExpressionPredictionHead, MaskedGenePredictionHead
from scmodelforge.models.components.pooling import cls_pool, mean_pool

__all__ = [
    "BinPredictionHead",
    "ExpressionPredictionHead",
    "FlashSelfAttention",
    "GeneExpressionEmbedding",
    "GeneGeneAttention",
    "LinearAttention",
    "MaskedGenePredictionHead",
    "ScModelForgeEncoder",
    "ScModelForgeEncoderLayer",
    "build_attention",
    "build_encoder",
    "build_encoder_layer",
    "cls_pool",
    "generate_causal_mask",
    "mean_pool",
]
