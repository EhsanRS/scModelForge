"""Model zoo — adapters for external pretrained single-cell models."""

# Import adapter modules to trigger registration via decorators.
import scmodelforge.zoo.geneformer  # noqa: F401
from scmodelforge.zoo.base import BaseModelAdapter, ExternalModelInfo, GeneOverlapReport
from scmodelforge.zoo.registry import get_external_model, list_external_models, register_external_model

__all__ = [
    "BaseModelAdapter",
    "ExternalModelInfo",
    "GeneOverlapReport",
    "get_external_model",
    "list_external_models",
    "register_external_model",
]
