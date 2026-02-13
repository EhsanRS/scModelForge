"""Model zoo — adapters for external pretrained single-cell models."""

# Import adapter modules to trigger registration via decorators.
import scmodelforge.zoo.geneformer  # noqa: F401
import scmodelforge.zoo.scfoundation  # noqa: F401
import scmodelforge.zoo.scgpt  # noqa: F401
import scmodelforge.zoo.scprint  # noqa: F401
import scmodelforge.zoo.stack  # noqa: F401
import scmodelforge.zoo.uce  # noqa: F401
from scmodelforge.zoo.base import BaseModelAdapter, ExternalModelInfo, GeneOverlapReport
from scmodelforge.zoo.isolation import IsolatedAdapter, install_env
from scmodelforge.zoo.registry import get_external_model, list_external_models, register_external_model

__all__ = [
    "BaseModelAdapter",
    "ExternalModelInfo",
    "GeneOverlapReport",
    "IsolatedAdapter",
    "get_external_model",
    "install_env",
    "list_external_models",
    "register_external_model",
]
