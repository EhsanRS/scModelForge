"""Base adapter interface and data types for external pretrained models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch.nn as nn
    from anndata import AnnData

logger = logging.getLogger(__name__)


@dataclass
class ExternalModelInfo:
    """Metadata describing an external pretrained model.

    Attributes
    ----------
    name
        Short identifier (e.g. ``"geneformer"``).
    full_name
        Human-readable name (e.g. ``"Geneformer 30M"``).
    paper_url
        URL to the model's publication.
    repo_url
        URL to the model's code repository.
    hidden_dim
        Dimensionality of cell embeddings produced by the model.
    species
        List of species the model supports.
    pip_package
        Python package name required to use this model.
    n_parameters
        Approximate number of model parameters.
    gene_id_format
        Gene identifier format expected: ``"ensembl"`` or ``"symbol"``.
    supports_finetune
        Whether the adapter exposes a ``get_backbone()`` for fine-tuning.
    """

    name: str
    full_name: str = ""
    paper_url: str = ""
    repo_url: str = ""
    hidden_dim: int = 0
    species: list[str] = field(default_factory=lambda: ["human"])
    pip_package: str = ""
    n_parameters: int = 0
    gene_id_format: str = "symbol"
    supports_finetune: bool = False


@dataclass
class GeneOverlapReport:
    """Statistics for gene overlap between an AnnData object and a model's vocabulary.

    Attributes
    ----------
    matched
        Number of genes present in both the data and the model vocabulary.
    missing
        Genes in the model vocabulary but absent from the data.
    extra
        Genes in the data but absent from the model vocabulary.
    coverage
        Fraction of model vocabulary genes found in the data (``matched / model_vocab_size``).
    model_vocab_size
        Total size of the model's gene vocabulary.
    adata_n_genes
        Total number of genes in the AnnData object.
    """

    matched: int
    missing: int
    extra: int
    coverage: float
    model_vocab_size: int
    adata_n_genes: int


class BaseModelAdapter(ABC):
    """Abstract base class for external model adapters.

    Each adapter wraps a single external pretrained model and exposes a
    uniform interface to extract cell embeddings from an AnnData object.
    The adapter owns its entire data pipeline: gene alignment,
    preprocessing, tokenization, and inference.

    Parameters
    ----------
    model_name_or_path
        Model identifier — HuggingFace repo ID or local path.
    device
        Device for inference (e.g. ``"cpu"``, ``"cuda"``).
    batch_size
        Default batch size for embedding extraction.
    **kwargs
        Additional adapter-specific arguments.
    """

    def __init__(
        self,
        model_name_or_path: str = "",
        device: str = "cpu",
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._device = device
        self._batch_size = batch_size
        self._loaded = False

    @property
    @abstractmethod
    def info(self) -> ExternalModelInfo:
        """Return metadata about this model."""
        ...

    @abstractmethod
    def _require_package(self) -> None:
        """Check that required external packages are installed.

        Should raise :class:`ImportError` with install instructions if missing.
        """
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and any required artifacts (vocab, config, etc.)."""
        ...

    @abstractmethod
    def extract_embeddings(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> np.ndarray:
        """Extract cell embeddings from an AnnData object.

        Parameters
        ----------
        adata
            Annotated data matrix with cells in rows and genes in columns.
        batch_size
            Override the default batch size.
        device
            Override the default device.

        Returns
        -------
        np.ndarray
            Cell embeddings of shape ``(n_cells, hidden_dim)``.
        """
        ...

    @abstractmethod
    def _get_model_genes(self) -> list[str]:
        """Return the model's gene vocabulary as a list of gene identifiers."""
        ...

    def gene_overlap_report(self, adata: AnnData) -> GeneOverlapReport:
        """Compute gene overlap between *adata* and this model's vocabulary.

        Parameters
        ----------
        adata
            AnnData object to compare against.

        Returns
        -------
        GeneOverlapReport
        """
        from scmodelforge.zoo._utils import compute_gene_overlap

        model_genes = self._get_model_genes()
        return compute_gene_overlap(list(adata.var_names), model_genes)

    def get_backbone(self) -> nn.Module:
        """Return the model's backbone as an ``nn.Module`` for fine-tuning.

        Raises
        ------
        NotImplementedError
            If this adapter does not support fine-tuning.
        """
        raise NotImplementedError(
            f"Adapter '{self.info.name}' does not support fine-tuning via get_backbone(). "
            f"Check adapter.info.supports_finetune before calling."
        )

    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded, calling ``_require_package()`` and ``load_model()`` once."""
        if not self._loaded:
            self._require_package()
            self.load_model()
            self._loaded = True
