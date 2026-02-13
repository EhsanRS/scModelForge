"""scPRINT adapter — Kalfon et al. (2024).

Large-scale pretrained model for single-cell transcriptomics using
protein language model (pLLM) gene embeddings.

Install::

    pip install scprint
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from scmodelforge.zoo.base import BaseModelAdapter, ExternalModelInfo
from scmodelforge.zoo.registry import register_external_model

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "jkobject/scPRINT"


@register_external_model("scprint")
class ScPRINTAdapter(BaseModelAdapter):
    """Adapter for scPRINT (Kalfon et al., 2024).

    scPRINT is a large-scale pretrained model using pLLM gene embeddings
    from ESM2. Works with Ensembl IDs or gene symbols.

    Parameters
    ----------
    model_name_or_path
        HuggingFace repo ID or local path (default: ``"jkobject/scPRINT"``).
    device
        Device for inference.
    batch_size
        Default batch size.
    **kwargs
        Passed to :class:`BaseModelAdapter`.
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, device=device, batch_size=batch_size, **kwargs)
        self._model: Any = None
        self._gene_names: list[str] = []

    @property
    def info(self) -> ExternalModelInfo:
        return ExternalModelInfo(
            name="scprint",
            full_name="scPRINT",
            paper_url="https://doi.org/10.1101/2024.07.29.605556",
            repo_url="https://github.com/cantinilab/scPRINT",
            hidden_dim=512,
            species=["human"],
            pip_package="scprint",
            n_parameters=130_000_000,
            gene_id_format="symbol",
            supports_finetune=False,
        )

    @classmethod
    def isolation_deps(cls) -> list[str]:
        """Pip requirements for isolated environment."""
        return ["scprint>=0.2", "torch>=2.0"]

    def _require_package(self) -> None:
        from scmodelforge.zoo._utils import require_package

        require_package("scprint", "scprint")

    def _get_model_genes(self) -> list[str]:
        self._ensure_loaded()
        return list(self._gene_names)

    def load_model(self) -> None:
        """Load the scPRINT model from HuggingFace or local path."""
        import torch
        from scprint.model import scPrint

        logger.info("Loading scPRINT model from '%s'", self._model_name_or_path)

        self._model = scPrint.from_pretrained(self._model_name_or_path)

        self._model.to(torch.device(self._device))
        # Set to inference mode
        self._model.train(False)

        # Extract gene vocabulary from model
        if hasattr(self._model, "genes"):
            self._gene_names = list(self._model.genes)

        logger.info(
            "scPRINT loaded: %d gene tokens, device=%s",
            len(self._gene_names),
            self._device,
        )

    def extract_embeddings(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> np.ndarray:
        """Extract cell embeddings using scPRINT.

        Uses the scPRINT Embedder task for batch embedding extraction.
        """
        import numpy as np
        import torch
        from scprint.tasks import Embedder

        self._ensure_loaded()

        bs = batch_size or self._batch_size
        dev = device or self._device

        logger.info(
            "scPRINT: extracting embeddings for %d cells (batch_size=%d, device=%s)",
            adata.n_obs,
            bs,
            dev,
        )
        logger.info("scPRINT: raw counts, pLLM gene embeddings")

        # Gene overlap
        if self._gene_names:
            report = self.gene_overlap_report(adata)
            logger.info(
                "Gene overlap: %d matched, %d missing (%.1f%% coverage)",
                report.matched,
                report.missing,
                report.coverage * 100,
            )

        # Use scPRINT's Embedder task
        embedder = Embedder(
            batch_size=bs,
            how="random expr",
        )

        embeddings = embedder.embed(self._model, adata)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        elif not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        logger.info("scPRINT: extracted embeddings shape %s", embeddings.shape)
        return embeddings.astype(np.float32)
