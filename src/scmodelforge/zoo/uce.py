"""UCE adapter — Rosen et al. (2023).

Universal Cell Embeddings: a foundation model for single-cell gene expression
using ESM2 protein embeddings. Cross-species, zero-shot.

Install::

    pip install uce-model
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

_DEFAULT_MODEL = "33l_8ep_46vnorm_24heads_16bsz"


@register_external_model("uce")
class UCEAdapter(BaseModelAdapter):
    """Adapter for UCE (Rosen et al., 2023).

    UCE is a zero-shot foundation model for single-cell gene expression
    that uses ESM2 protein language model embeddings to represent genes.
    Supports cross-species analysis with no fine-tuning.

    Parameters
    ----------
    model_name_or_path
        Model name or path. Use ``"33l_8ep_46vnorm_24heads_16bsz"``
        for the 33-layer model (default).
    device
        Device for inference.
    batch_size
        Default batch size.
    model_variant
        ``"large"`` (33-layer, default) or ``"small"`` (4-layer).
    **kwargs
        Passed to :class:`BaseModelAdapter`.
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
        model_variant: str = "large",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, device=device, batch_size=batch_size, **kwargs)
        self._model_variant = model_variant
        self._model: Any = None
        self._processor: Any = None

    @property
    def info(self) -> ExternalModelInfo:
        return ExternalModelInfo(
            name="uce",
            full_name="Universal Cell Embeddings",
            paper_url="https://doi.org/10.1101/2023.11.28.568918",
            repo_url="https://github.com/snap-stanford/UCE",
            hidden_dim=1280,
            species=["human", "mouse", "zebrafish", "pig", "macaque", "frog"],
            pip_package="uce-model",
            n_parameters=650_000_000,
            gene_id_format="symbol",
            supports_finetune=False,
        )

    def _require_package(self) -> None:
        from scmodelforge.zoo._utils import require_package

        require_package("uce", "uce-model")

    def _get_model_genes(self) -> list[str]:
        # UCE uses ESM2 protein embeddings to handle any gene set;
        # it does not have a fixed gene vocabulary
        return []

    def load_model(self) -> None:
        """Load the UCE model."""
        import torch
        import uce

        logger.info("Loading UCE model (variant=%s)", self._model_variant)
        self._model = uce.get_pretrained(self._model_variant)

        self._model.to(torch.device(self._device))
        # Set to inference mode
        self._model.train(False)

        logger.info("UCE loaded: variant=%s, device=%s", self._model_variant, self._device)

    def extract_embeddings(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> np.ndarray:
        """Extract cell embeddings using UCE.

        UCE handles gene mapping internally via ESM2 protein embeddings,
        so no explicit gene alignment is needed.
        """
        import numpy as np
        import torch

        self._ensure_loaded()

        bs = batch_size or self._batch_size
        dev = device or self._device
        torch_device = torch.device(dev)

        logger.info(
            "UCE: extracting embeddings for %d cells (batch_size=%d, device=%s)",
            adata.n_obs,
            bs,
            dev,
        )
        logger.info("UCE: zero-shot mode, gene symbols, ESM2 protein embeddings")

        self._model.to(torch_device)

        # UCE accepts raw AnnData and processes internally
        # For batch processing, we chunk the adata
        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, adata.n_obs, bs):
                end = min(start + bs, adata.n_obs)
                batch_adata = adata[start:end]

                # UCE's get_cell_embeddings expects processed batch data
                embeddings = self._model.get_cell_embeddings(batch_adata)

                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()

                all_embeddings.append(embeddings)

        result = np.concatenate(all_embeddings, axis=0)
        logger.info("UCE: extracted embeddings shape %s", result.shape)
        return result.astype(np.float32)
