"""scFoundation adapter — Hao et al. (2024).

100M-parameter asymmetric encoder-decoder model (xTrimoGene architecture)
pretrained on 50M+ single-cell transcriptomes covering 19,264 genes.

Install::

    pip install scfoundation

Or clone from: https://github.com/biomap-research/scFoundation
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

_DEFAULT_MODEL = "genbio-ai/scFoundation"

# scFoundation works with a fixed vocabulary of 19,264 genes
_N_GENES = 19264


@register_external_model("scfoundation")
class ScFoundationAdapter(BaseModelAdapter):
    """Adapter for scFoundation (Hao et al., 2024).

    scFoundation uses an asymmetric encoder-decoder (xTrimoGene) architecture
    with auto-discretization for continuous gene expression values.
    Works with 19,264 human gene symbols.

    Parameters
    ----------
    model_name_or_path
        HuggingFace model ID or local path to model checkpoint.
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
        self._gene_list: list[str] = []
        self._config: dict[str, Any] | None = None

    @property
    def info(self) -> ExternalModelInfo:
        return ExternalModelInfo(
            name="scfoundation",
            full_name="scFoundation",
            paper_url="https://doi.org/10.1038/s41592-024-02305-7",
            repo_url="https://github.com/biomap-research/scFoundation",
            hidden_dim=768,
            species=["human"],
            pip_package="scfoundation",
            n_parameters=100_000_000,
            gene_id_format="symbol",
            supports_finetune=False,
        )

    @classmethod
    def isolation_deps(cls) -> list[str]:
        """Pip requirements for isolated environment."""
        return ["scfoundation>=0.1", "torch>=2.0"]

    def _require_package(self) -> None:
        from scmodelforge.zoo._utils import require_package

        require_package("scfoundation", "scfoundation")

    def _get_model_genes(self) -> list[str]:
        self._ensure_loaded()
        return list(self._gene_list)

    def load_model(self) -> None:
        """Load the scFoundation model and gene index."""
        import scfoundation
        import torch

        logger.info("Loading scFoundation model from '%s'", self._model_name_or_path)

        self._model, self._config = scfoundation.load_model(self._model_name_or_path)

        self._model.to(torch.device(self._device))
        # Set to inference mode
        self._model.train(False)

        # Load gene index (19,264 genes)
        if hasattr(scfoundation, "get_gene_list"):
            self._gene_list = scfoundation.get_gene_list()
        elif self._config and "gene_list" in self._config:
            self._gene_list = self._config["gene_list"]

        logger.info(
            "scFoundation loaded: %d genes, hidden_dim=768, device=%s",
            len(self._gene_list),
            self._device,
        )

    def extract_embeddings(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> np.ndarray:
        """Extract cell embeddings using scFoundation.

        Steps:
        1. Normalize and log-transform expression
        2. Align genes to model's 19,264-gene vocabulary
        3. Encode through the encoder (decoder disabled for inference)
        4. Max-pool over gene-level embeddings to get cell embeddings
        """
        import numpy as np
        import scipy.sparse as sp
        import torch

        self._ensure_loaded()

        bs = batch_size or self._batch_size
        dev = device or self._device
        torch_device = torch.device(dev)

        logger.info(
            "scFoundation: extracting embeddings for %d cells (batch_size=%d, device=%s)",
            adata.n_obs,
            bs,
            dev,
        )
        logger.info("scFoundation: normalized + log1p, gene symbols, 19264 genes")

        # Gene overlap
        if self._gene_list:
            report = self.gene_overlap_report(adata)
            logger.info(
                "Gene overlap: %d matched, %d missing (%.1f%% coverage)",
                report.matched,
                report.missing,
                report.coverage * 100,
            )

        # Align genes
        from scmodelforge.zoo._utils import align_genes_to_model

        if self._gene_list:
            adata = align_genes_to_model(adata, self._gene_list)

        # Preprocess: normalize and log1p
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Library size normalization + log1p
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, a_min=1e-8, a_max=None)
        X = X / row_sums * 10000.0
        X = np.log1p(X)

        self._model.to(torch_device)

        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for start_idx in range(0, adata.n_obs, bs):
                end_idx = min(start_idx + bs, adata.n_obs)
                batch_X = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=torch_device)

                # Forward through encoder
                outputs = self._model.encode(batch_X)

                embeddings = outputs if isinstance(outputs, torch.Tensor) else outputs.last_hidden_state

                # Max-pool over gene dimension to get cell embeddings
                if embeddings.dim() == 3:
                    embeddings = embeddings.max(dim=1).values

                all_embeddings.append(embeddings.cpu().numpy())

        result = np.concatenate(all_embeddings, axis=0)
        logger.info("scFoundation: extracted embeddings shape %s", result.shape)
        return result.astype(np.float32)
