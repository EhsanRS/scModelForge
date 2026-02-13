"""STACK adapter — Dong et al. (2026).

Tabular attention transformer with in-context learning for single-cell biology.
Processes cell-by-gene matrix chunks using intra-cell and inter-cell attention.

Install::

    pip install arc-stack
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

_DEFAULT_MODEL = "ArcInstitute/stack-pretrained"


@register_external_model("stack")
class StackAdapter(BaseModelAdapter):
    """Adapter for STACK (Dong et al., 2026).

    STACK is a tabular attention transformer that processes cell-by-gene
    matrix chunks using alternating intra-cell and inter-cell attention.
    Uses gene symbols, requires HVG selection.

    Parameters
    ----------
    model_name_or_path
        Path to Lightning checkpoint or model directory.
    device
        Device for inference.
    batch_size
        Number of chunks per batch (default: 8).
    sample_size
        Number of cells per chunk (default: 256).
    **kwargs
        Passed to :class:`BaseModelAdapter`.
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 8,
        sample_size: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, device=device, batch_size=batch_size, **kwargs)
        self._sample_size = sample_size
        self._model: Any = None
        self._gene_list: list[str] = []

    @property
    def info(self) -> ExternalModelInfo:
        return ExternalModelInfo(
            name="stack",
            full_name="STACK",
            paper_url="https://doi.org/10.64898/2026.01.09.698608",
            repo_url="https://github.com/ArcInstitute/stack",
            hidden_dim=100,
            species=["human"],
            pip_package="arc-stack",
            n_parameters=50_000_000,
            gene_id_format="symbol",
            supports_finetune=False,
        )

    def _require_package(self) -> None:
        from scmodelforge.zoo._utils import require_package

        require_package("stack", "arc-stack")

    def _get_model_genes(self) -> list[str]:
        self._ensure_loaded()
        return list(self._gene_list)

    def load_model(self) -> None:
        """Load the STACK model from a Lightning checkpoint."""
        import stack
        import torch

        logger.info("Loading STACK model from '%s'", self._model_name_or_path)

        self._model = stack.load_model(self._model_name_or_path)

        self._model.to(torch.device(self._device))
        # Set to inference mode
        self._model.train(False)

        # Extract HVG gene list if available
        if hasattr(self._model, "gene_list"):
            self._gene_list = list(self._model.gene_list)

        logger.info(
            "STACK loaded: %d genes, n_hidden=100, sample_size=%d, device=%s",
            len(self._gene_list),
            self._sample_size,
            self._device,
        )

    def extract_embeddings(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> np.ndarray:
        """Extract cell embeddings using STACK.

        STACK processes chunks of cells together (not individual cells).
        Cells are grouped into chunks of ``sample_size``, processed through
        tabular attention blocks, and resulting embeddings are collected.
        """
        import numpy as np
        import scipy.sparse as sp
        import torch

        self._ensure_loaded()

        bs = batch_size or self._batch_size
        dev = device or self._device
        torch_device = torch.device(dev)

        logger.info(
            "STACK: extracting embeddings for %d cells (batch_size=%d, sample_size=%d, device=%s)",
            adata.n_obs,
            bs,
            self._sample_size,
            dev,
        )
        logger.info("STACK: gene symbols, tabular attention, chunk-based processing")

        # Align genes if model has a gene list
        if self._gene_list:
            from scmodelforge.zoo._utils import align_genes_to_model

            adata = align_genes_to_model(adata, self._gene_list)

            report = self.gene_overlap_report(adata)
            logger.info(
                "Gene overlap: %d matched, %d missing (%.1f%% coverage)",
                report.matched,
                report.missing,
                report.coverage * 100,
            )

        # Prepare expression matrix
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        n_cells = X.shape[0]
        sample_size = self._sample_size

        self._model.to(torch_device)

        # Process cells in chunks of sample_size
        all_embeddings = np.zeros((n_cells, self.info.hidden_dim), dtype=np.float32)

        with torch.no_grad():
            for chunk_start in range(0, n_cells, sample_size):
                chunk_end = min(chunk_start + sample_size, n_cells)
                chunk_X = X[chunk_start:chunk_end]

                # Pad if chunk is smaller than sample_size
                actual_size = chunk_X.shape[0]
                if actual_size < sample_size:
                    pad_rows = np.zeros((sample_size - actual_size, chunk_X.shape[1]), dtype=np.float32)
                    chunk_X = np.concatenate([chunk_X, pad_rows], axis=0)

                chunk_tensor = torch.tensor(chunk_X, dtype=torch.float32, device=torch_device)
                # STACK expects (batch, n_cells, n_genes)
                chunk_tensor = chunk_tensor.unsqueeze(0)

                embeddings = self._model.encode(chunk_tensor)

                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.squeeze(0).cpu().numpy()

                # Take only actual (non-padded) cells
                all_embeddings[chunk_start:chunk_end] = embeddings[:actual_size]

        logger.info("STACK: extracted embeddings shape %s", all_embeddings.shape)
        return all_embeddings
