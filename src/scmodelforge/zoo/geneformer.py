"""Geneformer adapter — Theodoris et al. (2023).

BERT-style model using rank-value tokenization with Ensembl gene IDs.
Expects raw counts (no normalization).

Install::

    pip install geneformer transformers
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

# Default HuggingFace model identifier
_DEFAULT_MODEL = "ctheodoris/Geneformer"


@register_external_model("geneformer")
class GeneformerAdapter(BaseModelAdapter):
    """Adapter for Geneformer (Theodoris et al., 2023).

    Geneformer is a BERT-style transformer pretrained on ~30M single-cell
    transcriptomes using rank-value encoding.  Gene IDs are Ensembl format.

    Parameters
    ----------
    model_name_or_path
        HuggingFace repo ID or local path (default: ``"ctheodoris/Geneformer"``).
    device
        Device for inference.
    batch_size
        Default batch size.
    emb_layer
        Which transformer layer to extract embeddings from (default: -1, last).
    emb_mode
        Embedding mode: ``"cell"`` for CLS-pooled cell embeddings.
    **kwargs
        Passed to :class:`BaseModelAdapter`.
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
        emb_layer: int = -1,
        emb_mode: str = "cell",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, device=device, batch_size=batch_size, **kwargs)
        self._emb_layer = emb_layer
        self._emb_mode = emb_mode
        self._model: Any = None
        self._token_dict: dict[str, int] | None = None
        self._gene_median_dict: dict[str, float] | None = None

    @property
    def info(self) -> ExternalModelInfo:
        return ExternalModelInfo(
            name="geneformer",
            full_name="Geneformer",
            paper_url="https://doi.org/10.1038/s41586-023-06139-9",
            repo_url="https://huggingface.co/ctheodoris/Geneformer",
            hidden_dim=256,
            species=["human"],
            pip_package="geneformer",
            n_parameters=30_000_000,
            gene_id_format="ensembl",
            supports_finetune=False,
        )

    @classmethod
    def isolation_deps(cls) -> list[str]:
        """Pip requirements for isolated environment."""
        return ["geneformer>=0.1", "transformers>=4.30", "torch>=2.0"]

    def _require_package(self) -> None:
        from scmodelforge.zoo._utils import require_package

        require_package("geneformer", "geneformer")
        require_package("transformers", "transformers>=4.30")

    def _get_model_genes(self) -> list[str]:
        self._ensure_loaded()
        if self._token_dict is None:
            return []
        return list(self._token_dict.keys())

    def load_model(self) -> None:
        """Load the Geneformer model and token dictionaries."""
        from transformers import BertModel

        logger.info("Loading Geneformer model from '%s'", self._model_name_or_path)
        self._model = BertModel.from_pretrained(self._model_name_or_path)

        # Load gene token dictionary from geneformer package
        from geneformer import TranscriptomeTokenizer

        tokenizer = TranscriptomeTokenizer()
        self._token_dict = tokenizer.gene_token_dict
        self._gene_median_dict = tokenizer.gene_median_dict

        logger.info(
            "Geneformer loaded: %d gene tokens, device=%s",
            len(self._token_dict) if self._token_dict else 0,
            self._device,
        )

        import torch

        self._model.to(torch.device(self._device))

    def extract_embeddings(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> np.ndarray:
        """Extract cell embeddings using Geneformer.

        Steps:
        1. Ensure raw counts
        2. Log gene overlap report
        3. Rank genes per cell by expression (descending)
        4. Map to token IDs, truncate to model's max input size
        5. Batch forward pass, collect hidden states
        6. Pool (mean over tokens), return numpy array
        """
        import numpy as np
        import scipy.sparse as sp
        import torch

        self._ensure_loaded()

        bs = batch_size or self._batch_size
        dev = device or self._device
        torch_device = torch.device(dev)

        logger.info(
            "Geneformer: extracting embeddings for %d cells (batch_size=%d, device=%s)",
            adata.n_obs,
            bs,
            dev,
        )
        logger.info("Geneformer: using raw counts, Ensembl gene IDs, rank-value tokenization")

        # Step 1: Use raw counts
        from scmodelforge.zoo._utils import ensure_raw_counts

        adata = ensure_raw_counts(adata)

        # Step 2: Gene overlap
        report = self.gene_overlap_report(adata)
        logger.info(
            "Gene overlap: %d matched, %d missing (%.1f%% coverage)",
            report.matched,
            report.missing,
            report.coverage * 100,
        )

        # Step 3-4: Tokenize cells — rank genes by expression, map to token IDs
        assert self._token_dict is not None  # noqa: S101
        assert self._gene_median_dict is not None  # noqa: S101

        model_max_len = self._model.config.max_position_embeddings
        token_dict = self._token_dict
        median_dict = self._gene_median_dict

        # Build gene->column index for adata
        gene_to_col = {g: i for i, g in enumerate(adata.var_names)}
        valid_genes = [g for g in token_dict if g in gene_to_col]

        all_token_ids: list[list[int]] = []
        X = adata.X

        for cell_idx in range(adata.n_obs):
            row = X[cell_idx]
            row = row.toarray().ravel() if sp.issparse(row) else np.asarray(row).ravel()

            # Get expression values for valid genes and normalize by median
            gene_expr = []
            for gene in valid_genes:
                col = gene_to_col[gene]
                expr = float(row[col])
                if expr > 0:
                    median = median_dict.get(gene, 1.0)
                    rank_val = expr / median if median > 0 else expr
                    gene_expr.append((gene, rank_val))

            # Sort by normalized expression descending (rank-value encoding)
            gene_expr.sort(key=lambda x: x[1], reverse=True)

            # Truncate and convert to token IDs
            token_ids = [token_dict[g] for g, _ in gene_expr[:model_max_len]]
            all_token_ids.append(token_ids)

        # Step 5-6: Batch forward pass
        self._model.to(torch_device)

        all_embeddings: list[np.ndarray] = []
        n_cells = len(all_token_ids)

        with torch.no_grad():
            for start in range(0, n_cells, bs):
                end = min(start + bs, n_cells)
                batch_tokens = all_token_ids[start:end]

                # Pad to max length in batch
                max_len = max(len(t) for t in batch_tokens) if batch_tokens else 1
                padded = []
                attention_masks = []
                for tokens in batch_tokens:
                    pad_len = max_len - len(tokens)
                    padded.append(tokens + [0] * pad_len)
                    attention_masks.append([1] * len(tokens) + [0] * pad_len)

                input_ids = torch.tensor(padded, dtype=torch.long, device=torch_device)
                attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=torch_device)

                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state  # (batch, seq, hidden)

                # Mean pool over non-padded tokens
                mask_expanded = attention_mask.unsqueeze(-1).float()
                summed = (hidden * mask_expanded).sum(dim=1)
                lengths = mask_expanded.sum(dim=1).clamp(min=1)
                pooled = summed / lengths  # (batch, hidden)

                all_embeddings.append(pooled.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info("Geneformer: extracted embeddings shape %s", embeddings.shape)
        return embeddings
