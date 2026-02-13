"""scGPT adapter — Cui et al. (2024).

Generative pre-trained transformer for single-cell multi-omics.
Uses binned expression tokenization with gene symbols.

Install::

    pip install scgpt
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

# Default pretrained model directory
_DEFAULT_MODEL = "scGPT_human"


@register_external_model("scgpt")
class ScGPTAdapter(BaseModelAdapter):
    """Adapter for scGPT (Cui et al., 2024).

    scGPT is a generative pre-trained transformer for single-cell
    multi-omics using binned expression values. Gene IDs are gene symbols.

    Parameters
    ----------
    model_name_or_path
        Path to the scGPT model directory containing ``vocab.json``,
        ``args.json``, and ``best_model.pt``.
    device
        Device for inference.
    batch_size
        Default batch size.
    max_length
        Maximum input sequence length (default: 1200).
    gene_col
        Column in ``adata.var`` containing gene names (default: ``"index"``).
    **kwargs
        Passed to :class:`BaseModelAdapter`.
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 1200,
        gene_col: str = "index",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, device=device, batch_size=batch_size, **kwargs)
        self._max_length = max_length
        self._gene_col = gene_col
        self._model: Any = None
        self._vocab: Any = None
        self._model_configs: dict[str, Any] | None = None

    @property
    def info(self) -> ExternalModelInfo:
        return ExternalModelInfo(
            name="scgpt",
            full_name="scGPT",
            paper_url="https://doi.org/10.1038/s41592-024-02201-0",
            repo_url="https://github.com/bowang-lab/scGPT",
            hidden_dim=512,
            species=["human"],
            pip_package="scgpt",
            n_parameters=50_000_000,
            gene_id_format="symbol",
            supports_finetune=False,
        )

    def _require_package(self) -> None:
        from scmodelforge.zoo._utils import require_package

        require_package("scgpt", "scgpt")

    def _get_model_genes(self) -> list[str]:
        self._ensure_loaded()
        if self._vocab is None:
            return []
        return list(self._vocab.get_stoi().keys())

    def load_model(self) -> None:
        """Load the scGPT model, vocabulary, and configuration."""
        import json
        from pathlib import Path

        import torch
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab

        model_dir = Path(self._model_name_or_path)

        logger.info("Loading scGPT model from '%s'", model_dir)

        # Load vocabulary
        vocab_file = model_dir / "vocab.json"
        self._vocab = GeneVocab.from_file(vocab_file)
        special_tokens = ["<pad>", "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in self._vocab:
                self._vocab.append_token(s)

        # Load model config
        config_file = model_dir / "args.json"
        with open(config_file) as f:
            self._model_configs = json.load(f)

        self._vocab.set_default_index(self._vocab["<pad>"])

        # Build model
        self._model = TransformerModel(
            ntoken=len(self._vocab),
            d_model=self._model_configs["embsize"],
            nhead=self._model_configs["nheads"],
            d_hid=self._model_configs["d_hid"],
            nlayers=self._model_configs["nlayers"],
            nlayers_cls=self._model_configs.get("n_layers_cls", 3),
            n_cls=1,
            vocab=self._vocab,
            dropout=self._model_configs.get("dropout", 0.2),
            pad_token=self._model_configs.get("pad_token", "<pad>"),
            pad_value=self._model_configs.get("pad_value", -2),
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=False,
        )

        # Load weights
        model_file = model_dir / "best_model.pt"
        try:
            self._model.load_state_dict(torch.load(model_file, map_location=self._device))
        except RuntimeError:
            # Partial loading for mismatched state dicts
            model_dict = self._model.state_dict()
            pretrained_dict = torch.load(model_file, map_location=self._device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self._model.load_state_dict(model_dict)

        self._model.to(torch.device(self._device))
        # Set to inference mode
        self._model.train(False)

        logger.info(
            "scGPT loaded: %d gene tokens, embsize=%d, device=%s",
            len(self._vocab),
            self._model_configs["embsize"],
            self._device,
        )

    def extract_embeddings(
        self,
        adata: AnnData,
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> np.ndarray:
        """Extract cell embeddings using scGPT.

        Steps:
        1. Ensure loaded
        2. Map genes to vocabulary IDs, filter unmatched
        3. Batch forward pass with CLS token embedding
        4. L2-normalize and return
        """
        import numpy as np
        import scipy.sparse as sp
        import torch

        self._ensure_loaded()

        bs = batch_size or self._batch_size
        dev = device or self._device
        torch_device = torch.device(dev)

        logger.info(
            "scGPT: extracting embeddings for %d cells (batch_size=%d, device=%s)",
            adata.n_obs,
            bs,
            dev,
        )

        # Gene overlap
        report = self.gene_overlap_report(adata)
        logger.info(
            "Gene overlap: %d matched, %d missing (%.1f%% coverage)",
            report.matched,
            report.missing,
            report.coverage * 100,
        )

        assert self._vocab is not None  # noqa: S101
        assert self._model_configs is not None  # noqa: S101

        # Map genes to vocab IDs
        gene_col = self._gene_col
        gene_names = list(adata.var_names) if gene_col == "index" else list(adata.var[gene_col])

        gene_ids_in_vocab = np.array([self._vocab.get(g, -1) for g in gene_names])

        # Filter to genes in vocab
        valid_mask = gene_ids_in_vocab >= 0
        valid_gene_ids = gene_ids_in_vocab[valid_mask]

        X = adata.X
        embsize = self._model_configs["embsize"]
        pad_token_id = self._vocab[self._model_configs.get("pad_token", "<pad>")]
        pad_value = self._model_configs.get("pad_value", -2)
        cls_token_id = self._vocab["<cls>"]
        max_length = self._max_length

        self._model.to(torch_device)

        all_embeddings = np.zeros((adata.n_obs, embsize), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, adata.n_obs, bs):
                end = min(start + bs, adata.n_obs)
                batch_genes_list: list[list[int]] = []
                batch_values_list: list[list[float]] = []

                for cell_idx in range(start, end):
                    row = X[cell_idx]
                    row = row.toarray().ravel() if sp.issparse(row) else np.asarray(row).ravel()
                    row_valid = row[valid_mask]

                    # Non-zero genes
                    nonzero_idx = np.nonzero(row_valid)[0]
                    values = row_valid[nonzero_idx].tolist()
                    genes = valid_gene_ids[nonzero_idx].tolist()

                    # Prepend CLS token
                    genes = [cls_token_id] + genes[:max_length - 1]
                    values = [float(pad_value)] + values[:max_length - 1]

                    batch_genes_list.append(genes)
                    batch_values_list.append(values)

                # Pad batch
                max_len = max(len(g) for g in batch_genes_list)
                padded_genes = []
                padded_values = []
                for genes, values in zip(batch_genes_list, batch_values_list, strict=True):
                    pad_len = max_len - len(genes)
                    padded_genes.append(genes + [pad_token_id] * pad_len)
                    padded_values.append(values + [float(pad_value)] * pad_len)

                input_gene_ids = torch.tensor(padded_genes, dtype=torch.long, device=torch_device)
                input_values = torch.tensor(padded_values, dtype=torch.float, device=torch_device)
                src_key_padding_mask = input_gene_ids.eq(pad_token_id)

                embeddings = self._model._encode(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                )

                # CLS position embedding
                cls_emb = embeddings[:, 0, :].cpu().numpy()
                all_embeddings[start:end] = cls_emb

        # L2 normalize
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-8, a_max=None)
        all_embeddings = all_embeddings / norms

        logger.info("scGPT: extracted embeddings shape %s", all_embeddings.shape)
        return all_embeddings
