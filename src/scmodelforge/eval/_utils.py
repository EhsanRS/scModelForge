"""Evaluation utilities — embedding extraction from a model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from anndata import AnnData

    from scmodelforge.tokenizers.base import BaseTokenizer

logger = logging.getLogger(__name__)


def extract_embeddings(
    model: torch.nn.Module,
    adata: AnnData,
    tokenizer: BaseTokenizer,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Extract cell embeddings from a model.

    Uses the existing data pipeline (``CellDataset`` + ``TokenizedCellDataset``)
    without masking, then calls ``model.encode()`` to get cell-level embeddings.

    Parameters
    ----------
    model
        Model with an ``encode(input_ids, attention_mask, values)`` method
        that returns a ``(B, hidden_dim)`` tensor.
    adata
        AnnData object with expression data.
    tokenizer
        Tokenizer instance for converting cells to model inputs.
    batch_size
        Batch size for inference.
    device
        Device to run inference on (e.g. ``"cpu"`` or ``"cuda"``).

    Returns
    -------
    np.ndarray
        Cell embeddings of shape ``(n_cells, hidden_dim)``.
    """
    from scmodelforge.data.dataset import CellDataset
    from scmodelforge.training.data_module import TokenizedCellDataset

    cell_dataset = CellDataset(adata, tokenizer.gene_vocab)
    tok_dataset = TokenizedCellDataset(cell_dataset, tokenizer, masking=None)
    loader = DataLoader(
        tok_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tokenizer._collate,
        drop_last=False,
    )

    was_training = model.training
    model.eval()
    model.to(device)

    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            values = batch.get("values")
            if values is not None:
                values = values.to(device)
            emb = model.encode(input_ids, attention_mask, values=values)
            all_embeddings.append(emb.cpu().numpy())

    model.train(was_training)

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info("Extracted embeddings: shape %s", embeddings.shape)
    return embeddings
