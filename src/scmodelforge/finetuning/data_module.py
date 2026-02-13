"""Data module for fine-tuning with task labels."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scmodelforge.config.schema import DataConfig, FinetuneConfig, TokenizerConfig
    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell

logger = logging.getLogger(__name__)


class LabelEncoder:
    """Map string labels to integer indices and back.

    Parameters
    ----------
    labels
        Sequence of raw string labels (e.g. cell type names).
        Unique values are sorted to ensure deterministic ordering.
    """

    def __init__(self, labels: Sequence[str]) -> None:
        self._classes = sorted(set(labels))
        self._label_to_idx = {label: idx for idx, label in enumerate(self._classes)}

    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(self._classes)

    @property
    def classes(self) -> list[str]:
        """Sorted list of class labels."""
        return list(self._classes)

    def encode(self, label: str) -> int:
        """Encode a string label to its integer index.

        Raises
        ------
        KeyError
            If the label was not seen during construction.
        """
        if label not in self._label_to_idx:
            msg = f"Unknown label: {label!r}. Known labels: {self._classes}"
            raise KeyError(msg)
        return self._label_to_idx[label]

    def decode(self, idx: int) -> str:
        """Decode an integer index back to its string label.

        Raises
        ------
        IndexError
            If *idx* is out of range.
        """
        if idx < 0 or idx >= len(self._classes):
            msg = f"Index {idx} out of range [0, {len(self._classes)})."
            raise IndexError(msg)
        return self._classes[idx]


class _LabeledTokenizedDataset(Dataset):
    """Wraps a TokenizedCellDataset to inject integer task labels."""

    def __init__(self, tokenized_dataset: Dataset, labels: list[int | float]) -> None:
        self._dataset = tokenized_dataset
        self._labels = labels

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> TokenizedCell:
        cell = self._dataset[idx]
        cell.metadata["task_label"] = self._labels[idx]
        return cell


def _finetune_collate(
    tokenizer: BaseTokenizer,
    cells: list[TokenizedCell],
) -> dict[str, torch.Tensor]:
    """Collate tokenized cells and extract task labels into a tensor.

    Parameters
    ----------
    tokenizer
        Tokenizer whose ``_collate`` produces the standard batch dict.
    cells
        List of tokenized cells, each with ``metadata["task_label"]``.

    Returns
    -------
    dict[str, torch.Tensor]
        Batch dict from the tokenizer, plus a ``task_labels`` tensor.
    """
    # Extract task labels before collation
    raw_labels = [c.metadata["task_label"] for c in cells]

    # Collate via tokenizer
    batch = tokenizer._collate(cells)

    # Determine label dtype
    if isinstance(raw_labels[0], float):
        task_labels = torch.tensor(raw_labels, dtype=torch.float)
    else:
        task_labels = torch.tensor(raw_labels, dtype=torch.long)

    batch["task_labels"] = task_labels
    return batch


class FineTuneDataModule:
    """Data module for fine-tuning with task labels.

    Handles data loading, label encoding, stratified splitting, and
    DataLoader construction.  Does **not** apply masking (fine-tuning uses
    the full unmasked input).

    Parameters
    ----------
    data_config
        Data loading and preprocessing configuration.
    tokenizer_config
        Tokenizer strategy configuration.
    finetune_config
        Fine-tuning configuration (label_key, head, etc.).
    training_batch_size
        Batch size for train and val DataLoaders.
    num_workers
        DataLoader worker count.
    val_split
        Fraction of data reserved for validation.
    seed
        Random seed for reproducible splits.
    adata
        Optional pre-loaded AnnData (skips file loading).
    """

    def __init__(
        self,
        data_config: DataConfig,
        tokenizer_config: TokenizerConfig,
        finetune_config: FinetuneConfig,
        training_batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 42,
        adata: Any | None = None,
    ) -> None:
        self._data_config = data_config
        self._tokenizer_config = tokenizer_config
        self._finetune_config = finetune_config
        self._batch_size = training_batch_size
        self._num_workers = num_workers
        self._val_split = val_split
        self._seed = seed
        self._adata = adata

        self._gene_vocab: GeneVocab | None = None
        self._tokenizer: BaseTokenizer | None = None
        self._label_encoder: LabelEncoder | None = None
        self._train_dataset: _LabeledTokenizedDataset | None = None
        self._val_dataset: _LabeledTokenizedDataset | None = None
        self._is_setup = False

    @property
    def gene_vocab(self) -> GeneVocab:
        """Gene vocabulary (available after :meth:`setup`)."""
        if self._gene_vocab is None:
            msg = "Call setup() before accessing gene_vocab."
            raise RuntimeError(msg)
        return self._gene_vocab

    @property
    def tokenizer(self) -> BaseTokenizer:
        """Tokenizer instance (available after :meth:`setup`)."""
        if self._tokenizer is None:
            msg = "Call setup() before accessing tokenizer."
            raise RuntimeError(msg)
        return self._tokenizer

    @property
    def label_encoder(self) -> LabelEncoder | None:
        """Label encoder for classification tasks (``None`` for regression)."""
        return self._label_encoder

    def setup(self, stage: str | None = None) -> None:
        """Load data, encode labels, split into train/val.

        Idempotent — calling multiple times is safe.
        """
        if self._is_setup:
            return

        import numpy as np

        from scmodelforge.data._utils import load_adata
        from scmodelforge.data.dataset import CellDataset
        from scmodelforge.data.gene_vocab import GeneVocab
        from scmodelforge.data.preprocessing import PreprocessingPipeline
        from scmodelforge.tokenizers.registry import get_tokenizer
        from scmodelforge.training.data_module import TokenizedCellDataset

        ft_cfg = self._finetune_config
        label_key = ft_cfg.label_key
        task = ft_cfg.head.task.lower()

        # 1. Load AnnData
        adata = load_adata(self._data_config, adata=self._adata, obs_keys=[label_key])

        # 2. Build GeneVocab
        self._gene_vocab = GeneVocab.from_adata(adata)

        # 3. Preprocessing
        pp_cfg = self._data_config.preprocessing
        preprocessing = PreprocessingPipeline(
            normalize=pp_cfg.normalize,
            target_sum=pp_cfg.target_sum,
            log1p=pp_cfg.log1p,
        )

        # 4. Build CellDataset with obs_keys to pass through metadata
        full_dataset = CellDataset(
            adata, self._gene_vocab, preprocessing, obs_keys=[label_key],
        )

        # 5. Build tokenizer (no masking)
        tok_cfg = self._tokenizer_config
        from scmodelforge.tokenizers._utils import build_tokenizer_kwargs

        self._tokenizer = get_tokenizer(tok_cfg.strategy, **build_tokenizer_kwargs(tok_cfg, self._gene_vocab))

        # 6. Build labels
        raw_labels = list(adata.obs[label_key])
        if task == "classification":
            self._label_encoder = LabelEncoder(raw_labels)
            int_labels = [self._label_encoder.encode(lbl) for lbl in raw_labels]
        else:
            # Regression: labels are floats
            int_labels = [float(lbl) for lbl in raw_labels]

        # 7. Stratified train/val split
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * self._val_split))

        rng = np.random.default_rng(self._seed)
        if task == "classification":
            # Stratified split
            indices = np.arange(n_total)
            label_arr = np.array(int_labels)
            train_indices: list[int] = []
            val_indices: list[int] = []
            for cls_id in range(self._label_encoder.n_classes):  # type: ignore[union-attr]
                cls_idx = indices[label_arr == cls_id]
                rng.shuffle(cls_idx)
                n_cls_val = max(1, int(len(cls_idx) * self._val_split))
                val_indices.extend(cls_idx[:n_cls_val].tolist())
                train_indices.extend(cls_idx[n_cls_val:].tolist())
        else:
            indices = np.arange(n_total)
            rng.shuffle(indices)
            val_indices = indices[:n_val].tolist()
            train_indices = indices[n_val:].tolist()

        # 8. Wrap subsets with TokenizedCellDataset (no masking)
        from torch.utils.data import Subset

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_tokenized = TokenizedCellDataset(train_subset, self._tokenizer, masking=None)
        val_tokenized = TokenizedCellDataset(val_subset, self._tokenizer, masking=None)

        # 9. Wrap with label injection
        train_labels = [int_labels[i] for i in train_indices]
        val_labels = [int_labels[i] for i in val_indices]

        self._train_dataset = _LabeledTokenizedDataset(train_tokenized, train_labels)
        self._val_dataset = _LabeledTokenizedDataset(val_tokenized, val_labels)

        self._is_setup = True
        logger.info(
            "FineTuneDataModule setup: %d train, %d val cells (task=%s)",
            len(train_indices), len(val_indices), task,
        )

    def train_dataloader(self) -> DataLoader:
        """Training DataLoader with shuffle."""
        if self._train_dataset is None:
            msg = "Call setup() before train_dataloader()."
            raise RuntimeError(msg)
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            collate_fn=partial(_finetune_collate, self._tokenizer),
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader without shuffle."""
        if self._val_dataset is None:
            msg = "Call setup() before val_dataloader()."
            raise RuntimeError(msg)
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=partial(_finetune_collate, self._tokenizer),
            pin_memory=True,
            drop_last=False,
        )
