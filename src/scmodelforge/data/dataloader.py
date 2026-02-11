"""DataLoader wrapper with single-cell-specific defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader

from scmodelforge.data._utils import collate_cells

if TYPE_CHECKING:
    from scmodelforge.data.dataset import CellDataset


class CellDataLoader:
    """Wraps PyTorch DataLoader with single-cell-specific defaults.

    Handles variable-length gene sequences by padding to the batch
    maximum and building attention masks.

    Parameters
    ----------
    dataset
        A :class:`CellDataset` instance.
    batch_size
        Number of cells per batch.
    num_workers
        Number of data loading workers.
    shuffle
        Whether to shuffle the data each epoch.
    drop_last
        Whether to drop the last incomplete batch.
    pin_memory
        Whether to pin memory for faster GPU transfer.
    seed
        Random seed for reproducible shuffling.
    """

    def __init__(
        self,
        dataset: CellDataset,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = True,
        seed: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=collate_cells,
            generator=generator,
            persistent_workers=num_workers > 0,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._dataloader

    def __len__(self) -> int:
        return len(self._dataloader)

    def __repr__(self) -> str:
        return f"CellDataLoader(n_cells={len(self.dataset)}, batch_size={self.batch_size}, n_batches={len(self)})"
