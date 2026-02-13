"""Distributed shard-aware sampler for multi-GPU training.

:class:`DistributedShardSampler` assigns whole shards to DDP/FSDP
ranks so each rank reads from a disjoint set of memory-mapped files,
avoiding contention on shared file handles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scmodelforge.data.memmap_store import MemoryMappedStore


class DistributedShardSampler(Sampler[int]):
    """Shard-aware distributed sampler.

    Assigns whole shards to DDP ranks, then yields cell indices
    from those shards. This minimises I/O contention because each
    rank only reads its own subset of memory-mapped files.

    Parameters
    ----------
    store
        :class:`MemoryMappedStore` instance.
    num_replicas
        Number of DDP processes. ``None`` auto-detects.
    rank
        This process's rank. ``None`` auto-detects.
    shuffle
        Whether to shuffle shard and cell order each epoch.
    seed
        Random seed for reproducible shuffling.
    drop_last
        Whether to drop trailing cells so all ranks get the
        same number of samples.
    """

    def __init__(
        self,
        store: MemoryMappedStore,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self._store = store
        self._num_replicas = num_replicas
        self._rank = rank
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        self._epoch = 0

        # Precompute global index ranges per shard
        self._shard_ranges: list[tuple[int, int]] = []
        offset = 0
        for size in store.shard_sizes:
            self._shard_ranges.append((offset, offset + size))
            offset += size

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed + self._epoch)

        # Assign shards to this rank
        shard_order = list(range(self._store.n_shards))
        if self._shuffle:
            rng.shuffle(shard_order)

        my_shards = shard_order[self._rank :: self._num_replicas]

        # Collect cell indices from assigned shards
        indices: list[int] = []
        for shard_idx in my_shards:
            start, end = self._shard_ranges[shard_idx]
            shard_indices = list(range(start, end))
            if self._shuffle:
                rng.shuffle(shard_indices)
            indices.extend(shard_indices)

        # Truncate to globally consistent count so all ranks have equal length
        if self._drop_last and self._num_replicas > 1:
            target = self._min_per_rank_count(shard_order)
            indices = indices[:target]

        return iter(indices)

    def __len__(self) -> int:
        shard_order = self._epoch_shard_order()
        my_shards = shard_order[self._rank :: self._num_replicas]
        total = sum(
            self._shard_ranges[s][1] - self._shard_ranges[s][0]
            for s in my_shards
        )
        if self._drop_last and self._num_replicas > 1:
            return self._min_per_rank_count(shard_order)
        return total

    def _epoch_shard_order(self) -> list[int]:
        """Deterministic shard order for the current epoch."""
        rng = np.random.default_rng(self._seed + self._epoch)
        shard_order = list(range(self._store.n_shards))
        if self._shuffle:
            rng.shuffle(shard_order)
        return shard_order

    def _min_per_rank_count(self, shard_order: list[int]) -> int:
        """Minimum per-rank cell count across all ranks.

        Each rank can independently compute this (same deterministic
        shard order) and truncate to the same target, ensuring equal
        step counts across DDP/FSDP processes.
        """
        return min(
            sum(
                self._shard_ranges[s][1] - self._shard_ranges[s][0]
                for s in shard_order[r :: self._num_replicas]
            )
            for r in range(self._num_replicas)
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling.

        Parameters
        ----------
        epoch
            Current epoch number.
        """
        self._epoch = epoch
