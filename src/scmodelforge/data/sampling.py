"""Weighted cell sampling for class-imbalanced datasets."""

from __future__ import annotations

import collections
import collections.abc

import torch
from torch.utils.data import Sampler


class WeightedCellSampler(Sampler):
    """Inverse-frequency weighted sampler for class-imbalanced datasets.

    Oversamples rare cell types and undersamples common ones using
    multinomial sampling with replacement.

    Supports **curriculum learning**: linearly interpolates from uniform
    weights to full inverse-frequency weights over a warmup period.

    Parameters
    ----------
    labels
        Class label for each sample (same length as the dataset).
    replacement
        Whether to sample with replacement. ``True`` by default (standard
        for weighted sampling).
    seed
        Random seed for reproducibility.
    curriculum_warmup_epochs
        Number of epochs over which to ramp from uniform to full
        inverse-frequency weights. ``0`` disables curriculum (full
        weighting from the start).
    """

    def __init__(
        self,
        labels: list[str],
        replacement: bool = True,
        seed: int = 0,
        curriculum_warmup_epochs: int = 0,
    ) -> None:
        self._labels = labels
        self._replacement = replacement
        self._seed = seed
        self._curriculum_warmup_epochs = curriculum_warmup_epochs
        self._epoch = 0

        n = len(labels)
        if n == 0:
            msg = "labels must be non-empty"
            raise ValueError(msg)

        # Compute per-class inverse-frequency weights
        counts: dict[str, int] = dict(collections.Counter(labels))
        self._class_weights_dict = {cls: n / cnt for cls, cnt in counts.items()}

        # Per-sample weights
        self._inverse_freq_weights = torch.tensor(
            [self._class_weights_dict[lbl] for lbl in labels], dtype=torch.float64,
        )
        self._uniform_weights = torch.ones(n, dtype=torch.float64)

        # Normalise both to sum to 1 for clean interpolation
        self._inverse_freq_weights = self._inverse_freq_weights / self._inverse_freq_weights.sum()
        self._uniform_weights = self._uniform_weights / self._uniform_weights.sum()

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __iter__(self) -> collections.abc.Iterator[int]:
        """Yield sample indices drawn via multinomial weighted sampling."""
        effective = self._effective_weights()
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        indices = torch.multinomial(effective, len(self._labels), replacement=self._replacement, generator=g)
        yield from indices.tolist()

    def __len__(self) -> int:
        """Return dataset length (one full pass)."""
        return len(self._labels)

    # ------------------------------------------------------------------
    # Curriculum learning
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for curriculum interpolation.

        Parameters
        ----------
        epoch
            Current training epoch (0-indexed).
        """
        self._epoch = epoch

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_weights(self) -> dict[str, float]:
        """Raw inverse-frequency weight per class label (unnormalised)."""
        return dict(self._class_weights_dict)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _effective_weights(self) -> torch.Tensor:
        """Compute interpolated weights for the current epoch."""
        if self._curriculum_warmup_epochs <= 0:
            return self._inverse_freq_weights

        alpha = min(1.0, self._epoch / self._curriculum_warmup_epochs)
        return (1.0 - alpha) * self._uniform_weights + alpha * self._inverse_freq_weights


def extract_labels_from_dataset(dataset: object, label_key: str = "cell_type") -> list[str]:
    """Extract labels from a CellDataset or Subset for weighted sampling.

    Iterates through the underlying store to read ``metadata[label_key]``
    for each cell without triggering tokenization or preprocessing.

    Parameters
    ----------
    dataset
        A :class:`~scmodelforge.data.dataset.CellDataset` or a
        ``torch.utils.data.Subset`` wrapping one.
    label_key
        The metadata key to read labels from.

    Returns
    -------
    list[str]
        One label string per sample.
    """
    from torch.utils.data import Subset

    # Unwrap Subset to get indices and underlying dataset
    if isinstance(dataset, Subset):
        indices = dataset.indices
        base = dataset.dataset
    else:
        base = dataset
        indices = range(len(base))  # type: ignore[arg-type]

    labels: list[str] = []
    for idx in indices:
        item = base[idx]
        metadata = item.get("metadata", {})
        labels.append(metadata.get(label_key, "unknown"))
    return labels
