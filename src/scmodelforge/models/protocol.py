"""Model output dataclass returned by all scModelForge models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ModelOutput:
    """Standard output container for all scModelForge models.

    Attributes
    ----------
    loss
        Scalar training loss (present only when ``labels`` are provided).
    logits
        Token-level predictions of shape ``(B, S, V)`` where *V* is vocab size.
    embeddings
        Cell-level embeddings of shape ``(B, H)`` from pooling.
    hidden_states
        Per-layer hidden states, each of shape ``(B, S, H)``.
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    embeddings: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
