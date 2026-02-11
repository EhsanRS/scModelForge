"""Shared type aliases and protocols used across scModelForge modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class TokenizedCellProtocol(Protocol):
    """Protocol for tokenized cell output from any tokenizer."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    metadata: dict[str, Any]


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol that all scModelForge models must implement."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass for training."""
        ...

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract cell embeddings of shape (batch_size, hidden_dim)."""
        ...
