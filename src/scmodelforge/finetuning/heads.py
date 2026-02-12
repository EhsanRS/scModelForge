"""Task-specific prediction heads for fine-tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from scmodelforge.config.schema import TaskHeadConfig


class ClassificationHead(nn.Module):
    """Classification head mapping cell embeddings to class logits.

    Parameters
    ----------
    input_dim
        Dimension of input embeddings (backbone hidden dim).
    n_classes
        Number of output classes.
    hidden_dim
        Optional intermediate hidden layer dimension.  If ``None``, uses
        a direct linear projection.
    dropout
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, n_classes),
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        embeddings
            Cell embeddings of shape ``(B, H)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, n_classes)``.
        """
        return self.net(embeddings)


class RegressionHead(nn.Module):
    """Regression head mapping cell embeddings to continuous outputs.

    Parameters
    ----------
    input_dim
        Dimension of input embeddings (backbone hidden dim).
    output_dim
        Number of output dimensions.
    hidden_dim
        Optional intermediate hidden layer dimension.  If ``None``, uses
        a direct linear projection.
    dropout
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, output_dim),
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        embeddings
            Cell embeddings of shape ``(B, H)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(B, output_dim)``.
        """
        return self.net(embeddings)


def build_task_head(config: TaskHeadConfig, input_dim: int) -> nn.Module:
    """Build a task head from configuration.

    Parameters
    ----------
    config
        Task head configuration.
    input_dim
        Backbone hidden dimension.

    Returns
    -------
    nn.Module
        A :class:`ClassificationHead` or :class:`RegressionHead`.

    Raises
    ------
    ValueError
        If ``config.task`` is not ``"classification"`` or ``"regression"``.
    """
    task = config.task.lower()
    if task == "classification":
        if config.n_classes is None:
            msg = "n_classes must be set for classification tasks."
            raise ValueError(msg)
        return ClassificationHead(
            input_dim=input_dim,
            n_classes=config.n_classes,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    if task == "regression":
        return RegressionHead(
            input_dim=input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    msg = f"Unknown task type: {config.task!r}. Supported: 'classification', 'regression'."
    raise ValueError(msg)
