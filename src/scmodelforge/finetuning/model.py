"""FineTuneModel: backbone + task head composition."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from scmodelforge.models.protocol import ModelOutput


class FineTuneModel(nn.Module):
    """Wraps a pretrained backbone with a task-specific head.

    ``forward()`` calls ``backbone.encode()`` to obtain cell embeddings,
    passes them through the task head, and optionally computes the loss.

    Parameters
    ----------
    backbone
        Pretrained model with an ``encode()`` method returning ``(B, H)``
        embeddings.
    head
        Task-specific head (classification or regression).
    task
        Task type: ``"classification"`` or ``"regression"``.
    freeze_backbone
        If ``True``, freeze all backbone parameters on construction.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        task: str = "classification",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self._task = task

        if freeze_backbone:
            self.freeze_backbone()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Forward pass: encode then predict.

        Parameters
        ----------
        input_ids
            Token IDs of shape ``(B, S)``.
        attention_mask
            Mask of shape ``(B, S)``.
        values
            Optional expression values of shape ``(B, S)``.
        labels
            Task labels. For classification: ``(B,)`` long tensor.
            For regression: ``(B,)`` or ``(B, D)`` float tensor.

        Returns
        -------
        ModelOutput
        """
        embeddings = self.backbone.encode(
            input_ids, attention_mask, values=values, **kwargs,
        )
        logits = self.head(embeddings)

        loss = None
        if labels is not None:
            if self._task == "classification":
                loss = F.cross_entropy(logits, labels)
            elif self._task == "regression":
                pred = logits.squeeze(-1) if logits.size(-1) == 1 else logits
                loss = F.mse_loss(pred, labels.float())

        return ModelOutput(loss=loss, logits=logits, embeddings=embeddings)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract cell embeddings from the backbone.

        Parameters
        ----------
        input_ids
            Token IDs of shape ``(B, S)``.
        attention_mask
            Mask of shape ``(B, S)``.
        values
            Optional expression values.

        Returns
        -------
        torch.Tensor
            Cell embeddings of shape ``(B, H)``.
        """
        return self.backbone.encode(
            input_ids, attention_mask, values=values, **kwargs,
        )

    @property
    def has_lora(self) -> bool:
        """Whether the backbone has LoRA adapters applied."""
        from scmodelforge.finetuning.adapters import has_lora

        return has_lora(self.backbone)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters.

        When LoRA is active this is a no-op — LoRA manages its own
        trainable/frozen parameter split.
        """
        if self.has_lora:
            return
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters.

        When LoRA is active this is a no-op — LoRA manages its own
        trainable/frozen parameter split.
        """
        if self.has_lora:
            return
        for p in self.backbone.parameters():
            p.requires_grad = True

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count the number of parameters.

        Parameters
        ----------
        trainable_only
            If ``True`` (default), count only trainable parameters.

        Returns
        -------
        int
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
