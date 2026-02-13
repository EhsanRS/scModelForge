"""PyTorch Lightning module for fine-tuning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import lightning.pytorch as pl

from scmodelforge.training.optimizers import build_scheduler

if TYPE_CHECKING:
    import torch

    from scmodelforge.config.schema import OptimizerConfig, SchedulerConfig
    from scmodelforge.models.protocol import ModelOutput

logger = logging.getLogger(__name__)


class FineTuneLightningModule(pl.LightningModule):
    """Lightning module for fine-tuning a pretrained backbone.

    Supports discriminative learning rates for backbone vs head,
    gradual unfreezing, and task-specific metrics (accuracy for
    classification).

    Parameters
    ----------
    model
        A :class:`FineTuneModel` wrapping backbone + head.
    optimizer_config
        Optimizer configuration (name, lr, weight_decay).
    scheduler_config
        Optional scheduler configuration.
    task
        Task type: ``"classification"`` or ``"regression"``.
    backbone_lr
        Learning rate for backbone param groups.  ``None`` uses the
        global LR from *optimizer_config*.
    head_lr
        Learning rate for head param groups.  ``None`` uses the
        global LR from *optimizer_config*.
    freeze_backbone_epochs
        Unfreeze the backbone after this many epochs.  ``0`` means
        no scheduled unfreezing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None = None,
        task: str = "classification",
        backbone_lr: float | None = None,
        head_lr: float | None = None,
        freeze_backbone_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config
        self._task = task
        self._backbone_lr = backbone_lr
        self._head_lr = head_lr
        self._freeze_epochs = freeze_backbone_epochs
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        """Forward pass through the fine-tune model."""
        standard_keys = {"input_ids", "attention_mask", "values", "task_labels"}
        extra = {k: v for k, v in batch.items() if k not in standard_keys}
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            values=batch.get("values"),
            labels=batch.get("task_labels"),
            **extra,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Single training step with task-specific logging."""
        output = self.forward(batch)
        self.log("train/loss", output.loss, prog_bar=True, sync_dist=True)

        if self._task == "classification" and "task_labels" in batch:
            preds = output.logits.argmax(dim=-1)
            acc = (preds == batch["task_labels"]).float().mean()
            self.log("train/accuracy", acc, sync_dist=True)

        return output.loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Single validation step with task-specific logging."""
        output = self.forward(batch)
        self.log("val/loss", output.loss, prog_bar=True, sync_dist=True)

        if self._task == "classification" and "task_labels" in batch:
            preds = output.logits.argmax(dim=-1)
            acc = (preds == batch["task_labels"]).float().mean()
            self.log("val/accuracy", acc, sync_dist=True)

    def on_train_epoch_start(self) -> None:
        """Handle gradual unfreezing at the configured epoch."""
        if self._freeze_epochs > 0 and self.current_epoch == self._freeze_epochs:
            logger.info("Unfreezing backbone at epoch %d", self.current_epoch)
            self.model.unfreeze_backbone()

    def configure_optimizers(self) -> dict[str, Any]:
        """Build optimizer with discriminative LRs for backbone and head."""
        import torch.optim

        cfg = self._optimizer_config
        backbone_lr = self._backbone_lr if self._backbone_lr is not None else cfg.lr
        head_lr = self._head_lr if self._head_lr is not None else cfg.lr

        # Separate backbone and head parameters with weight decay grouping.
        # Backbone params are ALWAYS included (even when frozen) so that
        # gradual unfreezing works correctly — the optimizer already has the
        # params when requires_grad is later set to True.  PyTorch optimizers
        # safely skip params with no .grad attribute (i.e. frozen params).
        backbone_decay = []
        backbone_no_decay = []
        head_decay = []
        head_no_decay = []

        no_decay_keywords = ("bias", "layer_norm", "LayerNorm", "layernorm")

        for name, param in self.model.backbone.named_parameters():
            if any(kw in name for kw in no_decay_keywords):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

        for name, param in self.model.head.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in no_decay_keywords):
                head_no_decay.append(param)
            else:
                head_decay.append(param)

        param_groups = [
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": cfg.weight_decay},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_lr, "weight_decay": cfg.weight_decay},
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ]

        # Remove empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        name = cfg.name.lower()
        if name == "adamw":
            optimizer = torch.optim.AdamW(param_groups)
        elif name == "adam":
            optimizer = torch.optim.Adam(param_groups)
        else:
            msg = f"Unknown optimizer: {cfg.name!r}. Supported: 'adamw', 'adam'."
            raise ValueError(msg)

        result: dict[str, Any] = {"optimizer": optimizer}
        if self._scheduler_config is not None:
            lr_scheduler = build_scheduler(optimizer, self._scheduler_config)
            result["lr_scheduler"] = lr_scheduler
        return result
