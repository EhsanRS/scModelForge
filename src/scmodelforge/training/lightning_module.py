"""PyTorch Lightning module for scModelForge pretraining."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning.pytorch as pl
import torch
import torch.nn as nn

from scmodelforge.training.optimizers import build_optimizer, build_scheduler

if TYPE_CHECKING:
    from scmodelforge.config.schema import OptimizerConfig, SchedulerConfig
    from scmodelforge.models.protocol import ModelOutput


class ScModelForgeLightningModule(pl.LightningModule):
    """Lightning module wrapping a scModelForge model for pretraining.

    Parameters
    ----------
    model
        The neural network model (e.g. :class:`TransformerEncoder`).
    optimizer_config
        Optimizer configuration.
    scheduler_config
        Optional scheduler configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        """Forward pass through the model.

        Parameters
        ----------
        batch
            Dict with keys ``input_ids``, ``attention_mask``, and optionally
            ``values``, ``labels``, plus any extra keys (e.g. ``bin_ids``,
            ``masked_positions``) forwarded to the model via ``**kwargs``.

        Returns
        -------
        ModelOutput
        """
        standard_keys = {"input_ids", "attention_mask", "values", "labels"}
        extra = {k: v for k, v in batch.items() if k not in standard_keys}
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            values=batch.get("values"),
            labels=batch.get("labels"),
            **extra,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run a single training step.

        Logs ``train/loss`` and ``train/perplexity``.
        """
        output = self.forward(batch)
        loss = output.loss
        perplexity = torch.exp(loss.clamp(max=100))

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/perplexity", perplexity, sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Run a single validation step.

        Logs ``val/loss`` and ``val/perplexity``.
        """
        output = self.forward(batch)
        loss = output.loss
        perplexity = torch.exp(loss.clamp(max=100))

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", perplexity, sync_dist=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Build optimizer and optional scheduler."""
        optimizer = build_optimizer(self.model, self._optimizer_config)

        result: dict[str, Any] = {"optimizer": optimizer}
        if self._scheduler_config is not None:
            lr_scheduler = build_scheduler(optimizer, self._scheduler_config)
            result["lr_scheduler"] = lr_scheduler
        return result
