"""Custom Lightning callbacks for training monitoring."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import lightning.pytorch as pl

if TYPE_CHECKING:
    import torch


class TrainingMetricsLogger(pl.Callback):
    """Log throughput and timing metrics during training.

    Logs ``perf/cells_per_sec``, ``perf/step_time_ms``, and
    ``perf/epoch_time_sec``.

    Parameters
    ----------
    log_every_n_steps
        How often to log step-level metrics.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self._log_every_n_steps = log_every_n_steps
        self._step_start: float = 0.0
        self._epoch_start: float = 0.0

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        self._epoch_start = time.monotonic()

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._step_start = time.monotonic()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self._log_every_n_steps != 0:
            return

        step_time = time.monotonic() - self._step_start
        step_time_ms = step_time * 1000.0

        # Estimate batch size from attention_mask
        batch_size = 0
        if isinstance(batch, dict) and "attention_mask" in batch:
            batch_size = batch["attention_mask"].shape[0]

        pl_module.log("perf/step_time_ms", step_time_ms, sync_dist=False)
        if step_time > 0 and batch_size > 0:
            cells_per_sec = batch_size / step_time
            pl_module.log("perf/cells_per_sec", cells_per_sec, sync_dist=False)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        epoch_time = time.monotonic() - self._epoch_start
        pl_module.log("perf/epoch_time_sec", epoch_time, sync_dist=False)


class GradientNormLogger(pl.Callback):
    """Log gradient L2 norm before the optimizer step.

    Logs ``train/grad_norm``.

    Parameters
    ----------
    log_every_n_steps
        How often to log gradient norms.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self._log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if trainer.global_step % self._log_every_n_steps != 0:
            return

        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        pl_module.log("train/grad_norm", total_norm, sync_dist=False)
