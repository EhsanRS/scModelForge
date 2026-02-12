"""Tests for training callbacks."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from scmodelforge.training.callbacks import GradientNormLogger, TrainingMetricsLogger


class TestTrainingMetricsLogger:
    """Tests for TrainingMetricsLogger."""

    def test_init(self) -> None:
        cb = TrainingMetricsLogger(log_every_n_steps=10)
        assert cb._log_every_n_steps == 10

    def test_epoch_start_records_time(self) -> None:
        cb = TrainingMetricsLogger()
        trainer = MagicMock()
        pl_module = MagicMock()
        cb.on_train_epoch_start(trainer, pl_module)
        assert cb._epoch_start > 0

    def test_batch_start_records_time(self) -> None:
        cb = TrainingMetricsLogger()
        trainer = MagicMock()
        pl_module = MagicMock()
        cb.on_train_batch_start(trainer, pl_module, batch={}, batch_idx=0)
        assert cb._step_start > 0

    def test_batch_end_logs_on_interval(self) -> None:
        cb = TrainingMetricsLogger(log_every_n_steps=1)
        trainer = MagicMock()
        trainer.global_step = 0  # 0 % 1 == 0
        pl_module = MagicMock()
        batch = {"attention_mask": torch.ones(4, 10)}

        cb.on_train_batch_start(trainer, pl_module, batch=batch, batch_idx=0)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=batch, batch_idx=0)
        pl_module.log.assert_called()

    def test_batch_end_skips_off_interval(self) -> None:
        cb = TrainingMetricsLogger(log_every_n_steps=10)
        trainer = MagicMock()
        trainer.global_step = 3  # 3 % 10 != 0
        pl_module = MagicMock()
        batch = {"attention_mask": torch.ones(4, 10)}

        cb.on_train_batch_start(trainer, pl_module, batch=batch, batch_idx=3)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=batch, batch_idx=3)
        pl_module.log.assert_not_called()

    def test_epoch_end_logs_epoch_time(self) -> None:
        cb = TrainingMetricsLogger(log_every_n_steps=1)
        trainer = MagicMock()
        pl_module = MagicMock()

        cb.on_train_epoch_start(trainer, pl_module)
        cb.on_train_epoch_end(trainer, pl_module)
        pl_module.log.assert_called_once()
        call_args = pl_module.log.call_args
        assert call_args[0][0] == "perf/epoch_time_sec"


class TestGradientNormLogger:
    """Tests for GradientNormLogger."""

    def test_init(self) -> None:
        cb = GradientNormLogger(log_every_n_steps=5)
        assert cb._log_every_n_steps == 5

    def test_logs_grad_norm_on_interval(self) -> None:
        cb = GradientNormLogger(log_every_n_steps=1)
        trainer = MagicMock()
        trainer.global_step = 0

        # Create a simple model with gradients
        model = torch.nn.Linear(4, 2)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()

        pl_module = MagicMock()
        pl_module.parameters.return_value = model.parameters()

        optimizer = MagicMock()
        cb.on_before_optimizer_step(trainer, pl_module, optimizer)
        pl_module.log.assert_called_once()
        call_args = pl_module.log.call_args
        assert call_args[0][0] == "train/grad_norm"
        assert call_args[0][1] > 0

    def test_skips_off_interval(self) -> None:
        cb = GradientNormLogger(log_every_n_steps=10)
        trainer = MagicMock()
        trainer.global_step = 3

        pl_module = MagicMock()
        optimizer = MagicMock()
        cb.on_before_optimizer_step(trainer, pl_module, optimizer)
        pl_module.log.assert_not_called()

    def test_grad_norm_is_nonnegative(self) -> None:
        cb = GradientNormLogger(log_every_n_steps=1)
        trainer = MagicMock()
        trainer.global_step = 0

        model = torch.nn.Linear(4, 2)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()

        pl_module = MagicMock()
        pl_module.parameters.return_value = model.parameters()

        optimizer = MagicMock()
        cb.on_before_optimizer_step(trainer, pl_module, optimizer)
        logged_value = pl_module.log.call_args[0][1]
        assert logged_value >= 0.0
