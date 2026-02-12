"""Tests for optimizer and scheduler factories."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scmodelforge.config.schema import OptimizerConfig, SchedulerConfig
from scmodelforge.training.optimizers import (
    _make_cosine_warmup_lambda,
    _make_linear_warmup_decay_lambda,
    build_optimizer,
    build_scheduler,
)

# ------------------------------------------------------------------
# build_optimizer
# ------------------------------------------------------------------


class TestBuildOptimizer:
    """Tests for build_optimizer."""

    def test_adamw_default(self, dummy_model: nn.Module) -> None:
        config = OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01)
        opt = build_optimizer(dummy_model, config)
        assert isinstance(opt, torch.optim.AdamW)
        assert len(opt.param_groups) == 2

    def test_adam(self, dummy_model: nn.Module) -> None:
        config = OptimizerConfig(name="adam", lr=5e-4, weight_decay=0.0)
        opt = build_optimizer(dummy_model, config)
        assert isinstance(opt, torch.optim.Adam)

    def test_weight_decay_groups(self, dummy_model: nn.Module) -> None:
        config = OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.05)
        opt = build_optimizer(dummy_model, config)
        decay_group = opt.param_groups[0]
        no_decay_group = opt.param_groups[1]
        assert decay_group["weight_decay"] == 0.05
        assert no_decay_group["weight_decay"] == 0.0

    def test_lr_is_set(self, dummy_model: nn.Module) -> None:
        config = OptimizerConfig(name="adamw", lr=3e-4, weight_decay=0.01)
        opt = build_optimizer(dummy_model, config)
        for pg in opt.param_groups:
            assert pg["lr"] == 3e-4

    def test_unknown_optimizer_raises(self, dummy_model: nn.Module) -> None:
        config = OptimizerConfig(name="sgd_nesterov", lr=1e-3, weight_decay=0.0)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(dummy_model, config)

    def test_case_insensitive(self, dummy_model: nn.Module) -> None:
        config = OptimizerConfig(name="AdamW", lr=1e-3, weight_decay=0.01)
        opt = build_optimizer(dummy_model, config)
        assert isinstance(opt, torch.optim.AdamW)


# ------------------------------------------------------------------
# build_scheduler
# ------------------------------------------------------------------


class TestBuildScheduler:
    """Tests for build_scheduler."""

    @pytest.fixture()
    def _simple_optimizer(self) -> torch.optim.Optimizer:
        model = nn.Linear(10, 10)
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def test_cosine_warmup_returns_dict(self, _simple_optimizer: torch.optim.Optimizer) -> None:
        config = SchedulerConfig(name="cosine_warmup", warmup_steps=10, total_steps=100)
        result = build_scheduler(_simple_optimizer, config)
        assert "scheduler" in result
        assert result["interval"] == "step"
        assert result["frequency"] == 1

    def test_cosine_no_warmup(self, _simple_optimizer: torch.optim.Optimizer) -> None:
        config = SchedulerConfig(name="cosine", warmup_steps=0, total_steps=100)
        result = build_scheduler(_simple_optimizer, config)
        assert "scheduler" in result

    def test_linear(self, _simple_optimizer: torch.optim.Optimizer) -> None:
        config = SchedulerConfig(name="linear", warmup_steps=5, total_steps=50)
        result = build_scheduler(_simple_optimizer, config)
        assert "scheduler" in result

    def test_unknown_scheduler_raises(self, _simple_optimizer: torch.optim.Optimizer) -> None:
        config = SchedulerConfig(name="exponential", warmup_steps=10, total_steps=100)
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(_simple_optimizer, config)


# ------------------------------------------------------------------
# LR lambda functions
# ------------------------------------------------------------------


class TestLRLambdas:
    """Tests for the learning rate schedule functions."""

    def test_cosine_warmup_starts_at_zero(self) -> None:
        fn = _make_cosine_warmup_lambda(warmup_steps=10, total_steps=100)
        assert fn(0) == pytest.approx(0.0)

    def test_cosine_warmup_ramps_up(self) -> None:
        fn = _make_cosine_warmup_lambda(warmup_steps=10, total_steps=100)
        assert fn(5) == pytest.approx(0.5)
        assert fn(10) == pytest.approx(1.0)

    def test_cosine_warmup_decays(self) -> None:
        fn = _make_cosine_warmup_lambda(warmup_steps=0, total_steps=100)
        assert fn(0) == pytest.approx(1.0)
        assert fn(50) == pytest.approx(0.5, abs=0.01)
        assert fn(100) == pytest.approx(0.0, abs=0.01)

    def test_cosine_warmup_ends_near_zero(self) -> None:
        fn = _make_cosine_warmup_lambda(warmup_steps=10, total_steps=100)
        assert fn(100) == pytest.approx(0.0, abs=0.01)

    def test_cosine_warmup_never_negative(self) -> None:
        fn = _make_cosine_warmup_lambda(warmup_steps=10, total_steps=100)
        for step in range(0, 200):
            assert fn(step) >= 0.0

    def test_linear_warmup_starts_at_zero(self) -> None:
        fn = _make_linear_warmup_decay_lambda(warmup_steps=10, total_steps=100)
        assert fn(0) == pytest.approx(0.0)

    def test_linear_warmup_ramps_then_decays(self) -> None:
        fn = _make_linear_warmup_decay_lambda(warmup_steps=10, total_steps=100)
        assert fn(10) == pytest.approx(1.0)
        assert fn(55) == pytest.approx(0.5)
        assert fn(100) == pytest.approx(0.0, abs=0.01)

    def test_linear_never_negative(self) -> None:
        fn = _make_linear_warmup_decay_lambda(warmup_steps=10, total_steps=100)
        for step in range(0, 200):
            assert fn(step) >= 0.0

    def test_zero_total_steps(self) -> None:
        fn = _make_cosine_warmup_lambda(warmup_steps=0, total_steps=0)
        assert fn(0) == 1.0
        assert fn(100) == 1.0
