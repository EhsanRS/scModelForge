"""Optimizer and learning-rate scheduler factories."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LambdaLR

    from scmodelforge.config.schema import OptimizerConfig, SchedulerConfig


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """Create an optimizer with per-parameter weight decay groups.

    Bias and LayerNorm parameters get ``weight_decay=0.0``.
    All other trainable parameters use the configured weight decay.

    Parameters
    ----------
    model
        The model whose parameters to optimise.
    config
        Optimizer configuration (name, lr, weight_decay).

    Returns
    -------
    Optimizer

    Raises
    ------
    ValueError
        If ``config.name`` is not a supported optimizer.
    """
    import torch.optim

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "layer_norm" in name or "LayerNorm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    name = config.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=config.lr)
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=config.lr)

    msg = f"Unknown optimizer: {config.name!r}. Supported: 'adamw', 'adam'."
    raise ValueError(msg)


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> dict:
    """Create a learning-rate scheduler dict for PyTorch Lightning.

    Parameters
    ----------
    optimizer
        The optimizer to schedule.
    config
        Scheduler configuration (name, warmup_steps, total_steps).

    Returns
    -------
    dict
        Lightning scheduler dict with keys ``scheduler``, ``interval``,
        and ``frequency``.

    Raises
    ------
    ValueError
        If ``config.name`` is not a supported scheduler.
    """
    from torch.optim.lr_scheduler import LambdaLR

    name = config.name.lower()

    if name == "cosine_warmup":
        lr_lambda = _make_cosine_warmup_lambda(config.warmup_steps, config.total_steps)
    elif name == "cosine":
        lr_lambda = _make_cosine_warmup_lambda(0, config.total_steps)
    elif name == "linear":
        lr_lambda = _make_linear_warmup_decay_lambda(config.warmup_steps, config.total_steps)
    else:
        msg = f"Unknown scheduler: {config.name!r}. Supported: 'cosine_warmup', 'cosine', 'linear'."
        raise ValueError(msg)

    scheduler: LambdaLR = LambdaLR(optimizer, lr_lambda)

    return {
        "scheduler": scheduler,
        "interval": "step",
        "frequency": 1,
    }


def _make_cosine_warmup_lambda(warmup_steps: int, total_steps: int):
    """Return a lambda for linear warmup then cosine decay."""

    def _lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return _lr_lambda


def _make_linear_warmup_decay_lambda(warmup_steps: int, total_steps: int):
    """Return a lambda for linear warmup then linear decay."""

    def _lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    return _lr_lambda
