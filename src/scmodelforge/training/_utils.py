"""Training utility functions."""

from __future__ import annotations

import logging
import platform
import sys

logger = logging.getLogger(__name__)


def get_environment_info() -> dict[str, str]:
    """Collect runtime environment information.

    Returns
    -------
    dict[str, str]
        Keys include ``python``, ``platform``, ``torch``, ``lightning``,
        ``cuda_available``, ``gpu_count``, and ``gpu_name``.
    """
    import torch

    info: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "gpu_count": str(torch.cuda.device_count()),
    }

    try:
        import lightning.pytorch as pl

        info["lightning"] = pl.__version__
    except ImportError:  # pragma: no cover
        info["lightning"] = "not installed"

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        info["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        info["gpu_name"] = "N/A"

    return info


def log_training_config(config: object) -> None:
    """Log key training configuration values at INFO level.

    Parameters
    ----------
    config
        A :class:`ScModelForgeConfig` or similar nested config object.
    """
    from scmodelforge.config.schema import ScModelForgeConfig

    if not isinstance(config, ScModelForgeConfig):
        logger.info("Training config: %s", config)
        return

    logger.info("=== Training Configuration ===")
    logger.info("  Model: %s (hidden=%d, layers=%d, heads=%d)",
                config.model.architecture, config.model.hidden_dim,
                config.model.num_layers, config.model.num_heads)
    logger.info("  Tokenizer: %s (max_genes=%d)",
                config.tokenizer.strategy, config.tokenizer.max_genes)
    logger.info("  Training: batch=%d, epochs=%d, precision=%s",
                config.training.batch_size, config.training.max_epochs,
                config.training.precision)
    logger.info("  Optimizer: %s (lr=%g, wd=%g)",
                config.training.optimizer.name, config.training.optimizer.lr,
                config.training.optimizer.weight_decay)
    if config.training.scheduler is not None:
        logger.info("  Scheduler: %s (warmup=%d, total=%d)",
                    config.training.scheduler.name,
                    config.training.scheduler.warmup_steps,
                    config.training.scheduler.total_steps)
    logger.info("  Logger: %s", config.training.logger)

    env = get_environment_info()
    logger.info("=== Environment ===")
    for k, v in env.items():
        logger.info("  %s: %s", k, v)
