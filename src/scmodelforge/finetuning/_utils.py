"""Utility functions for the fine-tuning module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


def load_pretrained_backbone(
    model: nn.Module,
    checkpoint_path: str,
    *,
    strict: bool = False,
) -> None:
    """Load pretrained weights into a backbone model.

    Handles both raw ``state_dict`` files and Lightning checkpoint files
    (which nest the state dict under a ``"state_dict"`` key and prefix
    parameter names with ``"model."``).

    Parameters
    ----------
    model
        The backbone model to load weights into.
    checkpoint_path
        Path to the checkpoint file.
    strict
        If ``True``, require an exact match between checkpoint and model
        keys.  Defaults to ``False`` (permissive loading).
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract state_dict from Lightning checkpoint
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip "model." prefix added by LightningModule
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key.removeprefix("model.")] = value

    # Filter to only keys present in the model (unless strict)
    if not strict:
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in cleaned.items() if k in model_keys}
        missing = model_keys - set(filtered.keys())
        unexpected = set(cleaned.keys()) - model_keys
        if missing:
            logger.info("Missing keys not loaded: %s", missing)
        if unexpected:
            logger.info("Unexpected keys ignored: %s", unexpected)
        model.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(cleaned, strict=True)

    logger.info("Loaded pretrained weights from %s", checkpoint_path)
