"""Utility functions for the fine-tuning module."""

from __future__ import annotations

import logging
from pathlib import Path
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

    Handles three formats:

    1. **HuggingFace-format directory** — a directory containing
       ``config.json`` (as created by :func:`~scmodelforge.models.hub.save_pretrained`).
    2. **Hub repo ID** — a string like ``"user/model"`` that will be
       downloaded via ``huggingface_hub.snapshot_download``.
    3. **Lightning / raw checkpoint file** — the original format. Lightning
       checkpoints nest the state dict under ``"state_dict"`` and prefix
       parameter names with ``"model."``.

    Parameters
    ----------
    model
        The backbone model to load weights into.
    checkpoint_path
        Path to checkpoint file, HF-format directory, or Hub repo ID.
    strict
        If ``True``, require an exact match between checkpoint and model
        keys.  Defaults to ``False`` (permissive loading).
    """
    import torch

    path = Path(checkpoint_path)

    # --- Case 1: HF-format directory (local) ---
    if path.is_dir() and (path / "config.json").exists():
        from scmodelforge.models.hub import _load_weights

        state_dict = _load_weights(path)
        _load_into_model(model, state_dict, strict=strict)
        logger.info("Loaded pretrained weights from HF directory %s", checkpoint_path)
        return

    # --- Case 2: Hub repo ID ---
    if not path.exists():
        from scmodelforge.models.hub import _is_hub_repo_id, _load_weights, _resolve_model_directory

        if _is_hub_repo_id(checkpoint_path):
            directory = _resolve_model_directory(checkpoint_path)
            state_dict = _load_weights(directory)
            _load_into_model(model, state_dict, strict=strict)
            logger.info("Loaded pretrained weights from Hub repo %s", checkpoint_path)
            return
        msg = f"Checkpoint path does not exist: {checkpoint_path}"
        raise FileNotFoundError(msg)

    # --- Case 3: Lightning or raw checkpoint file ---
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract state_dict from Lightning checkpoint
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip "model." prefix added by LightningModule
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key.removeprefix("model.")] = value

    _load_into_model(model, cleaned, strict=strict)
    logger.info("Loaded pretrained weights from %s", checkpoint_path)


def _load_into_model(
    model: nn.Module,
    state_dict: dict,
    *,
    strict: bool = False,
) -> None:
    """Load a state dict into a model with optional filtering."""
    if not strict:
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        missing = model_keys - set(filtered.keys())
        unexpected = set(state_dict.keys()) - model_keys
        if missing:
            logger.info("Missing keys not loaded: %s", missing)
        if unexpected:
            logger.info("Unexpected keys ignored: %s", unexpected)
        model.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)
