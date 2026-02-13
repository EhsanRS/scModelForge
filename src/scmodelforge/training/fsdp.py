"""FSDP strategy builder for multi-GPU training with large models."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scmodelforge.config.schema import FSDPConfig

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = frozenset({
    "FULL_SHARD",
    "SHARD_GRAD_OP",
    "NO_SHARD",
    "HYBRID_SHARD",
})


def build_fsdp_strategy(fsdp_config: FSDPConfig):
    """Build a Lightning FSDPStrategy from configuration.

    Parameters
    ----------
    fsdp_config
        FSDP configuration dataclass.

    Returns
    -------
    FSDPStrategy
        Configured strategy instance for ``pl.Trainer``.

    Raises
    ------
    ValueError
        If ``sharding_strategy`` is not a recognised value.
    """
    from lightning.pytorch.strategies import FSDPStrategy
    from torch import nn
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    strategy = fsdp_config.sharding_strategy.upper()
    if strategy not in _VALID_STRATEGIES:
        msg = (
            f"Unknown sharding_strategy '{fsdp_config.sharding_strategy}'. "
            f"Choose from: {sorted(_VALID_STRATEGIES)}"
        )
        raise ValueError(msg)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=fsdp_config.min_num_params,
    )

    activation_checkpointing_policy = None
    if fsdp_config.activation_checkpointing:
        from scmodelforge.models.components.encoder_layer import ScModelForgeEncoderLayer

        activation_checkpointing_policy = {nn.TransformerEncoderLayer, ScModelForgeEncoderLayer}

    logger.info(
        "Building FSDP strategy: sharding=%s, cpu_offload=%s, "
        "activation_checkpointing=%s, min_params=%d",
        strategy,
        fsdp_config.cpu_offload,
        fsdp_config.activation_checkpointing,
        fsdp_config.min_num_params,
    )

    return FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=strategy,
        cpu_offload=fsdp_config.cpu_offload,
        activation_checkpointing_policy=activation_checkpointing_policy,
    )
