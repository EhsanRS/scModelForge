"""Tests for FSDP strategy builder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scmodelforge.config.schema import FSDPConfig, TrainingConfig


class TestFSDPConfig:
    """FSDPConfig defaults and construction."""

    def test_defaults(self):
        cfg = FSDPConfig()
        assert cfg.sharding_strategy == "FULL_SHARD"
        assert cfg.cpu_offload is False
        assert cfg.activation_checkpointing is False
        assert cfg.min_num_params == 1_000_000

    def test_custom_values(self):
        cfg = FSDPConfig(
            sharding_strategy="SHARD_GRAD_OP",
            cpu_offload=True,
            activation_checkpointing=True,
            min_num_params=500_000,
        )
        assert cfg.sharding_strategy == "SHARD_GRAD_OP"
        assert cfg.cpu_offload is True
        assert cfg.activation_checkpointing is True
        assert cfg.min_num_params == 500_000


class TestBuildFsdpStrategy:
    """Tests for build_fsdp_strategy()."""

    @patch("lightning.pytorch.strategies.FSDPStrategy")
    def test_returns_strategy_instance(self, mock_cls):
        from scmodelforge.training.fsdp import build_fsdp_strategy

        mock_cls.return_value = MagicMock()
        cfg = FSDPConfig()
        result = build_fsdp_strategy(cfg)
        assert result is mock_cls.return_value
        mock_cls.assert_called_once()

    @patch("lightning.pytorch.strategies.FSDPStrategy")
    def test_full_shard(self, mock_cls):
        from scmodelforge.training.fsdp import build_fsdp_strategy

        mock_cls.return_value = MagicMock()
        build_fsdp_strategy(FSDPConfig(sharding_strategy="FULL_SHARD"))
        _, kwargs = mock_cls.call_args
        assert kwargs["sharding_strategy"] == "FULL_SHARD"

    @patch("lightning.pytorch.strategies.FSDPStrategy")
    def test_shard_grad_op(self, mock_cls):
        from scmodelforge.training.fsdp import build_fsdp_strategy

        mock_cls.return_value = MagicMock()
        build_fsdp_strategy(FSDPConfig(sharding_strategy="SHARD_GRAD_OP"))
        _, kwargs = mock_cls.call_args
        assert kwargs["sharding_strategy"] == "SHARD_GRAD_OP"

    @patch("lightning.pytorch.strategies.FSDPStrategy")
    def test_cpu_offload_propagates(self, mock_cls):
        from scmodelforge.training.fsdp import build_fsdp_strategy

        mock_cls.return_value = MagicMock()
        build_fsdp_strategy(FSDPConfig(cpu_offload=True))
        _, kwargs = mock_cls.call_args
        assert kwargs["cpu_offload"] is True

    @patch("lightning.pytorch.strategies.FSDPStrategy")
    def test_activation_checkpointing_on(self, mock_cls):
        from torch import nn

        from scmodelforge.models.components.encoder_layer import ScModelForgeEncoderLayer
        from scmodelforge.training.fsdp import build_fsdp_strategy

        mock_cls.return_value = MagicMock()
        build_fsdp_strategy(FSDPConfig(activation_checkpointing=True))
        _, kwargs = mock_cls.call_args
        assert kwargs["activation_checkpointing_policy"] == {nn.TransformerEncoderLayer, ScModelForgeEncoderLayer}

    @patch("lightning.pytorch.strategies.FSDPStrategy")
    def test_activation_checkpointing_off(self, mock_cls):
        from scmodelforge.training.fsdp import build_fsdp_strategy

        mock_cls.return_value = MagicMock()
        build_fsdp_strategy(FSDPConfig(activation_checkpointing=False))
        _, kwargs = mock_cls.call_args
        assert kwargs["activation_checkpointing_policy"] is None

    def test_invalid_strategy_raises(self):
        from scmodelforge.training.fsdp import build_fsdp_strategy

        with pytest.raises(ValueError, match="Unknown sharding_strategy"):
            build_fsdp_strategy(FSDPConfig(sharding_strategy="INVALID"))

    @patch("lightning.pytorch.strategies.FSDPStrategy")
    def test_case_insensitive_strategy(self, mock_cls):
        from scmodelforge.training.fsdp import build_fsdp_strategy

        mock_cls.return_value = MagicMock()
        build_fsdp_strategy(FSDPConfig(sharding_strategy="full_shard"))
        _, kwargs = mock_cls.call_args
        assert kwargs["sharding_strategy"] == "FULL_SHARD"


class TestTrainingConfigFSDP:
    """TrainingConfig FSDP integration."""

    def test_fsdp_none_by_default(self):
        cfg = TrainingConfig()
        assert cfg.fsdp is None

    def test_fsdp_nested(self):
        cfg = TrainingConfig(fsdp=FSDPConfig(sharding_strategy="NO_SHARD"))
        assert cfg.fsdp is not None
        assert cfg.fsdp.sharding_strategy == "NO_SHARD"
