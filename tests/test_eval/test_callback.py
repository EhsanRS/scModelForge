"""Tests for eval.callback — AssessmentCallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scmodelforge.config.schema import EvalConfig
from scmodelforge.eval.callback import AssessmentCallback


class TestAssessmentCallback:
    """Tests for AssessmentCallback."""

    def test_init(self, tiny_adata, tiny_tokenizer):
        config = EvalConfig(every_n_epochs=2, batch_size=16, benchmarks=["linear_probe"])
        cb = AssessmentCallback(
            config=config,
            datasets={"test": tiny_adata},
            tokenizer=tiny_tokenizer,
        )
        assert cb._batch_size == 16
        assert cb._harness is None  # lazily built

    def test_batch_size_override(self, tiny_adata, tiny_tokenizer):
        config = EvalConfig(every_n_epochs=1, batch_size=256)
        cb = AssessmentCallback(
            config=config,
            datasets={"test": tiny_adata},
            tokenizer=tiny_tokenizer,
            batch_size=32,
        )
        assert cb._batch_size == 32

    def test_harness_lazy_build(self, tiny_adata, tiny_tokenizer):
        config = EvalConfig(every_n_epochs=1, benchmarks=["linear_probe"])
        cb = AssessmentCallback(
            config=config,
            datasets={"test": tiny_adata},
            tokenizer=tiny_tokenizer,
        )
        assert cb._harness is None
        harness = cb._get_harness()
        assert harness is not None
        # Second call returns same instance
        assert cb._get_harness() is harness

    def test_skips_non_matching_epochs(self, tiny_adata, tiny_tokenizer, tiny_model):
        config = EvalConfig(every_n_epochs=3, batch_size=16, benchmarks=["linear_probe"])
        cb = AssessmentCallback(
            config=config,
            datasets={"test": tiny_adata},
            tokenizer=tiny_tokenizer,
        )

        trainer = MagicMock()
        pl_module = MagicMock()
        pl_module.model = tiny_model
        pl_module.device = "cpu"

        # Epoch 1 — should skip (1 % 3 != 0)
        trainer.current_epoch = 1
        with patch.object(cb, "_get_harness") as mock_harness:
            cb.on_validation_epoch_end(trainer, pl_module)
            mock_harness.assert_not_called()

    def test_runs_on_matching_epoch(self, tiny_adata, tiny_tokenizer, tiny_model):
        config = EvalConfig(every_n_epochs=2, batch_size=16, benchmarks=["linear_probe"])
        cb = AssessmentCallback(
            config=config,
            datasets={"test": tiny_adata},
            tokenizer=tiny_tokenizer,
            device="cpu",
        )

        trainer = MagicMock()
        pl_module = MagicMock()
        pl_module.model = tiny_model
        pl_module.device = "cpu"

        # Epoch 0 — should run (0 % 2 == 0)
        trainer.current_epoch = 0
        cb.on_validation_epoch_end(trainer, pl_module)

        # Should have logged metrics
        assert pl_module.log.called
        # Check at least one call has "assessment/" prefix
        log_calls = pl_module.log.call_args_list
        keys = [call[0][0] for call in log_calls]
        assert any(k.startswith("assessment/") for k in keys)

    def test_skips_empty_datasets(self, tiny_tokenizer, tiny_model):
        config = EvalConfig(every_n_epochs=1, batch_size=16, benchmarks=["linear_probe"])
        cb = AssessmentCallback(
            config=config,
            datasets={},
            tokenizer=tiny_tokenizer,
        )

        trainer = MagicMock()
        trainer.current_epoch = 0
        pl_module = MagicMock()
        pl_module.model = tiny_model

        with patch.object(cb, "_get_harness") as mock_harness:
            cb.on_validation_epoch_end(trainer, pl_module)
            mock_harness.assert_not_called()

    def test_skips_when_every_n_epochs_zero(self, tiny_adata, tiny_tokenizer, tiny_model):
        config = EvalConfig(every_n_epochs=0, batch_size=16, benchmarks=["linear_probe"])
        cb = AssessmentCallback(
            config=config,
            datasets={"test": tiny_adata},
            tokenizer=tiny_tokenizer,
        )

        trainer = MagicMock()
        trainer.current_epoch = 0
        pl_module = MagicMock()
        pl_module.model = tiny_model

        with patch.object(cb, "_get_harness") as mock_harness:
            cb.on_validation_epoch_end(trainer, pl_module)
            mock_harness.assert_not_called()

    def test_stores_last_results(self, tiny_adata, tiny_tokenizer, tiny_model):
        config = EvalConfig(every_n_epochs=1, batch_size=16, benchmarks=["linear_probe"])
        cb = AssessmentCallback(
            config=config,
            datasets={"test": tiny_adata},
            tokenizer=tiny_tokenizer,
            device="cpu",
        )

        trainer = MagicMock()
        trainer.current_epoch = 0
        pl_module = MagicMock()
        pl_module.model = tiny_model
        pl_module.device = "cpu"

        cb.on_validation_epoch_end(trainer, pl_module)
        assert len(cb._last_results) > 0
        assert cb._last_results[0].benchmark_name == "linear_probe"

    def test_logs_correct_metric_keys(self, tiny_adata, tiny_tokenizer, tiny_model):
        config = EvalConfig(every_n_epochs=1, batch_size=16, benchmarks=["linear_probe"])
        cb = AssessmentCallback(
            config=config,
            datasets={"myds": tiny_adata},
            tokenizer=tiny_tokenizer,
            device="cpu",
        )

        trainer = MagicMock()
        trainer.current_epoch = 0
        pl_module = MagicMock()
        pl_module.model = tiny_model
        pl_module.device = "cpu"

        cb.on_validation_epoch_end(trainer, pl_module)

        log_calls = pl_module.log.call_args_list
        keys = [call[0][0] for call in log_calls]
        # Should log accuracy for linear probe
        assert "assessment/linear_probe/myds/accuracy" in keys
