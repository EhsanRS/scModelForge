"""Lightning callback for in-training assessment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lightning.pytorch as pl

if TYPE_CHECKING:
    from anndata import AnnData

    from scmodelforge.config.schema import EvalConfig
    from scmodelforge.eval.base import BenchmarkResult
    from scmodelforge.tokenizers.base import BaseTokenizer

logger = logging.getLogger(__name__)


class AssessmentCallback(pl.Callback):
    """Run benchmarks during training.

    Triggered at the end of validation epochs at a configurable frequency.
    Logs results to the Lightning logger under
    ``assessment/{benchmark}/{dataset}/{metric}``.

    Parameters
    ----------
    config
        Configuration (controls frequency and benchmark list).
    datasets
        Mapping of dataset name to AnnData for assessment.
    tokenizer
        Tokenizer for embedding extraction.
    batch_size
        Batch size for embedding extraction.  Overrides ``config.batch_size``
        if provided.
    device
        Device override.  If ``None``, uses the trainer's device.
    """

    def __init__(
        self,
        config: EvalConfig,
        datasets: dict[str, AnnData],
        tokenizer: BaseTokenizer,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._datasets = datasets
        self._tokenizer = tokenizer
        self._batch_size = batch_size or config.batch_size
        self._device = device
        self._harness = None  # lazily built
        self._last_results: list[BenchmarkResult] = []

    def _get_harness(self):
        """Lazily build the harness."""
        if self._harness is None:
            from scmodelforge.eval.harness import EvalHarness

            self._harness = EvalHarness.from_config(self._config)
        return self._harness

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run benchmarks at the configured epoch interval."""
        epoch = trainer.current_epoch
        every_n = self._config.every_n_epochs
        if every_n <= 0 or epoch % every_n != 0:
            return

        if not self._datasets:
            return

        logger.info("AssessmentCallback: running benchmarks at epoch %d", epoch)

        harness = self._get_harness()
        device = self._device or str(pl_module.device)
        model = pl_module.model  # type: ignore[attr-defined]

        results = harness.run(
            model,
            self._datasets,
            self._tokenizer,
            batch_size=self._batch_size,
            device=device,
        )

        # Log metrics
        for result in results:
            for metric_name, value in result.metrics.items():
                key = f"assessment/{result.benchmark_name}/{result.dataset_name}/{metric_name}"
                pl_module.log(key, value, sync_dist=True)

        self._last_results = results
