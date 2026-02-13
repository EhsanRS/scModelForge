"""Smoke tests for the full training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from scmodelforge.training.data_module import CellDataModule
from scmodelforge.training.pipeline import TrainingPipeline

if TYPE_CHECKING:
    from anndata import AnnData

    from scmodelforge.config.schema import (
        DataConfig,
        ScModelForgeConfig,
        TokenizerConfig,
    )


class TestTrainingPipelineSmoke:
    """End-to-end smoke tests for TrainingPipeline on CPU."""

    def test_full_pipeline_2_epochs(self, tiny_adata: AnnData, tiny_full_config: ScModelForgeConfig) -> None:
        """Train for 2 epochs on tiny synthetic data — CPU, precision=32."""
        # Inject data module with adata
        cfg = tiny_full_config

        # We need to manually set up the data module since pipeline normally loads from paths.
        # Patch the pipeline to inject our adata.
        pipeline = _PipelineWithAdata(cfg, tiny_adata)
        trainer = pipeline.run()

        assert trainer is not None
        assert trainer.current_epoch == cfg.training.max_epochs

    def test_pipeline_logs_exist(self, tiny_adata: AnnData, tiny_full_config: ScModelForgeConfig) -> None:
        """Check that CSVLogger creates log files."""
        cfg = tiny_full_config
        pipeline = _PipelineWithAdata(cfg, tiny_adata)
        pipeline.run()

        log_dir = Path(cfg.training.log_dir)
        assert log_dir.exists()

    def test_pipeline_model_has_gradients(self, tiny_adata: AnnData, tiny_full_config: ScModelForgeConfig) -> None:
        """After training, model parameters should have been updated."""
        cfg = tiny_full_config
        pipeline = _PipelineWithAdata(cfg, tiny_adata)

        # Grab model params before training
        from scmodelforge.data.gene_vocab import GeneVocab
        from scmodelforge.models.registry import get_model

        vocab = GeneVocab.from_adata(tiny_adata)
        cfg.model.vocab_size = len(vocab)
        model_before = get_model(cfg.model.architecture, cfg.model)
        initial_params = {n: p.clone() for n, p in model_before.named_parameters()}

        trainer = pipeline.run()

        # Check the model inside the lightning module was updated
        trained_model = trainer.lightning_module.model
        params_changed = False
        for n, p in trained_model.named_parameters():
            if n in initial_params and not torch.equal(p, initial_params[n]):
                params_changed = True
                break
        assert params_changed, "Model parameters should change after training"


class TestCellDataModuleIntegration:
    """Integration tests for CellDataModule with real tiny data."""

    def test_multiple_batches(
        self,
        tiny_adata: AnnData,
        tiny_data_config: DataConfig,
        tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
        )
        dm.setup()
        dl = dm.train_dataloader()
        batches = list(dl)
        assert len(batches) >= 1
        for batch in batches:
            assert batch["input_ids"].dtype == torch.long
            assert batch["attention_mask"].dtype == torch.long

    def test_val_batches_have_labels(
        self,
        tiny_adata: AnnData,
        tiny_data_config: DataConfig,
        tiny_tokenizer_config: TokenizerConfig,
    ) -> None:
        dm = CellDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
        )
        dm.setup()
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        # Both train and val get masking (D2)
        assert "labels" in batch
        assert (batch["labels"] != -100).any()


class TestTrainingPipelineUtils:
    """Test pipeline utility methods."""

    def test_build_callbacks(self, tiny_full_config: ScModelForgeConfig) -> None:
        pipeline = TrainingPipeline(tiny_full_config)
        callbacks = pipeline._build_callbacks()
        assert len(callbacks) >= 4  # checkpoint, lr_monitor, metrics, grad_norm

    def test_build_logger_csv(self, tiny_full_config: ScModelForgeConfig) -> None:
        pipeline = TrainingPipeline(tiny_full_config)
        lgr = pipeline._build_logger()
        from lightning.pytorch.loggers import CSVLogger

        assert isinstance(lgr, CSVLogger)

    def test_resolve_devices_cpu(self, tiny_full_config: ScModelForgeConfig) -> None:
        pipeline = TrainingPipeline(tiny_full_config)
        devices, strategy = pipeline._resolve_devices_and_strategy()
        if not torch.cuda.is_available():
            assert devices == 1
            assert strategy == "auto"


class TestAssessmentCallbackWiring:
    """Tests for AssessmentCallback integration in _build_callbacks."""

    def test_no_assessment_callback_when_no_benchmarks(self, tiny_full_config: ScModelForgeConfig) -> None:
        """When eval.benchmarks is empty, no AssessmentCallback is added."""
        from scmodelforge.eval.callback import AssessmentCallback

        pipeline = TrainingPipeline(tiny_full_config)
        callbacks = pipeline._build_callbacks()
        assert not any(isinstance(c, AssessmentCallback) for c in callbacks)

    def test_no_assessment_callback_when_no_data_module(self, tiny_full_config: ScModelForgeConfig) -> None:
        """Even with benchmarks configured, no callback if data_module is None."""
        from scmodelforge.eval.callback import AssessmentCallback

        tiny_full_config.eval.benchmarks = ["linear_probe"]
        pipeline = TrainingPipeline(tiny_full_config)
        callbacks = pipeline._build_callbacks(data_module=None)
        assert not any(isinstance(c, AssessmentCallback) for c in callbacks)

    def test_assessment_callback_attached_when_configured(
        self,
        tiny_adata: AnnData,
        tiny_full_config: ScModelForgeConfig,
    ) -> None:
        """AssessmentCallback is attached when benchmarks are configured and data_module is available."""
        from scmodelforge.eval.callback import AssessmentCallback

        tiny_full_config.eval.benchmarks = ["linear_probe"]
        pipeline = TrainingPipeline(tiny_full_config)

        # Build a real data module so we can pass it
        dm = CellDataModule(
            data_config=tiny_full_config.data,
            tokenizer_config=tiny_full_config.tokenizer,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
        )
        dm.setup()

        callbacks = pipeline._build_callbacks(data_module=dm)
        assessment_cbs = [c for c in callbacks if isinstance(c, AssessmentCallback)]
        assert len(assessment_cbs) == 1

    def test_data_module_exposes_adata(
        self,
        tiny_adata: AnnData,
        tiny_full_config: ScModelForgeConfig,
    ) -> None:
        """CellDataModule.adata property returns loaded AnnData after setup."""
        dm = CellDataModule(
            data_config=tiny_full_config.data,
            tokenizer_config=tiny_full_config.tokenizer,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
        )
        dm.setup()
        assert dm.adata is not None
        assert dm.adata.shape[0] == tiny_adata.shape[0]


class TestSamplerEpochCallbackWiring:
    """Tests for SamplerEpochCallback wiring in _build_callbacks."""

    def test_no_sampler_callback_without_sampler(self, tiny_full_config: ScModelForgeConfig) -> None:
        """No SamplerEpochCallback when sampler is absent."""
        from scmodelforge.training.callbacks import SamplerEpochCallback

        pipeline = TrainingPipeline(tiny_full_config)
        callbacks = pipeline._build_callbacks()
        assert not any(isinstance(c, SamplerEpochCallback) for c in callbacks)

    def test_sampler_callback_attached_when_weighted(
        self,
        tiny_adata: AnnData,
        tiny_full_config: ScModelForgeConfig,
    ) -> None:
        """SamplerEpochCallback is attached when weighted sampling is configured."""
        from scmodelforge.config.schema import SamplingConfig
        from scmodelforge.training.callbacks import SamplerEpochCallback

        tiny_full_config.training.sampling = SamplingConfig(
            strategy="weighted",
            label_key="cell_type",
            curriculum_warmup_epochs=3,
        )
        pipeline = TrainingPipeline(tiny_full_config)
        dm = CellDataModule(
            data_config=tiny_full_config.data,
            tokenizer_config=tiny_full_config.tokenizer,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            adata=tiny_adata,
            sampling_config=tiny_full_config.training.sampling,
        )
        dm.setup()

        callbacks = pipeline._build_callbacks(data_module=dm)
        sampler_cbs = [c for c in callbacks if isinstance(c, SamplerEpochCallback)]
        assert len(sampler_cbs) == 1


class TestImportPipeline:
    """Verify that the public API import works."""

    def test_import_training_pipeline(self) -> None:
        from scmodelforge.training import TrainingPipeline

        assert TrainingPipeline is not None

    def test_import_all_public_names(self) -> None:
        pass


# ------------------------------------------------------------------
# Helper: pipeline subclass that injects adata
# ------------------------------------------------------------------


class _PipelineWithAdata(TrainingPipeline):
    """Test helper: overrides run() to inject pre-loaded adata into CellDataModule."""

    def __init__(self, config, adata) -> None:
        super().__init__(config)
        self._adata = adata

    def run(self):
        import lightning.pytorch as pl

        from scmodelforge.models.registry import get_model
        from scmodelforge.training._utils import log_training_config
        from scmodelforge.training.data_module import CellDataModule
        from scmodelforge.training.lightning_module import ScModelForgeLightningModule

        cfg = self.config

        pl.seed_everything(cfg.training.seed, workers=True)
        log_training_config(cfg)

        data_module = CellDataModule(
            data_config=cfg.data,
            tokenizer_config=cfg.tokenizer,
            training_batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
            adata=self._adata,
        )
        data_module.setup()

        cfg.model.vocab_size = len(data_module.gene_vocab)
        model = get_model(cfg.model.architecture, cfg.model)

        lightning_module = ScModelForgeLightningModule(
            model=model,
            optimizer_config=cfg.training.optimizer,
            scheduler_config=cfg.training.scheduler,
        )

        callbacks = self._build_callbacks(data_module)
        training_logger = self._build_logger()
        devices, strategy = self._resolve_devices_and_strategy()

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            precision=cfg.training.precision,
            gradient_clip_val=cfg.training.gradient_clip,
            accumulate_grad_batches=cfg.training.gradient_accumulation,
            log_every_n_steps=cfg.training.log_every_n_steps,
            callbacks=callbacks,
            logger=training_logger,
            devices=devices,
            strategy=strategy,
            default_root_dir=cfg.training.log_dir,
            enable_progress_bar=False,
        )

        trainer.fit(
            lightning_module,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )

        return trainer
