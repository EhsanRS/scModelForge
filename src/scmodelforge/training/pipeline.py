"""End-to-end training pipeline orchestrator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightning.pytorch as pl

    from scmodelforge.config.schema import ScModelForgeConfig

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Config-driven training pipeline.

    Orchestrates data loading, model creation, and Lightning training.

    Parameters
    ----------
    config
        Full scModelForge configuration.
    """

    def __init__(self, config: ScModelForgeConfig) -> None:
        self.config = config

    def run(self) -> pl.Trainer:
        """Execute the full training pipeline.

        Returns
        -------
        pl.Trainer
            The Lightning Trainer after fitting.
        """
        import lightning.pytorch as pl

        from scmodelforge.models.registry import get_model
        from scmodelforge.training._utils import log_training_config
        from scmodelforge.training.data_module import CellDataModule
        from scmodelforge.training.lightning_module import ScModelForgeLightningModule

        cfg = self.config

        # 1. Seed
        pl.seed_everything(cfg.training.seed, workers=True)

        # 2. Log config
        log_training_config(cfg)

        # 3. Data module
        data_module = CellDataModule(
            data_config=cfg.data,
            tokenizer_config=cfg.tokenizer,
            training_batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
            sampling_config=cfg.training.sampling,
            gene_selection_config=cfg.data.gene_selection,
        )
        data_module.setup()

        # 4. Set vocab_size from gene vocab
        cfg.model.vocab_size = len(data_module.gene_vocab)

        # 5. Build model
        model = get_model(cfg.model.architecture, cfg.model)
        logger.info("Model: %s (%d parameters)", cfg.model.architecture,
                     sum(p.numel() for p in model.parameters()))

        # 6. Lightning module
        lightning_module = ScModelForgeLightningModule(
            model=model,
            optimizer_config=cfg.training.optimizer,
            scheduler_config=cfg.training.scheduler,
        )

        # 7. Callbacks
        callbacks = self._build_callbacks()

        # 8. Logger
        training_logger = self._build_logger()

        # 9. Devices and strategy
        devices, strategy = self._resolve_devices_and_strategy()

        # 10. Trainer
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
            enable_progress_bar=True,
        )

        # 11. Fit
        resume_from = cfg.training.resume_from
        trainer.fit(
            lightning_module,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
            ckpt_path=resume_from,
        )

        logger.info("Training complete.")
        return trainer

    def _build_callbacks(self) -> list:
        """Build the list of Lightning callbacks."""
        import lightning.pytorch as pl

        from scmodelforge.training.callbacks import GradientNormLogger, TrainingMetricsLogger

        cfg = self.config.training
        callbacks: list[pl.Callback] = []

        # Model checkpoint
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                monitor="val/loss",
                mode="min",
                save_top_k=cfg.save_top_k,
                filename="epoch{epoch:02d}-val_loss{val/loss:.4f}",
                auto_insert_metric_name=False,
            )
        )

        # LR monitor
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

        # Custom callbacks
        callbacks.append(TrainingMetricsLogger(log_every_n_steps=cfg.log_every_n_steps))
        callbacks.append(GradientNormLogger(log_every_n_steps=cfg.log_every_n_steps))

        return callbacks

    def _build_logger(self):
        """Build the training logger (wandb, tensorboard, or csv)."""
        cfg = self.config.training
        logger_name = cfg.logger.lower()

        if logger_name == "wandb":
            try:
                from lightning.pytorch.loggers import WandbLogger

                return WandbLogger(
                    project=cfg.wandb_project,
                    name=cfg.run_name,
                    save_dir=cfg.log_dir,
                )
            except ImportError:
                logger.warning("wandb not installed — falling back to CSVLogger.")
                logger_name = "csv"

        if logger_name == "tensorboard":
            from lightning.pytorch.loggers import TensorBoardLogger

            return TensorBoardLogger(
                save_dir=cfg.log_dir,
                name=cfg.run_name or "default",
            )

        # Default: CSV
        from lightning.pytorch.loggers import CSVLogger

        return CSVLogger(
            save_dir=cfg.log_dir,
            name=cfg.run_name or "default",
        )

    def _resolve_devices_and_strategy(self) -> tuple:
        """Determine devices and strategy for the Trainer.

        Returns
        -------
        tuple
            ``(devices, strategy)`` suitable for ``pl.Trainer``.
        """
        import torch

        cfg = self.config.training

        if not torch.cuda.is_available():
            return 1, "auto"

        n_available = torch.cuda.device_count()
        n_devices = min(cfg.num_gpus, n_available) if cfg.num_gpus is not None else n_available

        if n_devices <= 0:
            return 1, "auto"

        strategy = cfg.strategy if n_devices > 1 else "auto"
        return n_devices, strategy
