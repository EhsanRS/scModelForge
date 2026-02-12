"""End-to-end fine-tuning pipeline orchestrator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightning.pytorch as pl

    from scmodelforge.config.schema import ScModelForgeConfig

logger = logging.getLogger(__name__)


class FineTunePipeline:
    """Config-driven fine-tuning pipeline.

    Orchestrates data loading, backbone loading, head construction,
    and Lightning training for downstream tasks.

    Parameters
    ----------
    config
        Full scModelForge configuration (must have ``finetune`` set).
    """

    def __init__(self, config: ScModelForgeConfig) -> None:
        if config.finetune is None:
            msg = "ScModelForgeConfig.finetune must be set for fine-tuning."
            raise ValueError(msg)
        self.config = config

    def run(self) -> pl.Trainer:
        """Execute the full fine-tuning pipeline.

        Returns
        -------
        pl.Trainer
            The Lightning Trainer after fitting.
        """
        import lightning.pytorch as pl

        from scmodelforge.finetuning._utils import load_pretrained_backbone
        from scmodelforge.finetuning.data_module import FineTuneDataModule
        from scmodelforge.finetuning.heads import build_task_head
        from scmodelforge.finetuning.lightning_module import FineTuneLightningModule
        from scmodelforge.finetuning.model import FineTuneModel
        from scmodelforge.models.registry import get_model

        cfg = self.config
        ft_cfg = cfg.finetune

        # 1. Seed
        pl.seed_everything(cfg.training.seed, workers=True)

        # 2. Data module
        data_module = FineTuneDataModule(
            data_config=cfg.data,
            tokenizer_config=cfg.tokenizer,
            finetune_config=ft_cfg,
            training_batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
        )
        data_module.setup()

        # 3. Set vocab_size from gene vocab
        cfg.model.vocab_size = len(data_module.gene_vocab)

        # 4. Build backbone
        backbone = get_model(cfg.model.architecture, cfg.model)
        logger.info(
            "Backbone: %s (%d parameters)",
            cfg.model.architecture,
            sum(p.numel() for p in backbone.parameters()),
        )

        # 5. Load pretrained checkpoint
        if ft_cfg.checkpoint_path:
            load_pretrained_backbone(backbone, ft_cfg.checkpoint_path)

        # 5b. Apply LoRA if configured
        if ft_cfg.lora.enabled:
            from scmodelforge.finetuning.adapters import apply_lora, count_lora_parameters

            backbone = apply_lora(backbone, ft_cfg.lora)
            trainable, total = count_lora_parameters(backbone)
            logger.info(
                "LoRA applied: %d trainable / %d total params (%.2f%%)",
                trainable,
                total,
                100.0 * trainable / total,
            )

            if ft_cfg.freeze_backbone_epochs > 0:
                logger.warning("freeze_backbone_epochs is ignored when LoRA is active.")

        # 6. Infer n_classes from label_encoder if needed
        if ft_cfg.head.task.lower() == "classification" and ft_cfg.head.n_classes is None:
            if data_module.label_encoder is not None:
                ft_cfg.head.n_classes = data_module.label_encoder.n_classes
            else:
                msg = "Cannot infer n_classes: no label_encoder available."
                raise ValueError(msg)

        # 7. Build task head
        head = build_task_head(ft_cfg.head, input_dim=cfg.model.hidden_dim)

        # 8. Compose FineTuneModel
        should_freeze = (ft_cfg.freeze_backbone or ft_cfg.freeze_backbone_epochs > 0) and not ft_cfg.lora.enabled
        model = FineTuneModel(
            backbone=backbone,
            head=head,
            task=ft_cfg.head.task,
            freeze_backbone=should_freeze,
        )
        logger.info(
            "FineTuneModel: %d total params, %d trainable",
            model.num_parameters(trainable_only=False),
            model.num_parameters(trainable_only=True),
        )

        # 9. Lightning module
        lightning_module = FineTuneLightningModule(
            model=model,
            optimizer_config=cfg.training.optimizer,
            scheduler_config=cfg.training.scheduler,
            task=ft_cfg.head.task,
            backbone_lr=ft_cfg.backbone_lr,
            head_lr=ft_cfg.head_lr,
            freeze_backbone_epochs=ft_cfg.freeze_backbone_epochs,
        )

        # 10. Callbacks
        callbacks = self._build_callbacks()

        # 11. Logger
        training_logger = self._build_logger()

        # 12. Devices and strategy
        devices, strategy = self._resolve_devices_and_strategy()

        # 13. Trainer
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

        # 14. Fit
        trainer.fit(
            lightning_module,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )

        logger.info("Fine-tuning complete.")
        return trainer

    def _build_callbacks(self) -> list:
        """Build the list of Lightning callbacks."""
        import lightning.pytorch as pl

        from scmodelforge.training.callbacks import TrainingMetricsLogger

        cfg = self.config.training
        callbacks: list[pl.Callback] = []

        # Model checkpoint
        monitor = "val/loss"
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                monitor=monitor,
                mode="min",
                save_top_k=cfg.save_top_k,
                filename="finetune-epoch{epoch:02d}-val_loss{val/loss:.4f}",
                auto_insert_metric_name=False,
            )
        )

        # LR monitor
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

        # Throughput
        callbacks.append(TrainingMetricsLogger(log_every_n_steps=cfg.log_every_n_steps))

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
        """Determine devices and strategy for the Trainer."""
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
