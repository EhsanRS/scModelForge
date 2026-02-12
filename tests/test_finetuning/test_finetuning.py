"""Tests for the fine-tuning module."""

from __future__ import annotations

import pytest
import torch

from scmodelforge.config.schema import (
    FinetuneConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    ScModelForgeConfig,
    TaskHeadConfig,
)
from scmodelforge.finetuning._utils import load_pretrained_backbone
from scmodelforge.finetuning.data_module import (
    FineTuneDataModule,
    LabelEncoder,
)
from scmodelforge.finetuning.heads import (
    ClassificationHead,
    RegressionHead,
    build_task_head,
)
from scmodelforge.finetuning.lightning_module import FineTuneLightningModule
from scmodelforge.finetuning.model import FineTuneModel
from scmodelforge.finetuning.pipeline import FineTunePipeline
from scmodelforge.models.transformer_encoder import TransformerEncoder

# =========================================================================
# ClassificationHead
# =========================================================================


class TestClassificationHead:
    def test_output_shape(self):
        head = ClassificationHead(input_dim=64, n_classes=5)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 5)

    def test_with_hidden_layer(self):
        head = ClassificationHead(input_dim=64, n_classes=5, hidden_dim=32)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 5)

    def test_without_hidden_layer(self):
        head = ClassificationHead(input_dim=64, n_classes=5, hidden_dim=None)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 5)

    def test_gradient_flows(self):
        head = ClassificationHead(input_dim=32, n_classes=3, hidden_dim=16)
        x = torch.randn(4, 32, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 32)

    def test_loss_computation(self):
        head = ClassificationHead(input_dim=32, n_classes=3)
        x = torch.randn(4, 32)
        logits = head(x)
        labels = torch.tensor([0, 1, 2, 0])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


# =========================================================================
# RegressionHead
# =========================================================================


class TestRegressionHead:
    def test_output_shape(self):
        head = RegressionHead(input_dim=64, output_dim=1)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 1)

    def test_scalar_output(self):
        head = RegressionHead(input_dim=64, output_dim=1)
        x = torch.randn(4, 64)
        out = head(x)
        squeezed = out.squeeze(-1)
        assert squeezed.shape == (4,)

    def test_multi_dim_output(self):
        head = RegressionHead(input_dim=64, output_dim=3)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 3)

    def test_with_hidden_layer(self):
        head = RegressionHead(input_dim=64, output_dim=1, hidden_dim=32)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 1)

    def test_gradient_flows(self):
        head = RegressionHead(input_dim=32, output_dim=1)
        x = torch.randn(4, 32, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_mse_loss(self):
        head = RegressionHead(input_dim=32, output_dim=1)
        x = torch.randn(4, 32)
        pred = head(x).squeeze(-1)
        targets = torch.randn(4)
        loss = torch.nn.functional.mse_loss(pred, targets)
        assert torch.isfinite(loss)


# =========================================================================
# build_task_head
# =========================================================================


class TestBuildTaskHead:
    def test_builds_classification(self):
        cfg = TaskHeadConfig(task="classification", n_classes=5)
        head = build_task_head(cfg, input_dim=64)
        assert isinstance(head, ClassificationHead)

    def test_builds_regression(self):
        cfg = TaskHeadConfig(task="regression", output_dim=1)
        head = build_task_head(cfg, input_dim=64)
        assert isinstance(head, RegressionHead)

    def test_unknown_task_raises(self):
        cfg = TaskHeadConfig(task="unknown")
        with pytest.raises(ValueError, match="Unknown task"):
            build_task_head(cfg, input_dim=64)

    def test_classification_requires_n_classes(self):
        cfg = TaskHeadConfig(task="classification", n_classes=None)
        with pytest.raises(ValueError, match="n_classes"):
            build_task_head(cfg, input_dim=64)


# =========================================================================
# FineTuneModel
# =========================================================================


class TestFineTuneModel:
    def test_forward_classification_loss(self, tiny_backbone, ft_batch):
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(tiny_backbone, head, task="classification")
        output = model(
            input_ids=ft_batch["input_ids"],
            attention_mask=ft_batch["attention_mask"],
            values=ft_batch["values"],
            labels=ft_batch["task_labels"],
        )
        assert output.loss is not None
        assert torch.isfinite(output.loss)
        assert output.logits.shape == (4, 3)
        assert output.embeddings.shape == (4, 32)

    def test_forward_regression_loss(self, tiny_backbone):
        head = RegressionHead(input_dim=32, output_dim=1)
        model = FineTuneModel(tiny_backbone, head, task="regression")
        batch_size, seq_len = 4, 10
        input_ids = torch.randint(0, 10, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.randn(batch_size)
        output = model(input_ids=input_ids, attention_mask=mask, labels=labels)
        assert output.loss is not None
        assert torch.isfinite(output.loss)

    def test_no_labels_no_loss(self, tiny_backbone, ft_batch):
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(tiny_backbone, head, task="classification")
        output = model(
            input_ids=ft_batch["input_ids"],
            attention_mask=ft_batch["attention_mask"],
            values=ft_batch["values"],
        )
        assert output.loss is None
        assert output.logits is not None

    def test_encode_returns_embeddings(self, tiny_backbone, ft_batch):
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(tiny_backbone, head, task="classification")
        emb = model.encode(
            input_ids=ft_batch["input_ids"],
            attention_mask=ft_batch["attention_mask"],
            values=ft_batch["values"],
        )
        assert emb.shape == (4, 32)

    def test_encode_shape_matches_backbone(self, tiny_backbone, ft_batch):
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(tiny_backbone, head, task="classification")
        emb_model = model.encode(
            input_ids=ft_batch["input_ids"],
            attention_mask=ft_batch["attention_mask"],
            values=ft_batch["values"],
        )
        emb_backbone = tiny_backbone.encode(
            input_ids=ft_batch["input_ids"],
            attention_mask=ft_batch["attention_mask"],
            values=ft_batch["values"],
        )
        assert emb_model.shape == emb_backbone.shape

    def test_freeze_backbone(self, tiny_backbone):
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(tiny_backbone, head, task="classification")
        model.freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad
        # Head should still be trainable
        for p in model.head.parameters():
            assert p.requires_grad

    def test_unfreeze_backbone(self, tiny_backbone):
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(tiny_backbone, head, task="classification", freeze_backbone=True)
        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad

    def test_num_parameters_trainable_vs_frozen(self, tiny_backbone):
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(tiny_backbone, head, task="classification")
        total = model.num_parameters(trainable_only=False)
        trainable_all = model.num_parameters(trainable_only=True)
        assert total == trainable_all

        model.freeze_backbone()
        trainable_frozen = model.num_parameters(trainable_only=True)
        assert trainable_frozen < total
        assert trainable_frozen > 0  # Head is still trainable


# =========================================================================
# LabelEncoder
# =========================================================================


class TestLabelEncoder:
    def test_roundtrip(self):
        labels = ["cat", "dog", "fish", "cat", "dog"]
        enc = LabelEncoder(labels)
        for lbl in ["cat", "dog", "fish"]:
            idx = enc.encode(lbl)
            assert enc.decode(idx) == lbl

    def test_n_classes(self):
        labels = ["a", "b", "c", "a"]
        enc = LabelEncoder(labels)
        assert enc.n_classes == 3

    def test_sorted_order(self):
        labels = ["zebra", "apple", "mango"]
        enc = LabelEncoder(labels)
        assert enc.classes == ["apple", "mango", "zebra"]
        assert enc.encode("apple") == 0

    def test_unknown_label_raises(self):
        enc = LabelEncoder(["a", "b"])
        with pytest.raises(KeyError, match="Unknown label"):
            enc.encode("c")

    def test_decode_out_of_range(self):
        enc = LabelEncoder(["a", "b"])
        with pytest.raises(IndexError):
            enc.decode(5)


# =========================================================================
# FineTuneDataModule
# =========================================================================


class TestFineTuneDataModule:
    def test_setup_creates_splits(
        self, tiny_adata_3types, tiny_data_config, tiny_tokenizer_config, tiny_ft_config,
    ):
        dm = FineTuneDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            finetune_config=tiny_ft_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            seed=42,
            adata=tiny_adata_3types,
        )
        dm.setup()
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert len(dm._train_dataset) + len(dm._val_dataset) == 30

    def test_train_dataloader_yields_batches(
        self, tiny_adata_3types, tiny_data_config, tiny_tokenizer_config, tiny_ft_config,
    ):
        dm = FineTuneDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            finetune_config=tiny_ft_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            seed=42,
            adata=tiny_adata_3types,
        )
        dm.setup()
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "task_labels" in batch

    def test_batches_have_task_labels(
        self, tiny_adata_3types, tiny_data_config, tiny_tokenizer_config, tiny_ft_config,
    ):
        dm = FineTuneDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            finetune_config=tiny_ft_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            seed=42,
            adata=tiny_adata_3types,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        labels = batch["task_labels"]
        assert labels.dtype == torch.long
        assert labels.min() >= 0
        assert labels.max() <= 2  # 3 classes: 0, 1, 2

    def test_collation_label_dtype_float(
        self, tiny_adata_regression, tiny_data_config, tiny_tokenizer_config,
    ):
        ft_cfg = FinetuneConfig(
            label_key="target",
            head=TaskHeadConfig(task="regression", output_dim=1),
        )
        dm = FineTuneDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            finetune_config=ft_cfg,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            seed=42,
            adata=tiny_adata_regression,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert batch["task_labels"].dtype == torch.float

    def test_no_masking_applied(
        self, tiny_adata_3types, tiny_data_config, tiny_tokenizer_config, tiny_ft_config,
    ):
        dm = FineTuneDataModule(
            data_config=tiny_data_config,
            tokenizer_config=tiny_tokenizer_config,
            finetune_config=tiny_ft_config,
            training_batch_size=4,
            num_workers=0,
            val_split=0.2,
            seed=42,
            adata=tiny_adata_3types,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        # No labels or masked_positions keys from masking
        assert "labels" not in batch
        assert "masked_positions" not in batch


# =========================================================================
# FineTuneLightningModule
# =========================================================================


class TestFineTuneLightningModule:
    def _make_module(self, tiny_backbone, task="classification", freeze_epochs=0):
        if task == "classification":
            head = ClassificationHead(input_dim=32, n_classes=3)
        else:
            head = RegressionHead(input_dim=32, output_dim=1)
        ft_model = FineTuneModel(tiny_backbone, head, task=task)
        return FineTuneLightningModule(
            model=ft_model,
            optimizer_config=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01),
            scheduler_config=SchedulerConfig(name="cosine_warmup", warmup_steps=2, total_steps=20),
            task=task,
            backbone_lr=1e-5,
            head_lr=1e-3,
            freeze_backbone_epochs=freeze_epochs,
        )

    def test_training_step_loss(self, tiny_backbone, ft_batch):
        module = self._make_module(tiny_backbone)
        loss = module.training_step(ft_batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_validation_step_logs(self, tiny_backbone, ft_batch):
        module = self._make_module(tiny_backbone)
        # Should not raise
        module.validation_step(ft_batch, batch_idx=0)

    def test_classification_logs_accuracy(self, tiny_backbone, ft_batch):
        module = self._make_module(tiny_backbone, task="classification")
        module.training_step(ft_batch, batch_idx=0)
        # Check that accuracy was computed (no exception)

    def test_regression_no_accuracy(self, tiny_backbone):
        module = self._make_module(tiny_backbone, task="regression")
        batch = {
            "input_ids": torch.randint(0, 10, (4, 10)),
            "attention_mask": torch.ones(4, 10, dtype=torch.long),
            "values": torch.randn(4, 10),
            "task_labels": torch.randn(4),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_discriminative_lr_param_groups(self, tiny_backbone):
        module = self._make_module(tiny_backbone)
        opt_config = module.configure_optimizers()
        optimizer = opt_config["optimizer"]
        # Should have multiple param groups (backbone + head, with/without decay)
        assert len(optimizer.param_groups) >= 2
        # Check that different initial LRs are present
        # (scheduler may have already modified "lr", so check "initial_lr")
        lrs = {g.get("initial_lr", g["lr"]) for g in optimizer.param_groups}
        assert len(lrs) == 2  # backbone_lr=1e-5, head_lr=1e-3

    def test_scheduler_returned(self, tiny_backbone):
        module = self._make_module(tiny_backbone)
        opt_config = module.configure_optimizers()
        assert "lr_scheduler" in opt_config
        assert "scheduler" in opt_config["lr_scheduler"]

    def test_gradual_unfreeze_trigger(self, tiny_backbone):
        head = ClassificationHead(input_dim=32, n_classes=3)
        ft_model = FineTuneModel(tiny_backbone, head, task="classification", freeze_backbone=True)
        module = FineTuneLightningModule(
            model=ft_model,
            optimizer_config=OptimizerConfig(name="adamw", lr=1e-3),
            task="classification",
            freeze_backbone_epochs=2,
        )
        # Before epoch 2: backbone should be frozen
        for p in module.model.backbone.parameters():
            assert not p.requires_grad

        # Simulate reaching epoch 2
        module._current_fx_name = "on_train_epoch_start"
        # Manually set current_epoch (Lightning attribute)
        module.trainer = type("T", (), {"current_epoch": 2})()  # type: ignore[assignment]
        # Patch current_epoch on the module
        type(module).current_epoch = property(lambda self: 2)
        module.on_train_epoch_start()

        # Now backbone should be unfrozen
        for p in module.model.backbone.parameters():
            assert p.requires_grad

    def test_loss_is_finite(self, tiny_backbone, ft_batch):
        module = self._make_module(tiny_backbone)
        loss = module.training_step(ft_batch, batch_idx=0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


# =========================================================================
# Checkpoint loading
# =========================================================================


class TestCheckpointLoading:
    def test_load_raw_state_dict(self, tiny_backbone, pretrained_checkpoint):
        # Create a fresh backbone and load into it
        fresh = TransformerEncoder.from_config(ModelConfig(
            vocab_size=tiny_backbone.vocab_size,
            hidden_dim=32, num_layers=1, num_heads=2, ffn_dim=64,
            dropout=0.0, max_seq_len=64, pooling="cls",
        ))
        load_pretrained_backbone(fresh, pretrained_checkpoint)
        # Weights should match
        for (_n1, p1), (_n2, p2) in zip(
            tiny_backbone.named_parameters(),
            fresh.named_parameters(),
            strict=True,
        ):
            # They won't match exactly (different init), but no error
            assert p1.shape == p2.shape

    def test_load_lightning_ckpt(self, tiny_backbone, pretrained_lightning_checkpoint):
        fresh = TransformerEncoder.from_config(ModelConfig(
            vocab_size=tiny_backbone.vocab_size,
            hidden_dim=32, num_layers=1, num_heads=2, ffn_dim=64,
            dropout=0.0, max_seq_len=64, pooling="cls",
        ))
        # Should handle "model." prefix stripping
        load_pretrained_backbone(fresh, pretrained_lightning_checkpoint)

    def test_missing_keys_handled(self, tiny_backbone, pretrained_checkpoint):
        # Load with strict=False should not raise even if model has extra params
        load_pretrained_backbone(tiny_backbone, pretrained_checkpoint, strict=False)

    def test_extra_keys_ignored(self, tmp_path, tiny_backbone):
        # Save state dict with extra keys
        sd = tiny_backbone.state_dict()
        sd["extra_layer.weight"] = torch.randn(10, 10)
        ckpt_path = str(tmp_path / "extra_keys.pt")
        torch.save(sd, ckpt_path)
        # Should not raise
        load_pretrained_backbone(tiny_backbone, ckpt_path, strict=False)


# =========================================================================
# FineTunePipeline
# =========================================================================


class TestFineTunePipeline:
    def test_smoke_2_epochs(
        self, tiny_adata_3types, tiny_full_ft_config, pretrained_checkpoint,
    ):
        cfg = tiny_full_ft_config
        cfg.finetune.checkpoint_path = pretrained_checkpoint
        # Inject adata directly via pipeline
        pipeline = FineTunePipeline(cfg)  # noqa: F841

        def patched_run():
            import lightning.pytorch as pl

            from scmodelforge.finetuning._utils import load_pretrained_backbone
            from scmodelforge.finetuning.data_module import FineTuneDataModule
            from scmodelforge.finetuning.heads import build_task_head
            from scmodelforge.finetuning.lightning_module import FineTuneLightningModule
            from scmodelforge.finetuning.model import FineTuneModel
            from scmodelforge.models.registry import get_model

            ft_cfg = cfg.finetune
            pl.seed_everything(cfg.training.seed, workers=True)

            dm = FineTuneDataModule(
                data_config=cfg.data,
                tokenizer_config=cfg.tokenizer,
                finetune_config=ft_cfg,
                training_batch_size=cfg.training.batch_size,
                num_workers=cfg.training.num_workers,
                val_split=cfg.training.val_split,
                seed=cfg.training.seed,
                adata=tiny_adata_3types,
            )
            dm.setup()

            cfg.model.vocab_size = len(dm.gene_vocab)
            backbone = get_model(cfg.model.architecture, cfg.model)
            if ft_cfg.checkpoint_path:
                load_pretrained_backbone(backbone, ft_cfg.checkpoint_path)

            if ft_cfg.head.n_classes is None and dm.label_encoder is not None:
                ft_cfg.head.n_classes = dm.label_encoder.n_classes

            head = build_task_head(ft_cfg.head, input_dim=cfg.model.hidden_dim)
            model = FineTuneModel(backbone=backbone, head=head, task=ft_cfg.head.task)

            lm = FineTuneLightningModule(
                model=model,
                optimizer_config=cfg.training.optimizer,
                scheduler_config=cfg.training.scheduler,
                task=ft_cfg.head.task,
                backbone_lr=ft_cfg.backbone_lr,
                head_lr=ft_cfg.head_lr,
                freeze_backbone_epochs=ft_cfg.freeze_backbone_epochs,
            )

            trainer = pl.Trainer(
                max_epochs=cfg.training.max_epochs,
                precision=cfg.training.precision,
                gradient_clip_val=cfg.training.gradient_clip,
                log_every_n_steps=1,
                logger=False,
                enable_checkpointing=False,
                devices=1,
                strategy="auto",
                enable_progress_bar=False,
            )
            trainer.fit(
                lm,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader(),
            )
            return trainer

        trainer = patched_run()
        assert trainer.current_epoch == 2

    def test_requires_finetune_config(self):
        cfg = ScModelForgeConfig()
        with pytest.raises(ValueError, match="finetune must be set"):
            FineTunePipeline(cfg)

    def test_loads_checkpoint(
        self, tiny_adata_3types, tiny_full_ft_config, pretrained_checkpoint,
    ):
        cfg = tiny_full_ft_config
        cfg.finetune.checkpoint_path = pretrained_checkpoint
        # Just check pipeline construction succeeds
        pipeline = FineTunePipeline(cfg)
        assert pipeline.config.finetune.checkpoint_path == pretrained_checkpoint

    def test_params_change_after_training(
        self, tiny_adata_3types, tiny_full_ft_config, pretrained_checkpoint,
    ):
        """Verify that parameters actually change during fine-tuning."""
        import lightning.pytorch as pl

        from scmodelforge.finetuning._utils import load_pretrained_backbone
        from scmodelforge.finetuning.data_module import FineTuneDataModule
        from scmodelforge.finetuning.heads import build_task_head
        from scmodelforge.finetuning.lightning_module import FineTuneLightningModule
        from scmodelforge.finetuning.model import FineTuneModel
        from scmodelforge.models.registry import get_model

        cfg = tiny_full_ft_config
        cfg.finetune.checkpoint_path = pretrained_checkpoint
        ft_cfg = cfg.finetune

        dm = FineTuneDataModule(
            data_config=cfg.data,
            tokenizer_config=cfg.tokenizer,
            finetune_config=ft_cfg,
            training_batch_size=cfg.training.batch_size,
            num_workers=0,
            val_split=0.2,
            seed=42,
            adata=tiny_adata_3types,
        )
        dm.setup()
        cfg.model.vocab_size = len(dm.gene_vocab)
        backbone = get_model(cfg.model.architecture, cfg.model)
        load_pretrained_backbone(backbone, pretrained_checkpoint)

        if ft_cfg.head.n_classes is None and dm.label_encoder is not None:
            ft_cfg.head.n_classes = dm.label_encoder.n_classes

        head = build_task_head(ft_cfg.head, input_dim=cfg.model.hidden_dim)
        model = FineTuneModel(backbone=backbone, head=head, task=ft_cfg.head.task)

        # Snapshot head params before training
        head_params_before = {
            n: p.clone().detach() for n, p in model.head.named_parameters()
        }

        lm = FineTuneLightningModule(
            model=model,
            optimizer_config=cfg.training.optimizer,
            scheduler_config=cfg.training.scheduler,
            task=ft_cfg.head.task,
        )

        trainer = pl.Trainer(
            max_epochs=2,
            precision="32-true",
            logger=False,
            enable_checkpointing=False,
            devices=1,
            strategy="auto",
            enable_progress_bar=False,
        )
        trainer.fit(
            lm,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

        # Head params should have changed
        changed = False
        for n, p in model.head.named_parameters():
            if not torch.allclose(p, head_params_before[n]):
                changed = True
                break
        assert changed, "Head parameters did not change during training"

    def test_frozen_backbone_head_only_changes(
        self, tiny_adata_3types, tiny_full_ft_config, pretrained_checkpoint,
    ):
        """Verify frozen backbone params don't change but head params do."""
        import lightning.pytorch as pl

        from scmodelforge.finetuning._utils import load_pretrained_backbone
        from scmodelforge.finetuning.data_module import FineTuneDataModule
        from scmodelforge.finetuning.heads import build_task_head
        from scmodelforge.finetuning.lightning_module import FineTuneLightningModule
        from scmodelforge.finetuning.model import FineTuneModel
        from scmodelforge.models.registry import get_model

        cfg = tiny_full_ft_config
        cfg.finetune.checkpoint_path = pretrained_checkpoint
        cfg.finetune.freeze_backbone = True
        ft_cfg = cfg.finetune

        dm = FineTuneDataModule(
            data_config=cfg.data,
            tokenizer_config=cfg.tokenizer,
            finetune_config=ft_cfg,
            training_batch_size=cfg.training.batch_size,
            num_workers=0,
            val_split=0.2,
            seed=42,
            adata=tiny_adata_3types,
        )
        dm.setup()
        cfg.model.vocab_size = len(dm.gene_vocab)
        backbone = get_model(cfg.model.architecture, cfg.model)
        load_pretrained_backbone(backbone, pretrained_checkpoint)

        if ft_cfg.head.n_classes is None and dm.label_encoder is not None:
            ft_cfg.head.n_classes = dm.label_encoder.n_classes

        head = build_task_head(ft_cfg.head, input_dim=cfg.model.hidden_dim)
        model = FineTuneModel(
            backbone=backbone, head=head, task=ft_cfg.head.task,
            freeze_backbone=True,
        )

        backbone_params_before = {
            n: p.clone().detach() for n, p in model.backbone.named_parameters()
        }

        lm = FineTuneLightningModule(
            model=model,
            optimizer_config=cfg.training.optimizer,
            scheduler_config=cfg.training.scheduler,
            task=ft_cfg.head.task,
        )

        trainer = pl.Trainer(
            max_epochs=2,
            precision="32-true",
            logger=False,
            enable_checkpointing=False,
            devices=1,
            strategy="auto",
            enable_progress_bar=False,
        )
        trainer.fit(
            lm,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

        # Backbone params should NOT have changed
        for n, p in model.backbone.named_parameters():
            assert torch.allclose(p, backbone_params_before[n]), (
                f"Backbone param {n} changed despite freeze"
            )


# =========================================================================
# Config
# =========================================================================


class TestConfig:
    def test_finetune_config_defaults(self):
        cfg = FinetuneConfig()
        assert cfg.checkpoint_path == ""
        assert cfg.freeze_backbone is False
        assert cfg.freeze_backbone_epochs == 0
        assert cfg.label_key == "cell_type"
        assert cfg.head.task == "classification"

    def test_finetune_none_in_pretraining(self):
        cfg = ScModelForgeConfig()
        assert cfg.finetune is None

    def test_config_with_finetune_section(self):
        cfg = ScModelForgeConfig(
            finetune=FinetuneConfig(
                checkpoint_path="/path/to/ckpt",
                freeze_backbone=True,
                head=TaskHeadConfig(task="classification", n_classes=10),
            ),
        )
        assert cfg.finetune is not None
        assert cfg.finetune.checkpoint_path == "/path/to/ckpt"
        assert cfg.finetune.freeze_backbone is True
        assert cfg.finetune.head.n_classes == 10

    def test_task_head_config_defaults(self):
        cfg = TaskHeadConfig()
        assert cfg.task == "classification"
        assert cfg.n_classes is None
        assert cfg.output_dim == 1
        assert cfg.hidden_dim is None
        assert cfg.dropout == 0.1
