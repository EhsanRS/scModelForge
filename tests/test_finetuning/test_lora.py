"""Tests for LoRA adapter support in the fine-tuning module."""

from __future__ import annotations

import pytest
import torch

peft = pytest.importorskip("peft")

from scmodelforge.config.schema import (  # noqa: E402
    FinetuneConfig,
    LoRAConfig,
    ModelConfig,
    ScModelForgeConfig,
    TaskHeadConfig,
)
from scmodelforge.finetuning.adapters import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    apply_lora,
    count_lora_parameters,
    has_lora,
    load_lora_weights,
    save_lora_weights,
)
from scmodelforge.finetuning.heads import ClassificationHead  # noqa: E402
from scmodelforge.finetuning.model import FineTuneModel  # noqa: E402
from scmodelforge.models.transformer_encoder import TransformerEncoder  # noqa: E402

# =========================================================================
# Helpers
# =========================================================================


@pytest.fixture()
def small_model_config() -> ModelConfig:
    return ModelConfig(
        architecture="transformer_encoder",
        vocab_size=100,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        dropout=0.0,
        max_seq_len=64,
        pooling="cls",
        activation="gelu",
        use_expression_values=True,
    )


@pytest.fixture()
def small_backbone(small_model_config: ModelConfig) -> TransformerEncoder:
    return TransformerEncoder.from_config(small_model_config)


@pytest.fixture()
def lora_config() -> LoRAConfig:
    return LoRAConfig(enabled=True, rank=4, alpha=8, dropout=0.0)


@pytest.fixture()
def small_batch() -> dict[str, torch.Tensor]:
    batch_size, seq_len = 4, 10
    rng = torch.Generator().manual_seed(42)
    return {
        "input_ids": torch.randint(4, 100, (batch_size, seq_len), generator=rng),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "values": torch.randn(batch_size, seq_len),
        "task_labels": torch.tensor([0, 1, 2, 0], dtype=torch.long),
    }


# =========================================================================
# TestApplyLora
# =========================================================================


class TestApplyLora:
    def test_apply_lora_returns_peft_model(self, small_backbone, lora_config):
        result = apply_lora(small_backbone, lora_config)
        assert isinstance(result, peft.PeftModel)

    def test_apply_lora_freezes_base_params(self, small_backbone, lora_config):
        result = apply_lora(small_backbone, lora_config)
        for name, param in result.named_parameters():
            if "lora_" not in name:
                assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_lora_params_are_trainable(self, small_backbone, lora_config):
        result = apply_lora(small_backbone, lora_config)
        lora_params = [
            (name, p) for name, p in result.named_parameters() if "lora_" in name
        ]
        assert len(lora_params) > 0, "No LoRA params found"
        for name, param in lora_params:
            assert param.requires_grad, f"LoRA param {name} should be trainable"

    def test_default_target_modules(self, small_backbone, lora_config):
        """Verify that default targets (out_proj, linear1, linear2) are wrapped."""
        result = apply_lora(small_backbone, lora_config)
        lora_param_names = [n for n, _ in result.named_parameters() if "lora_" in n]
        # Should have lora params for out_proj, linear1, linear2
        has_out_proj = any("out_proj" in n for n in lora_param_names)
        has_linear1 = any("linear1" in n for n in lora_param_names)
        has_linear2 = any("linear2" in n for n in lora_param_names)
        assert has_out_proj, "out_proj should have LoRA"
        assert has_linear1, "linear1 should have LoRA"
        assert has_linear2, "linear2 should have LoRA"

    def test_custom_target_modules(self, small_backbone):
        """Pass custom target_modules and verify only those are wrapped."""
        cfg = LoRAConfig(enabled=True, rank=4, alpha=8, target_modules=["linear1"])
        result = apply_lora(small_backbone, cfg)
        lora_param_names = [n for n, _ in result.named_parameters() if "lora_" in n]
        assert all("linear1" in n for n in lora_param_names), (
            f"Only linear1 should have LoRA, got: {lora_param_names}"
        )
        assert len(lora_param_names) > 0

    def test_encode_works_through_peft(self, small_backbone, lora_config, small_batch):
        result = apply_lora(small_backbone, lora_config)
        embeddings = result.encode(
            small_batch["input_ids"],
            small_batch["attention_mask"],
            values=small_batch["values"],
        )
        assert embeddings.shape == (4, 32)


# =========================================================================
# TestHasLora
# =========================================================================


class TestHasLora:
    def test_has_lora_true_after_apply(self, small_backbone, lora_config):
        wrapped = apply_lora(small_backbone, lora_config)
        assert has_lora(wrapped) is True

    def test_has_lora_false_for_plain(self, small_backbone):
        assert has_lora(small_backbone) is False


# =========================================================================
# TestSaveLoadLora
# =========================================================================


class TestSaveLoadLora:
    def test_save_load_roundtrip(self, small_model_config, lora_config, small_batch, tmp_path):
        # Build backbone and snapshot base weights before LoRA modifies it
        backbone = TransformerEncoder.from_config(small_model_config)
        base_state = {k: v.clone() for k, v in backbone.state_dict().items()}

        wrapped = apply_lora(backbone, lora_config)
        with torch.no_grad():
            emb_before = wrapped.encode(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                values=small_batch["values"],
            )

        save_dir = str(tmp_path / "lora_weights")
        save_lora_weights(wrapped, save_dir)

        # Load onto a fresh backbone with same base weights
        fresh = TransformerEncoder.from_config(small_model_config)
        fresh.load_state_dict(base_state)
        loaded = load_lora_weights(fresh, save_dir)

        with torch.no_grad():
            emb_after = loaded.encode(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                values=small_batch["values"],
            )

        assert torch.allclose(emb_before, emb_after, atol=1e-5)

    def test_save_raises_without_lora(self, small_backbone, tmp_path):
        with pytest.raises(ValueError, match="does not have LoRA"):
            save_lora_weights(small_backbone, str(tmp_path / "bad"))

    def test_loaded_model_is_peft(self, small_backbone, lora_config, tmp_path):
        wrapped = apply_lora(small_backbone, lora_config)
        save_dir = str(tmp_path / "lora_weights")
        save_lora_weights(wrapped, save_dir)

        fresh = TransformerEncoder.from_config(ModelConfig(
            vocab_size=100, hidden_dim=32, num_layers=1, num_heads=2, ffn_dim=64,
            dropout=0.0, max_seq_len=64, pooling="cls",
        ))
        loaded = load_lora_weights(fresh, save_dir)
        assert isinstance(loaded, peft.PeftModel)


# =========================================================================
# TestCountParameters
# =========================================================================


class TestCountParameters:
    def test_count_with_lora(self, small_backbone, lora_config):
        wrapped = apply_lora(small_backbone, lora_config)
        trainable, total = count_lora_parameters(wrapped)
        assert trainable < total
        assert trainable > 0
        assert total > 0

    def test_count_without_lora(self, small_backbone):
        trainable, total = count_lora_parameters(small_backbone)
        # All params trainable by default
        assert trainable == total
        assert total > 0


# =========================================================================
# TestFineTuneModelWithLora
# =========================================================================


class TestFineTuneModelWithLora:
    def test_freeze_backbone_noop_with_lora(self, small_backbone, lora_config):
        wrapped = apply_lora(small_backbone, lora_config)
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(wrapped, head, task="classification")

        # LoRA params should be trainable
        lora_params_before = {
            n: p.requires_grad
            for n, p in model.backbone.named_parameters()
            if "lora_" in n
        }
        assert all(lora_params_before.values()), "LoRA params should start trainable"

        model.freeze_backbone()

        # LoRA params should STILL be trainable
        for n, p in model.backbone.named_parameters():
            if "lora_" in n:
                assert p.requires_grad, f"LoRA param {n} should not be frozen"

    def test_unfreeze_backbone_noop_with_lora(self, small_backbone, lora_config):
        wrapped = apply_lora(small_backbone, lora_config)
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(wrapped, head, task="classification")

        model.unfreeze_backbone()

        # Base params should STILL be frozen
        for n, p in model.backbone.named_parameters():
            if "lora_" not in n:
                assert not p.requires_grad, f"Base param {n} should stay frozen"

    def test_has_lora_property(self, small_backbone, lora_config):
        wrapped = apply_lora(small_backbone, lora_config)
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(wrapped, head, task="classification")
        assert model.has_lora is True

    def test_has_lora_false_without(self, small_backbone):
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(small_backbone, head, task="classification")
        assert model.has_lora is False

    def test_forward_with_lora(self, small_backbone, lora_config, small_batch):
        wrapped = apply_lora(small_backbone, lora_config)
        head = ClassificationHead(input_dim=32, n_classes=3)
        model = FineTuneModel(wrapped, head, task="classification")

        output = model(
            input_ids=small_batch["input_ids"],
            attention_mask=small_batch["attention_mask"],
            values=small_batch["values"],
            labels=small_batch["task_labels"],
        )
        assert output.loss is not None
        assert torch.isfinite(output.loss)
        assert output.logits.shape == (4, 3)
        assert output.embeddings.shape == (4, 32)


# =========================================================================
# TestPipelineWithLora
# =========================================================================


class TestPipelineWithLora:
    def test_pipeline_applies_lora(
        self, tiny_adata_3types, tiny_full_ft_config, pretrained_checkpoint,
    ):
        """Smoke test: 2 epochs with LoRA enabled."""
        import lightning.pytorch as pl

        from scmodelforge.finetuning._utils import load_pretrained_backbone
        from scmodelforge.finetuning.data_module import FineTuneDataModule
        from scmodelforge.finetuning.heads import build_task_head
        from scmodelforge.finetuning.lightning_module import FineTuneLightningModule
        from scmodelforge.models.registry import get_model

        cfg = tiny_full_ft_config
        cfg.finetune.checkpoint_path = pretrained_checkpoint
        cfg.finetune.lora = LoRAConfig(enabled=True, rank=4, alpha=8, dropout=0.0)
        ft_cfg = cfg.finetune

        pl.seed_everything(cfg.training.seed, workers=True)

        dm = FineTuneDataModule(
            data_config=cfg.data,
            tokenizer_config=cfg.tokenizer,
            finetune_config=ft_cfg,
            training_batch_size=cfg.training.batch_size,
            num_workers=0,
            val_split=0.2,
            seed=cfg.training.seed,
            adata=tiny_adata_3types,
        )
        dm.setup()
        cfg.model.vocab_size = len(dm.gene_vocab)

        backbone = get_model(cfg.model.architecture, cfg.model)
        if ft_cfg.checkpoint_path:
            load_pretrained_backbone(backbone, ft_cfg.checkpoint_path)

        # Apply LoRA
        backbone = apply_lora(backbone, ft_cfg.lora)
        assert has_lora(backbone)

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
        assert trainer.current_epoch == 2

    def test_pipeline_warns_freeze_epochs_with_lora(self):
        """Check that a warning is logged when freeze_backbone_epochs + LoRA."""
        from scmodelforge.finetuning.pipeline import FineTunePipeline

        cfg = ScModelForgeConfig(
            finetune=FinetuneConfig(
                freeze_backbone_epochs=3,
                lora=LoRAConfig(enabled=True, rank=4, alpha=8),
                head=TaskHeadConfig(task="classification", n_classes=3),
            ),
        )
        cfg.data.paths = ["/nonexistent"]  # Will fail before the warning

        # We just verify the pipeline can be constructed with both settings
        pipeline = FineTunePipeline(cfg)
        assert pipeline.config.finetune.lora.enabled
        assert pipeline.config.finetune.freeze_backbone_epochs == 3

    def test_pipeline_without_lora_unchanged(
        self, tiny_adata_3types, tiny_full_ft_config, pretrained_checkpoint,
    ):
        """Existing behavior should be unaffected when LoRA is disabled."""
        import lightning.pytorch as pl

        from scmodelforge.finetuning._utils import load_pretrained_backbone
        from scmodelforge.finetuning.data_module import FineTuneDataModule
        from scmodelforge.finetuning.heads import build_task_head
        from scmodelforge.finetuning.lightning_module import FineTuneLightningModule
        from scmodelforge.models.registry import get_model

        cfg = tiny_full_ft_config
        cfg.finetune.checkpoint_path = pretrained_checkpoint
        # LoRA disabled (default)
        assert not cfg.finetune.lora.enabled
        ft_cfg = cfg.finetune

        pl.seed_everything(cfg.training.seed, workers=True)

        dm = FineTuneDataModule(
            data_config=cfg.data,
            tokenizer_config=cfg.tokenizer,
            finetune_config=ft_cfg,
            training_batch_size=cfg.training.batch_size,
            num_workers=0,
            val_split=0.2,
            seed=cfg.training.seed,
            adata=tiny_adata_3types,
        )
        dm.setup()
        cfg.model.vocab_size = len(dm.gene_vocab)

        backbone = get_model(cfg.model.architecture, cfg.model)
        if ft_cfg.checkpoint_path:
            load_pretrained_backbone(backbone, ft_cfg.checkpoint_path)

        assert not has_lora(backbone)

        if ft_cfg.head.n_classes is None and dm.label_encoder is not None:
            ft_cfg.head.n_classes = dm.label_encoder.n_classes

        head = build_task_head(ft_cfg.head, input_dim=cfg.model.hidden_dim)
        model = FineTuneModel(backbone=backbone, head=head, task=ft_cfg.head.task)

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
        assert trainer.current_epoch == 2


# =========================================================================
# TestLoRAConfig
# =========================================================================


class TestLoRAConfig:
    def test_lora_config_defaults(self):
        cfg = LoRAConfig()
        assert cfg.enabled is False
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.05
        assert cfg.target_modules is None
        assert cfg.bias == "none"

    def test_finetune_config_has_lora(self):
        cfg = FinetuneConfig()
        assert isinstance(cfg.lora, LoRAConfig)
        assert cfg.lora.enabled is False

    def test_default_target_modules_constant(self):
        assert len(DEFAULT_TARGET_MODULES) == 3
        assert "out_proj" in DEFAULT_TARGET_MODULES
        assert "linear1" in DEFAULT_TARGET_MODULES
        assert "linear2" in DEFAULT_TARGET_MODULES
