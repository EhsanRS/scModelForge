# {py:mod}`scmodelforge.training.callbacks`

```{py:module} scmodelforge.training.callbacks
```

```{autodoc2-docstring} scmodelforge.training.callbacks
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrainingMetricsLogger <scmodelforge.training.callbacks.TrainingMetricsLogger>`
  - ```{autodoc2-docstring} scmodelforge.training.callbacks.TrainingMetricsLogger
    :summary:
    ```
* - {py:obj}`GradientNormLogger <scmodelforge.training.callbacks.GradientNormLogger>`
  - ```{autodoc2-docstring} scmodelforge.training.callbacks.GradientNormLogger
    :summary:
    ```
````

### API

`````{py:class} TrainingMetricsLogger(log_every_n_steps: int = 50)
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger

Bases: {py:obj}`lightning.pytorch.Callback`

```{autodoc2-docstring} scmodelforge.training.callbacks.TrainingMetricsLogger
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.training.callbacks.TrainingMetricsLogger.__init__
```

````{py:method} on_train_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_train_epoch_start

````

````{py:method} on_train_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_train_batch_start

````

````{py:method} on_train_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: typing.Any, batch: typing.Any, batch_idx: int) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_train_batch_end

````

````{py:method} on_train_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_train_epoch_end

````

````{py:property} state_key
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.state_key
:type: str

````

````{py:method} setup(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, stage: str) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.setup

````

````{py:method} teardown(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, stage: str) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.teardown

````

````{py:method} on_fit_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_fit_start

````

````{py:method} on_fit_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_fit_end

````

````{py:method} on_sanity_check_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_sanity_check_start

````

````{py:method} on_sanity_check_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_sanity_check_end

````

````{py:method} on_validation_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_validation_epoch_start

````

````{py:method} on_validation_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_validation_epoch_end

````

````{py:method} on_test_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_test_epoch_start

````

````{py:method} on_test_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_test_epoch_end

````

````{py:method} on_predict_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_predict_epoch_start

````

````{py:method} on_predict_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_predict_epoch_end

````

````{py:method} on_validation_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_validation_batch_start

````

````{py:method} on_validation_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_validation_batch_end

````

````{py:method} on_test_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_test_batch_start

````

````{py:method} on_test_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_test_batch_end

````

````{py:method} on_predict_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_predict_batch_start

````

````{py:method} on_predict_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: typing.Any, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_predict_batch_end

````

````{py:method} on_train_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_train_start

````

````{py:method} on_train_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_train_end

````

````{py:method} on_validation_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_validation_start

````

````{py:method} on_validation_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_validation_end

````

````{py:method} on_test_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_test_start

````

````{py:method} on_test_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_test_end

````

````{py:method} on_predict_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_predict_start

````

````{py:method} on_predict_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_predict_end

````

````{py:method} on_exception(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, exception: BaseException) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_exception

````

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.state_dict

````

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.load_state_dict

````

````{py:method} on_save_checkpoint(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, checkpoint: dict[str, typing.Any]) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_save_checkpoint

````

````{py:method} on_load_checkpoint(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, checkpoint: dict[str, typing.Any]) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_load_checkpoint

````

````{py:method} on_before_backward(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, loss: torch.Tensor) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_before_backward

````

````{py:method} on_after_backward(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_after_backward

````

````{py:method} on_before_optimizer_step(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, optimizer: torch.optim.Optimizer) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_before_optimizer_step

````

````{py:method} on_before_zero_grad(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, optimizer: torch.optim.Optimizer) -> None
:canonical: scmodelforge.training.callbacks.TrainingMetricsLogger.on_before_zero_grad

````

`````

`````{py:class} GradientNormLogger(log_every_n_steps: int = 50)
:canonical: scmodelforge.training.callbacks.GradientNormLogger

Bases: {py:obj}`lightning.pytorch.Callback`

```{autodoc2-docstring} scmodelforge.training.callbacks.GradientNormLogger
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.training.callbacks.GradientNormLogger.__init__
```

````{py:method} on_before_optimizer_step(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, optimizer: torch.optim.Optimizer) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_before_optimizer_step

````

````{py:property} state_key
:canonical: scmodelforge.training.callbacks.GradientNormLogger.state_key
:type: str

````

````{py:method} setup(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, stage: str) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.setup

````

````{py:method} teardown(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, stage: str) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.teardown

````

````{py:method} on_fit_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_fit_start

````

````{py:method} on_fit_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_fit_end

````

````{py:method} on_sanity_check_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_sanity_check_start

````

````{py:method} on_sanity_check_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_sanity_check_end

````

````{py:method} on_train_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_train_batch_start

````

````{py:method} on_train_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_train_batch_end

````

````{py:method} on_train_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_train_epoch_start

````

````{py:method} on_train_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_train_epoch_end

````

````{py:method} on_validation_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_validation_epoch_start

````

````{py:method} on_validation_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_validation_epoch_end

````

````{py:method} on_test_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_test_epoch_start

````

````{py:method} on_test_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_test_epoch_end

````

````{py:method} on_predict_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_predict_epoch_start

````

````{py:method} on_predict_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_predict_epoch_end

````

````{py:method} on_validation_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_validation_batch_start

````

````{py:method} on_validation_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_validation_batch_end

````

````{py:method} on_test_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_test_batch_start

````

````{py:method} on_test_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_test_batch_end

````

````{py:method} on_predict_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_predict_batch_start

````

````{py:method} on_predict_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: typing.Any, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_predict_batch_end

````

````{py:method} on_train_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_train_start

````

````{py:method} on_train_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_train_end

````

````{py:method} on_validation_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_validation_start

````

````{py:method} on_validation_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_validation_end

````

````{py:method} on_test_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_test_start

````

````{py:method} on_test_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_test_end

````

````{py:method} on_predict_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_predict_start

````

````{py:method} on_predict_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_predict_end

````

````{py:method} on_exception(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, exception: BaseException) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_exception

````

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: scmodelforge.training.callbacks.GradientNormLogger.state_dict

````

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.load_state_dict

````

````{py:method} on_save_checkpoint(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, checkpoint: dict[str, typing.Any]) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_save_checkpoint

````

````{py:method} on_load_checkpoint(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, checkpoint: dict[str, typing.Any]) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_load_checkpoint

````

````{py:method} on_before_backward(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, loss: torch.Tensor) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_before_backward

````

````{py:method} on_after_backward(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_after_backward

````

````{py:method} on_before_zero_grad(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, optimizer: torch.optim.Optimizer) -> None
:canonical: scmodelforge.training.callbacks.GradientNormLogger.on_before_zero_grad

````

`````
