# {py:mod}`scmodelforge.eval.callback`

```{py:module} scmodelforge.eval.callback
```

```{autodoc2-docstring} scmodelforge.eval.callback
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AssessmentCallback <scmodelforge.eval.callback.AssessmentCallback>`
  - ```{autodoc2-docstring} scmodelforge.eval.callback.AssessmentCallback
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.eval.callback.logger>`
  - ```{autodoc2-docstring} scmodelforge.eval.callback.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.eval.callback.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.eval.callback.logger
```

````

`````{py:class} AssessmentCallback(config: scmodelforge.config.schema.EvalConfig, datasets: dict[str, anndata.AnnData], tokenizer: scmodelforge.tokenizers.base.BaseTokenizer, batch_size: int | None = None, device: str | None = None)
:canonical: scmodelforge.eval.callback.AssessmentCallback

Bases: {py:obj}`lightning.pytorch.Callback`

```{autodoc2-docstring} scmodelforge.eval.callback.AssessmentCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.eval.callback.AssessmentCallback.__init__
```

````{py:method} on_validation_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_validation_epoch_end

```{autodoc2-docstring} scmodelforge.eval.callback.AssessmentCallback.on_validation_epoch_end
```

````

````{py:property} state_key
:canonical: scmodelforge.eval.callback.AssessmentCallback.state_key
:type: str

````

````{py:method} setup(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, stage: str) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.setup

````

````{py:method} teardown(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, stage: str) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.teardown

````

````{py:method} on_fit_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_fit_start

````

````{py:method} on_fit_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_fit_end

````

````{py:method} on_sanity_check_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_sanity_check_start

````

````{py:method} on_sanity_check_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_sanity_check_end

````

````{py:method} on_train_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_train_batch_start

````

````{py:method} on_train_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_train_batch_end

````

````{py:method} on_train_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_train_epoch_start

````

````{py:method} on_train_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_train_epoch_end

````

````{py:method} on_validation_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_validation_epoch_start

````

````{py:method} on_test_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_test_epoch_start

````

````{py:method} on_test_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_test_epoch_end

````

````{py:method} on_predict_epoch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_predict_epoch_start

````

````{py:method} on_predict_epoch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_predict_epoch_end

````

````{py:method} on_validation_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_validation_batch_start

````

````{py:method} on_validation_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_validation_batch_end

````

````{py:method} on_test_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_test_batch_start

````

````{py:method} on_test_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_test_batch_end

````

````{py:method} on_predict_batch_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_predict_batch_start

````

````{py:method} on_predict_batch_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, outputs: typing.Any, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_predict_batch_end

````

````{py:method} on_train_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_train_start

````

````{py:method} on_train_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_train_end

````

````{py:method} on_validation_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_validation_start

````

````{py:method} on_validation_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_validation_end

````

````{py:method} on_test_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_test_start

````

````{py:method} on_test_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_test_end

````

````{py:method} on_predict_start(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_predict_start

````

````{py:method} on_predict_end(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_predict_end

````

````{py:method} on_exception(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, exception: BaseException) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_exception

````

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: scmodelforge.eval.callback.AssessmentCallback.state_dict

````

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.load_state_dict

````

````{py:method} on_save_checkpoint(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, checkpoint: dict[str, typing.Any]) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_save_checkpoint

````

````{py:method} on_load_checkpoint(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, checkpoint: dict[str, typing.Any]) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_load_checkpoint

````

````{py:method} on_before_backward(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, loss: torch.Tensor) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_before_backward

````

````{py:method} on_after_backward(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_after_backward

````

````{py:method} on_before_optimizer_step(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, optimizer: torch.optim.Optimizer) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_before_optimizer_step

````

````{py:method} on_before_zero_grad(trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule, optimizer: torch.optim.Optimizer) -> None
:canonical: scmodelforge.eval.callback.AssessmentCallback.on_before_zero_grad

````

`````
