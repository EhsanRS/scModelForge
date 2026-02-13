# {py:mod}`scmodelforge.training.lightning_module`

```{py:module} scmodelforge.training.lightning_module
```

```{autodoc2-docstring} scmodelforge.training.lightning_module
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ScModelForgeLightningModule <scmodelforge.training.lightning_module.ScModelForgeLightningModule>`
  - ```{autodoc2-docstring} scmodelforge.training.lightning_module.ScModelForgeLightningModule
    :summary:
    ```
````

### API

`````{py:class} ScModelForgeLightningModule(model: torch.nn.Module, optimizer_config: scmodelforge.config.schema.OptimizerConfig, scheduler_config: scmodelforge.config.schema.SchedulerConfig | None = None)
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule

Bases: {py:obj}`lightning.pytorch.LightningModule`

```{autodoc2-docstring} scmodelforge.training.lightning_module.ScModelForgeLightningModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.training.lightning_module.ScModelForgeLightningModule.__init__
```

````{py:method} forward(batch: dict[str, torch.Tensor]) -> scmodelforge.models.protocol.ModelOutput
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.forward

```{autodoc2-docstring} scmodelforge.training.lightning_module.ScModelForgeLightningModule.forward
```

````

````{py:method} training_step(batch: dict[str, typing.Any], batch_idx: int) -> torch.Tensor
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.training_step

```{autodoc2-docstring} scmodelforge.training.lightning_module.ScModelForgeLightningModule.training_step
```

````

````{py:method} validation_step(batch: dict[str, typing.Any], batch_idx: int) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.validation_step

```{autodoc2-docstring} scmodelforge.training.lightning_module.ScModelForgeLightningModule.validation_step
```

````

````{py:method} configure_optimizers() -> dict[str, typing.Any]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.configure_optimizers

```{autodoc2-docstring} scmodelforge.training.lightning_module.ScModelForgeLightningModule.configure_optimizers
```

````

````{py:attribute} CHECKPOINT_HYPER_PARAMS_KEY
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.CHECKPOINT_HYPER_PARAMS_KEY
:value: >
   'hyper_parameters'

````

````{py:attribute} CHECKPOINT_HYPER_PARAMS_NAME
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.CHECKPOINT_HYPER_PARAMS_NAME
:value: >
   'hparams_name'

````

````{py:attribute} CHECKPOINT_HYPER_PARAMS_TYPE
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.CHECKPOINT_HYPER_PARAMS_TYPE
:value: >
   'hparams_type'

````

````{py:method} optimizers(use_pl_optimizer: bool = True) -> lightning.pytorch.core.module.MODULE_OPTIMIZERS
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.optimizers

````

````{py:method} lr_schedulers() -> typing.Union[None, list[lightning.pytorch.utilities.types.LRSchedulerPLType], lightning.pytorch.utilities.types.LRSchedulerPLType]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.lr_schedulers

````

````{py:property} trainer
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.trainer
:type: lightning.pytorch.Trainer

````

````{py:property} fabric
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.fabric
:type: typing.Optional[lightning.fabric.Fabric]

````

````{py:property} example_input_array
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.example_input_array
:type: typing.Optional[typing.Union[torch.Tensor, tuple, dict]]

````

````{py:property} current_epoch
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.current_epoch
:type: int

````

````{py:property} global_step
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.global_step
:type: int

````

````{py:property} global_rank
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.global_rank
:type: int

````

````{py:property} local_rank
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.local_rank
:type: int

````

````{py:property} on_gpu
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_gpu
:type: bool

````

````{py:property} automatic_optimization
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.automatic_optimization
:type: bool

````

````{py:property} strict_loading
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.strict_loading
:type: bool

````

````{py:property} logger
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.logger
:type: typing.Optional[typing.Union[lightning.pytorch.loggers.Logger, lightning.fabric.loggers.Logger]]

````

````{py:property} loggers
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.loggers
:type: typing.Union[list[lightning.pytorch.loggers.Logger], list[lightning.fabric.loggers.Logger]]

````

````{py:property} device_mesh
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.device_mesh
:type: typing.Optional[torch.distributed.device_mesh.DeviceMesh]

````

````{py:method} print(*args: typing.Any, **kwargs: typing.Any) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.print

````

````{py:method} log(name: str, value: lightning.pytorch.utilities.types._METRIC, prog_bar: bool = False, logger: typing.Optional[bool] = None, on_step: typing.Optional[bool] = None, on_epoch: typing.Optional[bool] = None, reduce_fx: typing.Union[str, typing.Callable[[typing.Any], typing.Any]] = 'mean', enable_graph: bool = False, sync_dist: bool = False, sync_dist_group: typing.Optional[typing.Any] = None, add_dataloader_idx: bool = True, batch_size: typing.Optional[int] = None, metric_attribute: typing.Optional[str] = None, rank_zero_only: bool = False) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.log

````

````{py:method} log_dict(dictionary: typing.Union[collections.abc.Mapping[str, lightning.pytorch.utilities.types._METRIC], torchmetrics.MetricCollection], prog_bar: bool = False, logger: typing.Optional[bool] = None, on_step: typing.Optional[bool] = None, on_epoch: typing.Optional[bool] = None, reduce_fx: typing.Union[str, typing.Callable[[typing.Any], typing.Any]] = 'mean', enable_graph: bool = False, sync_dist: bool = False, sync_dist_group: typing.Optional[typing.Any] = None, add_dataloader_idx: bool = True, batch_size: typing.Optional[int] = None, rank_zero_only: bool = False) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.log_dict

````

````{py:method} all_gather(data: typing.Union[torch.Tensor, dict, list, tuple], group: typing.Optional[typing.Any] = None, sync_grads: bool = False) -> typing.Union[torch.Tensor, dict, list, tuple]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.all_gather

````

````{py:method} test_step(*args: typing.Any, **kwargs: typing.Any) -> lightning.pytorch.utilities.types.STEP_OUTPUT
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.test_step

````

````{py:method} predict_step(*args: typing.Any, **kwargs: typing.Any) -> typing.Any
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.predict_step

````

````{py:method} configure_callbacks() -> typing.Union[collections.abc.Sequence[lightning.pytorch.callbacks.callback.Callback], lightning.pytorch.callbacks.callback.Callback]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.configure_callbacks

````

````{py:method} manual_backward(loss: torch.Tensor, *args: typing.Any, **kwargs: typing.Any) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.manual_backward

````

````{py:method} backward(loss: torch.Tensor, *args: typing.Any, **kwargs: typing.Any) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.backward

````

````{py:method} toggle_optimizer(optimizer: typing.Union[torch.optim.optimizer.Optimizer, lightning.pytorch.core.optimizer.LightningOptimizer]) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.toggle_optimizer

````

````{py:method} untoggle_optimizer(optimizer: typing.Union[torch.optim.optimizer.Optimizer, lightning.pytorch.core.optimizer.LightningOptimizer]) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.untoggle_optimizer

````

````{py:method} toggled_optimizer(optimizer: typing.Union[torch.optim.optimizer.Optimizer, lightning.pytorch.core.optimizer.LightningOptimizer]) -> collections.abc.Generator
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.toggled_optimizer

````

````{py:method} clip_gradients(optimizer: torch.optim.optimizer.Optimizer, gradient_clip_val: typing.Optional[typing.Union[int, float]] = None, gradient_clip_algorithm: typing.Optional[str] = None) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.clip_gradients

````

````{py:method} configure_gradient_clipping(optimizer: torch.optim.optimizer.Optimizer, gradient_clip_val: typing.Optional[typing.Union[int, float]] = None, gradient_clip_algorithm: typing.Optional[str] = None) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.configure_gradient_clipping

````

````{py:method} lr_scheduler_step(scheduler: lightning.pytorch.utilities.types.LRSchedulerTypeUnion, metric: typing.Optional[typing.Any]) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.lr_scheduler_step

````

````{py:method} optimizer_step(epoch: int, batch_idx: int, optimizer: typing.Union[torch.optim.optimizer.Optimizer, lightning.pytorch.core.optimizer.LightningOptimizer], optimizer_closure: typing.Optional[typing.Callable[[], typing.Any]] = None) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.optimizer_step

````

````{py:method} optimizer_zero_grad(epoch: int, batch_idx: int, optimizer: torch.optim.optimizer.Optimizer) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.optimizer_zero_grad

````

````{py:method} freeze() -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.freeze

````

````{py:method} unfreeze() -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.unfreeze

````

````{py:method} to_onnx(file_path: typing.Union[str, pathlib.Path, io.BytesIO, None] = None, input_sample: typing.Optional[typing.Any] = None, **kwargs: typing.Any) -> typing.Optional[torch.onnx.ONNXProgram]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.to_onnx

````

````{py:method} to_torchscript(file_path: typing.Optional[typing.Union[str, pathlib.Path]] = None, method: typing.Optional[str] = 'script', example_inputs: typing.Optional[typing.Any] = None, **kwargs: typing.Any) -> typing.Union[torch.ScriptModule, dict[str, torch.ScriptModule]]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.to_torchscript

````

````{py:method} to_tensorrt(file_path: typing.Optional[typing.Union[str, pathlib.Path, io.BytesIO]] = None, input_sample: typing.Optional[typing.Any] = None, ir: typing.Literal[default, dynamo, ts] = 'default', output_format: typing.Literal[exported_program, torchscript] = 'exported_program', retrace: bool = False, default_device: typing.Union[str, torch.device] = 'cuda', **compile_kwargs: typing.Any) -> typing.Union[torch.ScriptModule, torch.fx.GraphModule]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.to_tensorrt

````

````{py:method} load_from_checkpoint(checkpoint_path: typing.Union[lightning.fabric.utilities.types._PATH, typing.IO], map_location: lightning.fabric.utilities.types._MAP_LOCATION_TYPE = None, hparams_file: typing.Optional[lightning.fabric.utilities.types._PATH] = None, strict: typing.Optional[bool] = None, weights_only: typing.Optional[bool] = None, **kwargs: typing.Any) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.load_from_checkpoint
:classmethod:

````

````{py:property} dtype
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.dtype
:type: typing.Union[str, torch.dtype]

````

````{py:property} device
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.device
:type: torch.device

````

````{py:method} to(*args: typing.Any, **kwargs: typing.Any) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.to

````

````{py:method} cuda(device: typing.Optional[typing.Union[torch.device, int]] = None) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.cuda

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.cpu

````

````{py:method} type(dst_type: typing.Union[str, torch.dtype]) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.half

````

````{py:attribute} dump_patches
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.apply

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.mtia

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.to_empty

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.compile

````

````{py:method} save_hyperparameters(*args: typing.Any, ignore: typing.Optional[typing.Union[collections.abc.Sequence[str], str]] = None, frame: typing.Optional[types.FrameType] = None, logger: bool = True) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.save_hyperparameters

````

````{py:method} remove_ignored_hparams(ignore_list: list[str]) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.remove_ignored_hparams

````

````{py:property} hparams
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.hparams
:type: typing.Union[lightning.fabric.utilities.data.AttributeDict, collections.abc.MutableMapping]

````

````{py:property} hparams_initial
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.hparams_initial
:type: lightning.fabric.utilities.data.AttributeDict

````

````{py:method} on_fit_start() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_fit_start

````

````{py:method} on_fit_end() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_fit_end

````

````{py:method} on_train_start() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_train_start

````

````{py:method} on_train_end() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_train_end

````

````{py:method} on_validation_start() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_validation_start

````

````{py:method} on_validation_end() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_validation_end

````

````{py:method} on_test_start() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_test_start

````

````{py:method} on_test_end() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_test_end

````

````{py:method} on_predict_start() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_predict_start

````

````{py:method} on_predict_end() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_predict_end

````

````{py:method} on_train_batch_start(batch: typing.Any, batch_idx: int) -> typing.Optional[int]
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_train_batch_start

````

````{py:method} on_train_batch_end(outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_train_batch_end

````

````{py:method} on_validation_batch_start(batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_validation_batch_start

````

````{py:method} on_validation_batch_end(outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_validation_batch_end

````

````{py:method} on_test_batch_start(batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_test_batch_start

````

````{py:method} on_test_batch_end(outputs: lightning.pytorch.utilities.types.STEP_OUTPUT, batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_test_batch_end

````

````{py:method} on_predict_batch_start(batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_predict_batch_start

````

````{py:method} on_predict_batch_end(outputs: typing.Optional[typing.Any], batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_predict_batch_end

````

````{py:method} on_validation_model_zero_grad() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_validation_model_zero_grad

````

````{py:method} on_validation_model_eval() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_validation_model_eval

````

````{py:method} on_validation_model_train() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_validation_model_train

````

````{py:method} on_test_model_eval() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_test_model_eval

````

````{py:method} on_test_model_train() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_test_model_train

````

````{py:method} on_predict_model_eval() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_predict_model_eval

````

````{py:method} on_train_epoch_start() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_train_epoch_start

````

````{py:method} on_train_epoch_end() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_train_epoch_end

````

````{py:method} on_validation_epoch_start() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_validation_epoch_start

````

````{py:method} on_validation_epoch_end() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_validation_epoch_end

````

````{py:method} on_test_epoch_start() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_test_epoch_start

````

````{py:method} on_test_epoch_end() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_test_epoch_end

````

````{py:method} on_predict_epoch_start() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_predict_epoch_start

````

````{py:method} on_predict_epoch_end() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_predict_epoch_end

````

````{py:method} on_before_zero_grad(optimizer: torch.optim.optimizer.Optimizer) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_before_zero_grad

````

````{py:method} on_before_backward(loss: torch.Tensor) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_before_backward

````

````{py:method} on_after_backward() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_after_backward

````

````{py:method} on_before_optimizer_step(optimizer: torch.optim.optimizer.Optimizer) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_before_optimizer_step

````

````{py:method} configure_sharded_model() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.configure_sharded_model

````

````{py:method} configure_model() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.configure_model

````

````{py:method} prepare_data() -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.prepare_data

````

````{py:method} setup(stage: str) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.setup

````

````{py:method} teardown(stage: str) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.teardown

````

````{py:method} train_dataloader() -> lightning.pytorch.utilities.types.TRAIN_DATALOADERS
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.train_dataloader

````

````{py:method} test_dataloader() -> lightning.pytorch.utilities.types.EVAL_DATALOADERS
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.test_dataloader

````

````{py:method} val_dataloader() -> lightning.pytorch.utilities.types.EVAL_DATALOADERS
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.val_dataloader

````

````{py:method} predict_dataloader() -> lightning.pytorch.utilities.types.EVAL_DATALOADERS
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.predict_dataloader

````

````{py:method} transfer_batch_to_device(batch: typing.Any, device: torch.device, dataloader_idx: int) -> typing.Any
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.transfer_batch_to_device

````

````{py:method} on_before_batch_transfer(batch: typing.Any, dataloader_idx: int) -> typing.Any
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_before_batch_transfer

````

````{py:method} on_after_batch_transfer(batch: typing.Any, dataloader_idx: int) -> typing.Any
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_after_batch_transfer

````

````{py:method} on_load_checkpoint(checkpoint: dict[str, typing.Any]) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_load_checkpoint

````

````{py:method} on_save_checkpoint(checkpoint: dict[str, typing.Any]) -> None
:canonical: scmodelforge.training.lightning_module.ScModelForgeLightningModule.on_save_checkpoint

````

`````
