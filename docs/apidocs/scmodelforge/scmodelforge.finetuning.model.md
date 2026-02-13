# {py:mod}`scmodelforge.finetuning.model`

```{py:module} scmodelforge.finetuning.model
```

```{autodoc2-docstring} scmodelforge.finetuning.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FineTuneModel <scmodelforge.finetuning.model.FineTuneModel>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.model.FineTuneModel
    :summary:
    ```
````

### API

`````{py:class} FineTuneModel(backbone: torch.nn.Module, head: torch.nn.Module, task: str = 'classification', freeze_backbone: bool = False)
:canonical: scmodelforge.finetuning.model.FineTuneModel

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} scmodelforge.finetuning.model.FineTuneModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.finetuning.model.FineTuneModel.__init__
```

````{py:method} forward(input_ids: torch.Tensor, attention_mask: torch.Tensor, values: torch.Tensor | None = None, labels: torch.Tensor | None = None, **kwargs: typing.Any) -> scmodelforge.models.protocol.ModelOutput
:canonical: scmodelforge.finetuning.model.FineTuneModel.forward

```{autodoc2-docstring} scmodelforge.finetuning.model.FineTuneModel.forward
```

````

````{py:method} encode(input_ids: torch.Tensor, attention_mask: torch.Tensor, values: torch.Tensor | None = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: scmodelforge.finetuning.model.FineTuneModel.encode

```{autodoc2-docstring} scmodelforge.finetuning.model.FineTuneModel.encode
```

````

````{py:property} has_lora
:canonical: scmodelforge.finetuning.model.FineTuneModel.has_lora
:type: bool

```{autodoc2-docstring} scmodelforge.finetuning.model.FineTuneModel.has_lora
```

````

````{py:method} freeze_backbone() -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.freeze_backbone

```{autodoc2-docstring} scmodelforge.finetuning.model.FineTuneModel.freeze_backbone
```

````

````{py:method} unfreeze_backbone() -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.unfreeze_backbone

```{autodoc2-docstring} scmodelforge.finetuning.model.FineTuneModel.unfreeze_backbone
```

````

````{py:method} num_parameters(*, trainable_only: bool = True) -> int
:canonical: scmodelforge.finetuning.model.FineTuneModel.num_parameters

```{autodoc2-docstring} scmodelforge.finetuning.model.FineTuneModel.num_parameters
```

````

````{py:attribute} dump_patches
:canonical: scmodelforge.finetuning.model.FineTuneModel.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.finetuning.model.FineTuneModel.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.finetuning.model.FineTuneModel.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.finetuning.model.FineTuneModel.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.finetuning.model.FineTuneModel.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.finetuning.model.FineTuneModel.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.finetuning.model.FineTuneModel.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.apply

````

````{py:method} cuda(device: int | torch.nn.modules.module.Module.cuda.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.cuda

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.mtia

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.cpu

````

````{py:method} type(dst_type: torch.dtype | str) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.half

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.to_empty

````

````{py:method} to(*args, **kwargs)
:canonical: scmodelforge.finetuning.model.FineTuneModel.to

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.finetuning.model.FineTuneModel.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.finetuning.model.FineTuneModel.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.finetuning.model.FineTuneModel.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.finetuning.model.FineTuneModel.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.finetuning.model.FineTuneModel.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.finetuning.model.FineTuneModel.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.finetuning.model.FineTuneModel.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.finetuning.model.FineTuneModel.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.finetuning.model.FineTuneModel.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.finetuning.model.FineTuneModel.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.finetuning.model.FineTuneModel.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.finetuning.model.FineTuneModel.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.model.FineTuneModel.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.finetuning.model.FineTuneModel.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.finetuning.model.FineTuneModel.compile

````

`````
