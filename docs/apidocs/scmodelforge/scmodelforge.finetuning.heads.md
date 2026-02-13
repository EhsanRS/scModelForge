# {py:mod}`scmodelforge.finetuning.heads`

```{py:module} scmodelforge.finetuning.heads
```

```{autodoc2-docstring} scmodelforge.finetuning.heads
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ClassificationHead <scmodelforge.finetuning.heads.ClassificationHead>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.heads.ClassificationHead
    :summary:
    ```
* - {py:obj}`RegressionHead <scmodelforge.finetuning.heads.RegressionHead>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.heads.RegressionHead
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_task_head <scmodelforge.finetuning.heads.build_task_head>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.heads.build_task_head
    :summary:
    ```
````

### API

`````{py:class} ClassificationHead(input_dim: int, n_classes: int, hidden_dim: int | None = None, dropout: float = 0.1)
:canonical: scmodelforge.finetuning.heads.ClassificationHead

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} scmodelforge.finetuning.heads.ClassificationHead
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.finetuning.heads.ClassificationHead.__init__
```

````{py:method} forward(embeddings: torch.Tensor) -> torch.Tensor
:canonical: scmodelforge.finetuning.heads.ClassificationHead.forward

```{autodoc2-docstring} scmodelforge.finetuning.heads.ClassificationHead.forward
```

````

````{py:attribute} dump_patches
:canonical: scmodelforge.finetuning.heads.ClassificationHead.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.finetuning.heads.ClassificationHead.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.finetuning.heads.ClassificationHead.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.finetuning.heads.ClassificationHead.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.finetuning.heads.ClassificationHead.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.finetuning.heads.ClassificationHead.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.finetuning.heads.ClassificationHead.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.finetuning.heads.ClassificationHead.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.finetuning.heads.ClassificationHead.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.finetuning.heads.ClassificationHead.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.apply

````

````{py:method} cuda(device: int | torch.nn.modules.module.Module.cuda.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.cuda

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.mtia

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.cpu

````

````{py:method} type(dst_type: torch.dtype | str) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.half

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.to_empty

````

````{py:method} to(*args, **kwargs)
:canonical: scmodelforge.finetuning.heads.ClassificationHead.to

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.finetuning.heads.ClassificationHead.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.finetuning.heads.ClassificationHead.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.finetuning.heads.ClassificationHead.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.finetuning.heads.ClassificationHead.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.finetuning.heads.ClassificationHead.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.finetuning.heads.ClassificationHead.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.finetuning.heads.ClassificationHead.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.finetuning.heads.ClassificationHead.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.finetuning.heads.ClassificationHead.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.finetuning.heads.ClassificationHead.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.finetuning.heads.ClassificationHead.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.finetuning.heads.ClassificationHead.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.finetuning.heads.ClassificationHead.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.ClassificationHead.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.finetuning.heads.ClassificationHead.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.finetuning.heads.ClassificationHead.compile

````

`````

`````{py:class} RegressionHead(input_dim: int, output_dim: int = 1, hidden_dim: int | None = None, dropout: float = 0.1)
:canonical: scmodelforge.finetuning.heads.RegressionHead

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} scmodelforge.finetuning.heads.RegressionHead
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.finetuning.heads.RegressionHead.__init__
```

````{py:method} forward(embeddings: torch.Tensor) -> torch.Tensor
:canonical: scmodelforge.finetuning.heads.RegressionHead.forward

```{autodoc2-docstring} scmodelforge.finetuning.heads.RegressionHead.forward
```

````

````{py:attribute} dump_patches
:canonical: scmodelforge.finetuning.heads.RegressionHead.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.finetuning.heads.RegressionHead.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.finetuning.heads.RegressionHead.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.finetuning.heads.RegressionHead.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.finetuning.heads.RegressionHead.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.finetuning.heads.RegressionHead.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.finetuning.heads.RegressionHead.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.finetuning.heads.RegressionHead.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.finetuning.heads.RegressionHead.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.finetuning.heads.RegressionHead.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.apply

````

````{py:method} cuda(device: int | torch.nn.modules.module.Module.cuda.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.cuda

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.mtia

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.cpu

````

````{py:method} type(dst_type: torch.dtype | str) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.half

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.to_empty

````

````{py:method} to(*args, **kwargs)
:canonical: scmodelforge.finetuning.heads.RegressionHead.to

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.finetuning.heads.RegressionHead.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.finetuning.heads.RegressionHead.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.finetuning.heads.RegressionHead.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.finetuning.heads.RegressionHead.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.finetuning.heads.RegressionHead.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.finetuning.heads.RegressionHead.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.finetuning.heads.RegressionHead.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.finetuning.heads.RegressionHead.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.finetuning.heads.RegressionHead.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.finetuning.heads.RegressionHead.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.finetuning.heads.RegressionHead.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.finetuning.heads.RegressionHead.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.finetuning.heads.RegressionHead.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.finetuning.heads.RegressionHead.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.finetuning.heads.RegressionHead.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.finetuning.heads.RegressionHead.compile

````

`````

````{py:function} build_task_head(config: scmodelforge.config.schema.TaskHeadConfig, input_dim: int) -> torch.nn.Module
:canonical: scmodelforge.finetuning.heads.build_task_head

```{autodoc2-docstring} scmodelforge.finetuning.heads.build_task_head
```
````
