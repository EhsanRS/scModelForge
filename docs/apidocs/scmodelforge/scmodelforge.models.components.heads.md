# {py:mod}`scmodelforge.models.components.heads`

```{py:module} scmodelforge.models.components.heads
```

```{autodoc2-docstring} scmodelforge.models.components.heads
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MaskedGenePredictionHead <scmodelforge.models.components.heads.MaskedGenePredictionHead>`
  - ```{autodoc2-docstring} scmodelforge.models.components.heads.MaskedGenePredictionHead
    :summary:
    ```
* - {py:obj}`BinPredictionHead <scmodelforge.models.components.heads.BinPredictionHead>`
  - ```{autodoc2-docstring} scmodelforge.models.components.heads.BinPredictionHead
    :summary:
    ```
* - {py:obj}`ExpressionPredictionHead <scmodelforge.models.components.heads.ExpressionPredictionHead>`
  - ```{autodoc2-docstring} scmodelforge.models.components.heads.ExpressionPredictionHead
    :summary:
    ```
````

### API

`````{py:class} MaskedGenePredictionHead(hidden_dim: int, vocab_size: int, layer_norm_eps: float = 1e-12)
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} scmodelforge.models.components.heads.MaskedGenePredictionHead
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.models.components.heads.MaskedGenePredictionHead.__init__
```

````{py:method} forward(hidden_states: torch.Tensor) -> torch.Tensor
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.forward

```{autodoc2-docstring} scmodelforge.models.components.heads.MaskedGenePredictionHead.forward
```

````

````{py:attribute} dump_patches
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.apply

````

````{py:method} cuda(device: int | torch.nn.modules.module.Module.cuda.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.cuda

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.mtia

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.cpu

````

````{py:method} type(dst_type: torch.dtype | str) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.half

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.to_empty

````

````{py:method} to(*args, **kwargs)
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.to

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.models.components.heads.MaskedGenePredictionHead.compile

````

`````

`````{py:class} BinPredictionHead(hidden_dim: int, n_bins: int, layer_norm_eps: float = 1e-12)
:canonical: scmodelforge.models.components.heads.BinPredictionHead

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} scmodelforge.models.components.heads.BinPredictionHead
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.models.components.heads.BinPredictionHead.__init__
```

````{py:method} forward(hidden_states: torch.Tensor) -> torch.Tensor
:canonical: scmodelforge.models.components.heads.BinPredictionHead.forward

```{autodoc2-docstring} scmodelforge.models.components.heads.BinPredictionHead.forward
```

````

````{py:attribute} dump_patches
:canonical: scmodelforge.models.components.heads.BinPredictionHead.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.models.components.heads.BinPredictionHead.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.models.components.heads.BinPredictionHead.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.components.heads.BinPredictionHead.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.models.components.heads.BinPredictionHead.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.models.components.heads.BinPredictionHead.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.models.components.heads.BinPredictionHead.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.models.components.heads.BinPredictionHead.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.models.components.heads.BinPredictionHead.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.models.components.heads.BinPredictionHead.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.apply

````

````{py:method} cuda(device: int | torch.nn.modules.module.Module.cuda.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.cuda

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.mtia

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.cpu

````

````{py:method} type(dst_type: torch.dtype | str) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.half

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.to_empty

````

````{py:method} to(*args, **kwargs)
:canonical: scmodelforge.models.components.heads.BinPredictionHead.to

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.models.components.heads.BinPredictionHead.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.models.components.heads.BinPredictionHead.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.models.components.heads.BinPredictionHead.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.models.components.heads.BinPredictionHead.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.models.components.heads.BinPredictionHead.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.models.components.heads.BinPredictionHead.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.models.components.heads.BinPredictionHead.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.models.components.heads.BinPredictionHead.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.components.heads.BinPredictionHead.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.models.components.heads.BinPredictionHead.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.components.heads.BinPredictionHead.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.models.components.heads.BinPredictionHead.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.models.components.heads.BinPredictionHead.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.BinPredictionHead.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.models.components.heads.BinPredictionHead.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.models.components.heads.BinPredictionHead.compile

````

`````

`````{py:class} ExpressionPredictionHead(hidden_dim: int)
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} scmodelforge.models.components.heads.ExpressionPredictionHead
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.models.components.heads.ExpressionPredictionHead.__init__
```

````{py:method} forward(hidden_states: torch.Tensor) -> torch.Tensor
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.forward

```{autodoc2-docstring} scmodelforge.models.components.heads.ExpressionPredictionHead.forward
```

````

````{py:attribute} dump_patches
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.apply

````

````{py:method} cuda(device: int | torch.nn.modules.module.Module.cuda.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.cuda

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.mtia

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.cpu

````

````{py:method} type(dst_type: torch.dtype | str) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.half

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.to_empty

````

````{py:method} to(*args, **kwargs)
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.to

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.models.components.heads.ExpressionPredictionHead.compile

````

`````
