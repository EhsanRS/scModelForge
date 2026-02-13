# {py:mod}`scmodelforge.models.components.embeddings`

```{py:module} scmodelforge.models.components.embeddings
```

```{autodoc2-docstring} scmodelforge.models.components.embeddings
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GeneExpressionEmbedding <scmodelforge.models.components.embeddings.GeneExpressionEmbedding>`
  - ```{autodoc2-docstring} scmodelforge.models.components.embeddings.GeneExpressionEmbedding
    :summary:
    ```
````

### API

`````{py:class} GeneExpressionEmbedding(vocab_size: int, hidden_dim: int, max_seq_len: int = 2048, dropout: float = 0.1, *, use_expression_values: bool = True, layer_norm_eps: float = 1e-12)
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} scmodelforge.models.components.embeddings.GeneExpressionEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.models.components.embeddings.GeneExpressionEmbedding.__init__
```

````{py:method} forward(input_ids: torch.Tensor, values: torch.Tensor | None = None) -> torch.Tensor
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.forward

```{autodoc2-docstring} scmodelforge.models.components.embeddings.GeneExpressionEmbedding.forward
```

````

````{py:attribute} dump_patches
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.apply

````

````{py:method} cuda(device: int | torch.nn.modules.module.Module.cuda.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.cuda

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.mtia

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.cpu

````

````{py:method} type(dst_type: torch.dtype | str) -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.half

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.to_empty

````

````{py:method} to(*args, **kwargs)
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.to

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.models.components.embeddings.GeneExpressionEmbedding.compile

````

`````
