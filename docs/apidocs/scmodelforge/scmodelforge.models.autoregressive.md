# {py:mod}`scmodelforge.models.autoregressive`

```{py:module} scmodelforge.models.autoregressive
```

```{autodoc2-docstring} scmodelforge.models.autoregressive
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AutoregressiveTransformer <scmodelforge.models.autoregressive.AutoregressiveTransformer>`
  - ```{autodoc2-docstring} scmodelforge.models.autoregressive.AutoregressiveTransformer
    :summary:
    ```
````

### API

`````{py:class} AutoregressiveTransformer(vocab_size: int, n_bins: int = 51, hidden_dim: int = 512, num_layers: int = 12, num_heads: int = 8, ffn_dim: int | None = None, dropout: float = 0.1, max_seq_len: int = 2048, pooling: str = 'cls', activation: str = 'gelu', *, use_expression_values: bool = True, layer_norm_eps: float = 1e-12, gene_loss_weight: float = 1.0, expression_loss_weight: float = 1.0)
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} scmodelforge.models.autoregressive.AutoregressiveTransformer
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.models.autoregressive.AutoregressiveTransformer.__init__
```

````{py:method} forward(input_ids: torch.Tensor, attention_mask: torch.Tensor, values: torch.Tensor | None = None, labels: torch.Tensor | None = None, **kwargs: typing.Any) -> scmodelforge.models.protocol.ModelOutput
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.forward

```{autodoc2-docstring} scmodelforge.models.autoregressive.AutoregressiveTransformer.forward
```

````

````{py:method} encode(input_ids: torch.Tensor, attention_mask: torch.Tensor, values: torch.Tensor | None = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.encode

```{autodoc2-docstring} scmodelforge.models.autoregressive.AutoregressiveTransformer.encode
```

````

````{py:method} from_config(config: scmodelforge.config.schema.ModelConfig) -> scmodelforge.models.autoregressive.AutoregressiveTransformer
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.from_config
:classmethod:

```{autodoc2-docstring} scmodelforge.models.autoregressive.AutoregressiveTransformer.from_config
```

````

````{py:method} num_parameters(*, trainable_only: bool = True) -> int
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.num_parameters

```{autodoc2-docstring} scmodelforge.models.autoregressive.AutoregressiveTransformer.num_parameters
```

````

````{py:attribute} dump_patches
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.apply

````

````{py:method} cuda(device: int | torch.nn.modules.module.Module.cuda.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.cuda

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.mtia

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.cpu

````

````{py:method} type(dst_type: torch.dtype | str) -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.half

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.to_empty

````

````{py:method} to(*args, **kwargs)
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.to

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.models.autoregressive.AutoregressiveTransformer.compile

````

`````
