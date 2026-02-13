# {py:mod}`scmodelforge.models.transformer_encoder`

```{py:module} scmodelforge.models.transformer_encoder
```

```{autodoc2-docstring} scmodelforge.models.transformer_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TransformerEncoder <scmodelforge.models.transformer_encoder.TransformerEncoder>`
  - ```{autodoc2-docstring} scmodelforge.models.transformer_encoder.TransformerEncoder
    :summary:
    ```
````

### API

`````{py:class} TransformerEncoder(vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int, ffn_dim: int | None = None, dropout: float = 0.1, max_seq_len: int = 2048, pooling: str = 'cls', activation: str = 'gelu', *, use_expression_values: bool = True, layer_norm_eps: float = 1e-12)
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} scmodelforge.models.transformer_encoder.TransformerEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.models.transformer_encoder.TransformerEncoder.__init__
```

````{py:method} forward(input_ids: torch.Tensor, attention_mask: torch.Tensor, values: torch.Tensor | None = None, labels: torch.Tensor | None = None, **kwargs: typing.Any) -> scmodelforge.models.protocol.ModelOutput
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.forward

```{autodoc2-docstring} scmodelforge.models.transformer_encoder.TransformerEncoder.forward
```

````

````{py:method} encode(input_ids: torch.Tensor, attention_mask: torch.Tensor, values: torch.Tensor | None = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.encode

```{autodoc2-docstring} scmodelforge.models.transformer_encoder.TransformerEncoder.encode
```

````

````{py:method} from_config(config: scmodelforge.config.schema.ModelConfig) -> scmodelforge.models.transformer_encoder.TransformerEncoder
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.from_config
:classmethod:

```{autodoc2-docstring} scmodelforge.models.transformer_encoder.TransformerEncoder.from_config
```

````

````{py:method} num_parameters(*, trainable_only: bool = True) -> int
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.num_parameters

```{autodoc2-docstring} scmodelforge.models.transformer_encoder.TransformerEncoder.num_parameters
```

````

````{py:attribute} dump_patches
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.dump_patches
:type: bool
:value: >
   False

````

````{py:attribute} training
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.training
:type: bool
:value: >
   None

````

````{py:attribute} call_super_init
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.call_super_init
:type: bool
:value: >
   False

````

````{py:method} register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_buffer

````

````{py:method} register_parameter(name: str, param: torch.nn.parameter.Parameter | None) -> None
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_parameter

````

````{py:method} add_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.add_module

````

````{py:method} register_module(name: str, module: typing.Optional[torch.nn.modules.module.Module]) -> None
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_module

````

````{py:method} get_submodule(target: str) -> torch.nn.modules.module.Module
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.get_submodule

````

````{py:method} set_submodule(target: str, module: torch.nn.modules.module.Module, strict: bool = False) -> None
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.set_submodule

````

````{py:method} get_parameter(target: str) -> torch.nn.parameter.Parameter
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.get_parameter

````

````{py:method} get_buffer(target: str) -> torch.Tensor
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.get_buffer

````

````{py:method} get_extra_state() -> typing.Any
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.get_extra_state

````

````{py:method} set_extra_state(state: typing.Any) -> None
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.set_extra_state

````

````{py:method} apply(fn: collections.abc.Callable[[torch.nn.modules.module.Module], None]) -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.apply

````

````{py:method} cuda(device: int | torch.nn.modules.module.Module.cuda.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.cuda

````

````{py:method} ipu(device: int | torch.nn.modules.module.Module.ipu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.ipu

````

````{py:method} xpu(device: int | torch.nn.modules.module.Module.xpu.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.xpu

````

````{py:method} mtia(device: int | torch.nn.modules.module.Module.mtia.device | None = None) -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.mtia

````

````{py:method} cpu() -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.cpu

````

````{py:method} type(dst_type: torch.dtype | str) -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.type

````

````{py:method} float() -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.float

````

````{py:method} double() -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.double

````

````{py:method} half() -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.half

````

````{py:method} bfloat16() -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.bfloat16

````

````{py:method} to_empty(*, device: torch._prims_common.DeviceLikeType | None, recurse: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.to_empty

````

````{py:method} to(*args, **kwargs)
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.to

````

````{py:method} register_full_backward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_full_backward_pre_hook

````

````{py:method} register_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t]) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_backward_hook

````

````{py:method} register_full_backward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.Module, torch.nn.modules.module._grad_t, torch.nn.modules.module._grad_t], None | torch.nn.modules.module._grad_t], prepend: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_full_backward_hook

````

````{py:method} register_forward_pre_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...]], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any]], tuple[typing.Any, dict[str, typing.Any]] | None], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_forward_pre_hook

````

````{py:method} register_forward_hook(hook: collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], typing.Any], typing.Any | None] | collections.abc.Callable[[torch.nn.modules.module.T, tuple[typing.Any, ...], dict[str, typing.Any], typing.Any], typing.Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_forward_hook

````

````{py:method} register_state_dict_post_hook(hook)
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_state_dict_post_hook

````

````{py:method} register_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_state_dict_pre_hook

````

````{py:attribute} T_destination
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.T_destination
:value: >
   'TypeVar(...)'

````

````{py:method} state_dict(*args, destination=None, prefix='', keep_vars=False)
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.state_dict

````

````{py:method} register_load_state_dict_pre_hook(hook)
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_load_state_dict_pre_hook

````

````{py:method} register_load_state_dict_post_hook(hook)
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.register_load_state_dict_post_hook

````

````{py:method} load_state_dict(state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.load_state_dict

````

````{py:method} parameters(recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.parameters

````

````{py:method} named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.named_parameters

````

````{py:method} buffers(recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.buffers

````

````{py:method} named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.named_buffers

````

````{py:method} children() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.children

````

````{py:method} named_children() -> collections.abc.Iterator[tuple[str, torch.nn.modules.module.Module]]
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.named_children

````

````{py:method} modules() -> collections.abc.Iterator[torch.nn.modules.module.Module]
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.modules

````

````{py:method} named_modules(memo: set[torch.nn.modules.module.Module] | None = None, prefix: str = '', remove_duplicate: bool = True)
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.named_modules

````

````{py:method} train(mode: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.train

````

````{py:method} eval() -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.eval

````

````{py:method} requires_grad_(requires_grad: bool = True) -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.requires_grad_

````

````{py:method} zero_grad(set_to_none: bool = True) -> None
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.zero_grad

````

````{py:method} share_memory() -> typing_extensions.Self
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.share_memory

````

````{py:method} extra_repr() -> str
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.extra_repr

````

````{py:method} compile(*args, **kwargs) -> None
:canonical: scmodelforge.models.transformer_encoder.TransformerEncoder.compile

````

`````
