# {py:mod}`scmodelforge.finetuning.adapters`

```{py:module} scmodelforge.finetuning.adapters
```

```{autodoc2-docstring} scmodelforge.finetuning.adapters
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_lora <scmodelforge.finetuning.adapters.apply_lora>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.adapters.apply_lora
    :summary:
    ```
* - {py:obj}`has_lora <scmodelforge.finetuning.adapters.has_lora>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.adapters.has_lora
    :summary:
    ```
* - {py:obj}`save_lora_weights <scmodelforge.finetuning.adapters.save_lora_weights>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.adapters.save_lora_weights
    :summary:
    ```
* - {py:obj}`load_lora_weights <scmodelforge.finetuning.adapters.load_lora_weights>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.adapters.load_lora_weights
    :summary:
    ```
* - {py:obj}`count_lora_parameters <scmodelforge.finetuning.adapters.count_lora_parameters>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.adapters.count_lora_parameters
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DEFAULT_TARGET_MODULES <scmodelforge.finetuning.adapters.DEFAULT_TARGET_MODULES>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.adapters.DEFAULT_TARGET_MODULES
    :summary:
    ```
````

### API

````{py:data} DEFAULT_TARGET_MODULES
:canonical: scmodelforge.finetuning.adapters.DEFAULT_TARGET_MODULES
:value: >
   ['out_proj', 'linear1', 'linear2']

```{autodoc2-docstring} scmodelforge.finetuning.adapters.DEFAULT_TARGET_MODULES
```

````

````{py:function} apply_lora(model: torch.nn.Module, config: scmodelforge.config.schema.LoRAConfig) -> torch.nn.Module
:canonical: scmodelforge.finetuning.adapters.apply_lora

```{autodoc2-docstring} scmodelforge.finetuning.adapters.apply_lora
```
````

````{py:function} has_lora(model: torch.nn.Module) -> bool
:canonical: scmodelforge.finetuning.adapters.has_lora

```{autodoc2-docstring} scmodelforge.finetuning.adapters.has_lora
```
````

````{py:function} save_lora_weights(model: torch.nn.Module, path: str) -> None
:canonical: scmodelforge.finetuning.adapters.save_lora_weights

```{autodoc2-docstring} scmodelforge.finetuning.adapters.save_lora_weights
```
````

````{py:function} load_lora_weights(model: torch.nn.Module, path: str) -> torch.nn.Module
:canonical: scmodelforge.finetuning.adapters.load_lora_weights

```{autodoc2-docstring} scmodelforge.finetuning.adapters.load_lora_weights
```
````

````{py:function} count_lora_parameters(model: torch.nn.Module) -> tuple[int, int]
:canonical: scmodelforge.finetuning.adapters.count_lora_parameters

```{autodoc2-docstring} scmodelforge.finetuning.adapters.count_lora_parameters
```
````
