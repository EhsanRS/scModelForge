# {py:mod}`scmodelforge.models.registry`

```{py:module} scmodelforge.models.registry
```

```{autodoc2-docstring} scmodelforge.models.registry
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`register_model <scmodelforge.models.registry.register_model>`
  - ```{autodoc2-docstring} scmodelforge.models.registry.register_model
    :summary:
    ```
* - {py:obj}`get_model <scmodelforge.models.registry.get_model>`
  - ```{autodoc2-docstring} scmodelforge.models.registry.get_model
    :summary:
    ```
* - {py:obj}`list_models <scmodelforge.models.registry.list_models>`
  - ```{autodoc2-docstring} scmodelforge.models.registry.list_models
    :summary:
    ```
````

### API

````{py:function} register_model(name: str)
:canonical: scmodelforge.models.registry.register_model

```{autodoc2-docstring} scmodelforge.models.registry.register_model
```
````

````{py:function} get_model(name: str, config: scmodelforge.config.schema.ModelConfig) -> torch.nn.Module
:canonical: scmodelforge.models.registry.get_model

```{autodoc2-docstring} scmodelforge.models.registry.get_model
```
````

````{py:function} list_models() -> list[str]
:canonical: scmodelforge.models.registry.list_models

```{autodoc2-docstring} scmodelforge.models.registry.list_models
```
````
