# {py:mod}`scmodelforge.tokenizers.registry`

```{py:module} scmodelforge.tokenizers.registry
```

```{autodoc2-docstring} scmodelforge.tokenizers.registry
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`register_tokenizer <scmodelforge.tokenizers.registry.register_tokenizer>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.registry.register_tokenizer
    :summary:
    ```
* - {py:obj}`get_tokenizer <scmodelforge.tokenizers.registry.get_tokenizer>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.registry.get_tokenizer
    :summary:
    ```
* - {py:obj}`list_tokenizers <scmodelforge.tokenizers.registry.list_tokenizers>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.registry.list_tokenizers
    :summary:
    ```
````

### API

````{py:function} register_tokenizer(name: str)
:canonical: scmodelforge.tokenizers.registry.register_tokenizer

```{autodoc2-docstring} scmodelforge.tokenizers.registry.register_tokenizer
```
````

````{py:function} get_tokenizer(name: str, **kwargs: typing.Any) -> scmodelforge.tokenizers.base.BaseTokenizer
:canonical: scmodelforge.tokenizers.registry.get_tokenizer

```{autodoc2-docstring} scmodelforge.tokenizers.registry.get_tokenizer
```
````

````{py:function} list_tokenizers() -> list[str]
:canonical: scmodelforge.tokenizers.registry.list_tokenizers

```{autodoc2-docstring} scmodelforge.tokenizers.registry.list_tokenizers
```
````
