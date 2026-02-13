---
orphan: true
---

# {py:mod}`scmodelforge._types`

```{py:module} scmodelforge._types
```

```{autodoc2-docstring} scmodelforge._types
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizedCellProtocol <scmodelforge._types.TokenizedCellProtocol>`
  - ```{autodoc2-docstring} scmodelforge._types.TokenizedCellProtocol
    :summary:
    ```
* - {py:obj}`ModelProtocol <scmodelforge._types.ModelProtocol>`
  - ```{autodoc2-docstring} scmodelforge._types.ModelProtocol
    :summary:
    ```
````

### API

`````{py:class} TokenizedCellProtocol
:canonical: scmodelforge._types.TokenizedCellProtocol

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} scmodelforge._types.TokenizedCellProtocol
```

````{py:attribute} input_ids
:canonical: scmodelforge._types.TokenizedCellProtocol.input_ids
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} scmodelforge._types.TokenizedCellProtocol.input_ids
```

````

````{py:attribute} attention_mask
:canonical: scmodelforge._types.TokenizedCellProtocol.attention_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} scmodelforge._types.TokenizedCellProtocol.attention_mask
```

````

````{py:attribute} metadata
:canonical: scmodelforge._types.TokenizedCellProtocol.metadata
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} scmodelforge._types.TokenizedCellProtocol.metadata
```

````

`````

`````{py:class} ModelProtocol
:canonical: scmodelforge._types.ModelProtocol

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} scmodelforge._types.ModelProtocol
```

````{py:method} forward(input_ids: torch.Tensor, attention_mask: torch.Tensor, values: torch.Tensor | None = None, labels: torch.Tensor | None = None, **kwargs: typing.Any) -> typing.Any
:canonical: scmodelforge._types.ModelProtocol.forward

```{autodoc2-docstring} scmodelforge._types.ModelProtocol.forward
```

````

````{py:method} encode(input_ids: torch.Tensor, attention_mask: torch.Tensor, values: torch.Tensor | None = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: scmodelforge._types.ModelProtocol.encode

```{autodoc2-docstring} scmodelforge._types.ModelProtocol.encode
```

````

`````
