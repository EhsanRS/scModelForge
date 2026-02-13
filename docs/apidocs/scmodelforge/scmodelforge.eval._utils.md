---
orphan: true
---

# {py:mod}`scmodelforge.eval._utils`

```{py:module} scmodelforge.eval._utils
```

```{autodoc2-docstring} scmodelforge.eval._utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`extract_embeddings <scmodelforge.eval._utils.extract_embeddings>`
  - ```{autodoc2-docstring} scmodelforge.eval._utils.extract_embeddings
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.eval._utils.logger>`
  - ```{autodoc2-docstring} scmodelforge.eval._utils.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.eval._utils.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.eval._utils.logger
```

````

````{py:function} extract_embeddings(model: torch.nn.Module, adata: anndata.AnnData, tokenizer: scmodelforge.tokenizers.base.BaseTokenizer, batch_size: int = 256, device: str = 'cpu') -> numpy.ndarray
:canonical: scmodelforge.eval._utils.extract_embeddings

```{autodoc2-docstring} scmodelforge.eval._utils.extract_embeddings
```
````
