---
orphan: true
---

# {py:mod}`scmodelforge.data._utils`

```{py:module} scmodelforge.data._utils
```

```{autodoc2-docstring} scmodelforge.data._utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sparse_to_dense <scmodelforge.data._utils.sparse_to_dense>`
  - ```{autodoc2-docstring} scmodelforge.data._utils.sparse_to_dense
    :summary:
    ```
* - {py:obj}`get_row_as_dense <scmodelforge.data._utils.get_row_as_dense>`
  - ```{autodoc2-docstring} scmodelforge.data._utils.get_row_as_dense
    :summary:
    ```
* - {py:obj}`get_nonzero_indices_and_values <scmodelforge.data._utils.get_nonzero_indices_and_values>`
  - ```{autodoc2-docstring} scmodelforge.data._utils.get_nonzero_indices_and_values
    :summary:
    ```
* - {py:obj}`collate_cells <scmodelforge.data._utils.collate_cells>`
  - ```{autodoc2-docstring} scmodelforge.data._utils.collate_cells
    :summary:
    ```
* - {py:obj}`load_adata <scmodelforge.data._utils.load_adata>`
  - ```{autodoc2-docstring} scmodelforge.data._utils.load_adata
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.data._utils.logger>`
  - ```{autodoc2-docstring} scmodelforge.data._utils.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.data._utils.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.data._utils.logger
```

````

````{py:function} sparse_to_dense(x: scipy.sparse.spmatrix | numpy.ndarray) -> numpy.ndarray
:canonical: scmodelforge.data._utils.sparse_to_dense

```{autodoc2-docstring} scmodelforge.data._utils.sparse_to_dense
```
````

````{py:function} get_row_as_dense(X: scipy.sparse.spmatrix | numpy.ndarray, idx: int) -> numpy.ndarray
:canonical: scmodelforge.data._utils.get_row_as_dense

```{autodoc2-docstring} scmodelforge.data._utils.get_row_as_dense
```
````

````{py:function} get_nonzero_indices_and_values(expression: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
:canonical: scmodelforge.data._utils.get_nonzero_indices_and_values

```{autodoc2-docstring} scmodelforge.data._utils.get_nonzero_indices_and_values
```
````

````{py:function} collate_cells(batch: list[dict[str, typing.Any]], pad_value: int = 0) -> dict[str, typing.Any]
:canonical: scmodelforge.data._utils.collate_cells

```{autodoc2-docstring} scmodelforge.data._utils.collate_cells
```
````

````{py:function} load_adata(data_config: scmodelforge.config.schema.DataConfig, adata: typing.Any | None = None, obs_keys: list[str] | None = None) -> anndata.AnnData
:canonical: scmodelforge.data._utils.load_adata

```{autodoc2-docstring} scmodelforge.data._utils.load_adata
```
````
