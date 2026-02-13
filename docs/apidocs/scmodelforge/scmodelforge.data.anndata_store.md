# {py:mod}`scmodelforge.data.anndata_store`

```{py:module} scmodelforge.data.anndata_store
```

```{autodoc2-docstring} scmodelforge.data.anndata_store
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AnnDataStore <scmodelforge.data.anndata_store.AnnDataStore>`
  - ```{autodoc2-docstring} scmodelforge.data.anndata_store.AnnDataStore
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.data.anndata_store.logger>`
  - ```{autodoc2-docstring} scmodelforge.data.anndata_store.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.data.anndata_store.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.data.anndata_store.logger
```

````

`````{py:class} AnnDataStore(adatas: list[anndata.AnnData | str | pathlib.Path], gene_vocab: scmodelforge.data.gene_vocab.GeneVocab, obs_keys: list[str] | None = None)
:canonical: scmodelforge.data.anndata_store.AnnDataStore

```{autodoc2-docstring} scmodelforge.data.anndata_store.AnnDataStore
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.data.anndata_store.AnnDataStore.__init__
```

````{py:method} get_cell(global_idx: int) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, str]]
:canonical: scmodelforge.data.anndata_store.AnnDataStore.get_cell

```{autodoc2-docstring} scmodelforge.data.anndata_store.AnnDataStore.get_cell
```

````

````{py:property} n_datasets
:canonical: scmodelforge.data.anndata_store.AnnDataStore.n_datasets
:type: int

```{autodoc2-docstring} scmodelforge.data.anndata_store.AnnDataStore.n_datasets
```

````

`````
