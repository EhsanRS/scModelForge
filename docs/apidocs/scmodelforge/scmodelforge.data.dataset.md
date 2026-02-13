# {py:mod}`scmodelforge.data.dataset`

```{py:module} scmodelforge.data.dataset
```

```{autodoc2-docstring} scmodelforge.data.dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CellDataset <scmodelforge.data.dataset.CellDataset>`
  - ```{autodoc2-docstring} scmodelforge.data.dataset.CellDataset
    :summary:
    ```
````

### API

````{py:class} CellDataset(adata: anndata.AnnData | str | pathlib.Path | list[anndata.AnnData | str | pathlib.Path], gene_vocab: scmodelforge.data.gene_vocab.GeneVocab, preprocessing: scmodelforge.data.preprocessing.PreprocessingPipeline | None = None, obs_keys: list[str] | None = None)
:canonical: scmodelforge.data.dataset.CellDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} scmodelforge.data.dataset.CellDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.data.dataset.CellDataset.__init__
```

````
