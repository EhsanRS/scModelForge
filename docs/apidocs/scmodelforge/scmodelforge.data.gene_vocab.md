# {py:mod}`scmodelforge.data.gene_vocab`

```{py:module} scmodelforge.data.gene_vocab
```

```{autodoc2-docstring} scmodelforge.data.gene_vocab
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GeneVocab <scmodelforge.data.gene_vocab.GeneVocab>`
  - ```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.data.gene_vocab.logger>`
  - ```{autodoc2-docstring} scmodelforge.data.gene_vocab.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.data.gene_vocab.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.data.gene_vocab.logger
```

````

`````{py:class} GeneVocab(gene_to_idx: dict[str, int])
:canonical: scmodelforge.data.gene_vocab.GeneVocab

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.__init__
```

````{py:method} from_genes(genes: list[str] | numpy.ndarray) -> scmodelforge.data.gene_vocab.GeneVocab
:canonical: scmodelforge.data.gene_vocab.GeneVocab.from_genes
:classmethod:

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.from_genes
```

````

````{py:method} from_adata(adata: anndata.AnnData, key: str = 'var_names') -> scmodelforge.data.gene_vocab.GeneVocab
:canonical: scmodelforge.data.gene_vocab.GeneVocab.from_adata
:classmethod:

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.from_adata
```

````

````{py:method} from_file(path: str | pathlib.Path) -> scmodelforge.data.gene_vocab.GeneVocab
:canonical: scmodelforge.data.gene_vocab.GeneVocab.from_file
:classmethod:

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.from_file
```

````

````{py:method} save(path: str | pathlib.Path) -> None
:canonical: scmodelforge.data.gene_vocab.GeneVocab.save

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.save
```

````

````{py:method} encode(gene_names: list[str] | numpy.ndarray) -> numpy.ndarray
:canonical: scmodelforge.data.gene_vocab.GeneVocab.encode

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.encode
```

````

````{py:method} decode(indices: list[int] | numpy.ndarray) -> list[str]
:canonical: scmodelforge.data.gene_vocab.GeneVocab.decode

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.decode
```

````

````{py:method} get_alignment_indices(gene_names: list[str] | numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
:canonical: scmodelforge.data.gene_vocab.GeneVocab.get_alignment_indices

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.get_alignment_indices
```

````

````{py:property} pad_token_id
:canonical: scmodelforge.data.gene_vocab.GeneVocab.pad_token_id
:type: int

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.pad_token_id
```

````

````{py:property} unk_token_id
:canonical: scmodelforge.data.gene_vocab.GeneVocab.unk_token_id
:type: int

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.unk_token_id
```

````

````{py:property} mask_token_id
:canonical: scmodelforge.data.gene_vocab.GeneVocab.mask_token_id
:type: int

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.mask_token_id
```

````

````{py:property} cls_token_id
:canonical: scmodelforge.data.gene_vocab.GeneVocab.cls_token_id
:type: int

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.cls_token_id
```

````

````{py:property} genes
:canonical: scmodelforge.data.gene_vocab.GeneVocab.genes
:type: list[str]

```{autodoc2-docstring} scmodelforge.data.gene_vocab.GeneVocab.genes
```

````

`````
