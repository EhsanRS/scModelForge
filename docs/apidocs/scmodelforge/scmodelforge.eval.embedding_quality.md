# {py:mod}`scmodelforge.eval.embedding_quality`

```{py:module} scmodelforge.eval.embedding_quality
```

```{autodoc2-docstring} scmodelforge.eval.embedding_quality
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EmbeddingQualityBenchmark <scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark>`
  - ```{autodoc2-docstring} scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.eval.embedding_quality.logger>`
  - ```{autodoc2-docstring} scmodelforge.eval.embedding_quality.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.eval.embedding_quality.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.eval.embedding_quality.logger
```

````

`````{py:class} EmbeddingQualityBenchmark(cell_type_key: str = 'cell_type', batch_key: str | None = 'batch', n_neighbors: int = 15)
:canonical: scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark

Bases: {py:obj}`scmodelforge.eval.base.BaseBenchmark`

```{autodoc2-docstring} scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark.__init__
```

````{py:property} name
:canonical: scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark.name
:type: str

````

````{py:property} required_obs_keys
:canonical: scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark.required_obs_keys
:type: list[str]

````

````{py:method} run(embeddings: numpy.ndarray, adata: anndata.AnnData, dataset_name: str) -> scmodelforge.eval.base.BenchmarkResult
:canonical: scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark.run

```{autodoc2-docstring} scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark.run
```

````

````{py:method} validate_adata(adata: anndata.AnnData) -> None
:canonical: scmodelforge.eval.embedding_quality.EmbeddingQualityBenchmark.validate_adata

````

`````
