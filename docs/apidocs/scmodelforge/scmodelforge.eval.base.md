# {py:mod}`scmodelforge.eval.base`

```{py:module} scmodelforge.eval.base
```

```{autodoc2-docstring} scmodelforge.eval.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BenchmarkResult <scmodelforge.eval.base.BenchmarkResult>`
  - ```{autodoc2-docstring} scmodelforge.eval.base.BenchmarkResult
    :summary:
    ```
* - {py:obj}`BaseBenchmark <scmodelforge.eval.base.BaseBenchmark>`
  - ```{autodoc2-docstring} scmodelforge.eval.base.BaseBenchmark
    :summary:
    ```
````

### API

`````{py:class} BenchmarkResult
:canonical: scmodelforge.eval.base.BenchmarkResult

```{autodoc2-docstring} scmodelforge.eval.base.BenchmarkResult
```

````{py:attribute} benchmark_name
:canonical: scmodelforge.eval.base.BenchmarkResult.benchmark_name
:type: str
:value: >
   None

```{autodoc2-docstring} scmodelforge.eval.base.BenchmarkResult.benchmark_name
```

````

````{py:attribute} dataset_name
:canonical: scmodelforge.eval.base.BenchmarkResult.dataset_name
:type: str
:value: >
   None

```{autodoc2-docstring} scmodelforge.eval.base.BenchmarkResult.dataset_name
```

````

````{py:attribute} metrics
:canonical: scmodelforge.eval.base.BenchmarkResult.metrics
:type: dict[str, float]
:value: >
   None

```{autodoc2-docstring} scmodelforge.eval.base.BenchmarkResult.metrics
```

````

````{py:attribute} metadata
:canonical: scmodelforge.eval.base.BenchmarkResult.metadata
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.eval.base.BenchmarkResult.metadata
```

````

````{py:method} to_dict() -> dict[str, typing.Any]
:canonical: scmodelforge.eval.base.BenchmarkResult.to_dict

```{autodoc2-docstring} scmodelforge.eval.base.BenchmarkResult.to_dict
```

````

````{py:method} summary() -> str
:canonical: scmodelforge.eval.base.BenchmarkResult.summary

```{autodoc2-docstring} scmodelforge.eval.base.BenchmarkResult.summary
```

````

`````

`````{py:class} BaseBenchmark
:canonical: scmodelforge.eval.base.BaseBenchmark

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} scmodelforge.eval.base.BaseBenchmark
```

````{py:property} name
:canonical: scmodelforge.eval.base.BaseBenchmark.name
:abstractmethod:
:type: str

```{autodoc2-docstring} scmodelforge.eval.base.BaseBenchmark.name
```

````

````{py:property} required_obs_keys
:canonical: scmodelforge.eval.base.BaseBenchmark.required_obs_keys
:abstractmethod:
:type: list[str]

```{autodoc2-docstring} scmodelforge.eval.base.BaseBenchmark.required_obs_keys
```

````

````{py:method} run(embeddings: numpy.ndarray, adata: anndata.AnnData, dataset_name: str) -> scmodelforge.eval.base.BenchmarkResult
:canonical: scmodelforge.eval.base.BaseBenchmark.run
:abstractmethod:

```{autodoc2-docstring} scmodelforge.eval.base.BaseBenchmark.run
```

````

````{py:method} validate_adata(adata: anndata.AnnData) -> None
:canonical: scmodelforge.eval.base.BaseBenchmark.validate_adata

```{autodoc2-docstring} scmodelforge.eval.base.BaseBenchmark.validate_adata
```

````

`````
