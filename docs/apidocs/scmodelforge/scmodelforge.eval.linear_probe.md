# {py:mod}`scmodelforge.eval.linear_probe`

```{py:module} scmodelforge.eval.linear_probe
```

```{autodoc2-docstring} scmodelforge.eval.linear_probe
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearProbeBenchmark <scmodelforge.eval.linear_probe.LinearProbeBenchmark>`
  - ```{autodoc2-docstring} scmodelforge.eval.linear_probe.LinearProbeBenchmark
    :summary:
    ```
````

### API

`````{py:class} LinearProbeBenchmark(cell_type_key: str = 'cell_type', test_size: float = 0.2, max_iter: int = 1000, seed: int = 42)
:canonical: scmodelforge.eval.linear_probe.LinearProbeBenchmark

Bases: {py:obj}`scmodelforge.eval.base.BaseBenchmark`

```{autodoc2-docstring} scmodelforge.eval.linear_probe.LinearProbeBenchmark
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.eval.linear_probe.LinearProbeBenchmark.__init__
```

````{py:property} name
:canonical: scmodelforge.eval.linear_probe.LinearProbeBenchmark.name
:type: str

````

````{py:property} required_obs_keys
:canonical: scmodelforge.eval.linear_probe.LinearProbeBenchmark.required_obs_keys
:type: list[str]

````

````{py:method} run(embeddings: numpy.ndarray, adata: anndata.AnnData, dataset_name: str) -> scmodelforge.eval.base.BenchmarkResult
:canonical: scmodelforge.eval.linear_probe.LinearProbeBenchmark.run

```{autodoc2-docstring} scmodelforge.eval.linear_probe.LinearProbeBenchmark.run
```

````

````{py:method} validate_adata(adata: anndata.AnnData) -> None
:canonical: scmodelforge.eval.linear_probe.LinearProbeBenchmark.validate_adata

````

`````
