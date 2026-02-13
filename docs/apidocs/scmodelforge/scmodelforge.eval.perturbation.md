# {py:mod}`scmodelforge.eval.perturbation`

```{py:module} scmodelforge.eval.perturbation
```

```{autodoc2-docstring} scmodelforge.eval.perturbation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PerturbationBenchmark <scmodelforge.eval.perturbation.PerturbationBenchmark>`
  - ```{autodoc2-docstring} scmodelforge.eval.perturbation.PerturbationBenchmark
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_expression_deltas <scmodelforge.eval.perturbation.compute_expression_deltas>`
  - ```{autodoc2-docstring} scmodelforge.eval.perturbation.compute_expression_deltas
    :summary:
    ```
* - {py:obj}`find_top_degs <scmodelforge.eval.perturbation.find_top_degs>`
  - ```{autodoc2-docstring} scmodelforge.eval.perturbation.find_top_degs
    :summary:
    ```
* - {py:obj}`mean_shift_baseline <scmodelforge.eval.perturbation.mean_shift_baseline>`
  - ```{autodoc2-docstring} scmodelforge.eval.perturbation.mean_shift_baseline
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.eval.perturbation.logger>`
  - ```{autodoc2-docstring} scmodelforge.eval.perturbation.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.eval.perturbation.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.eval.perturbation.logger
```

````

````{py:function} compute_expression_deltas(adata: anndata.AnnData, perturbation_key: str, control_label: str) -> dict[str, numpy.ndarray]
:canonical: scmodelforge.eval.perturbation.compute_expression_deltas

```{autodoc2-docstring} scmodelforge.eval.perturbation.compute_expression_deltas
```
````

````{py:function} find_top_degs(delta: numpy.ndarray, n_top: int) -> numpy.ndarray
:canonical: scmodelforge.eval.perturbation.find_top_degs

```{autodoc2-docstring} scmodelforge.eval.perturbation.find_top_degs
```
````

````{py:function} mean_shift_baseline(train_deltas: dict[str, numpy.ndarray]) -> numpy.ndarray
:canonical: scmodelforge.eval.perturbation.mean_shift_baseline

```{autodoc2-docstring} scmodelforge.eval.perturbation.mean_shift_baseline
```
````

`````{py:class} PerturbationBenchmark(perturbation_key: str = 'perturbation', control_label: str = 'control', n_top_genes: int = 50, test_fraction: float = 0.2, seed: int = 42)
:canonical: scmodelforge.eval.perturbation.PerturbationBenchmark

Bases: {py:obj}`scmodelforge.eval.base.BaseBenchmark`

```{autodoc2-docstring} scmodelforge.eval.perturbation.PerturbationBenchmark
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.eval.perturbation.PerturbationBenchmark.__init__
```

````{py:property} name
:canonical: scmodelforge.eval.perturbation.PerturbationBenchmark.name
:type: str

````

````{py:property} required_obs_keys
:canonical: scmodelforge.eval.perturbation.PerturbationBenchmark.required_obs_keys
:type: list[str]

````

````{py:method} run(embeddings: numpy.ndarray, adata: anndata.AnnData, dataset_name: str) -> scmodelforge.eval.base.BenchmarkResult
:canonical: scmodelforge.eval.perturbation.PerturbationBenchmark.run

```{autodoc2-docstring} scmodelforge.eval.perturbation.PerturbationBenchmark.run
```

````

````{py:method} validate_adata(adata: anndata.AnnData) -> None
:canonical: scmodelforge.eval.perturbation.PerturbationBenchmark.validate_adata

````

`````
