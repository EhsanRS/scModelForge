# {py:mod}`scmodelforge.eval.harness`

```{py:module} scmodelforge.eval.harness
```

```{autodoc2-docstring} scmodelforge.eval.harness
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EvalHarness <scmodelforge.eval.harness.EvalHarness>`
  - ```{autodoc2-docstring} scmodelforge.eval.harness.EvalHarness
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.eval.harness.logger>`
  - ```{autodoc2-docstring} scmodelforge.eval.harness.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.eval.harness.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.eval.harness.logger
```

````

`````{py:class} EvalHarness(benchmarks: list[scmodelforge.eval.base.BaseBenchmark])
:canonical: scmodelforge.eval.harness.EvalHarness

```{autodoc2-docstring} scmodelforge.eval.harness.EvalHarness
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.eval.harness.EvalHarness.__init__
```

````{py:method} from_config(config: scmodelforge.config.schema.EvalConfig) -> scmodelforge.eval.harness.EvalHarness
:canonical: scmodelforge.eval.harness.EvalHarness.from_config
:classmethod:

```{autodoc2-docstring} scmodelforge.eval.harness.EvalHarness.from_config
```

````

````{py:method} run(model: torch.nn.Module, datasets: dict[str, anndata.AnnData], tokenizer: scmodelforge.tokenizers.base.BaseTokenizer, batch_size: int = 256, device: str = 'cpu') -> list[scmodelforge.eval.base.BenchmarkResult]
:canonical: scmodelforge.eval.harness.EvalHarness.run

```{autodoc2-docstring} scmodelforge.eval.harness.EvalHarness.run
```

````

````{py:method} run_on_embeddings(embeddings: numpy.ndarray, adata: anndata.AnnData, dataset_name: str) -> list[scmodelforge.eval.base.BenchmarkResult]
:canonical: scmodelforge.eval.harness.EvalHarness.run_on_embeddings

```{autodoc2-docstring} scmodelforge.eval.harness.EvalHarness.run_on_embeddings
```

````

`````
