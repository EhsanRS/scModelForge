# {py:mod}`scmodelforge.eval.registry`

```{py:module} scmodelforge.eval.registry
```

```{autodoc2-docstring} scmodelforge.eval.registry
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`register_benchmark <scmodelforge.eval.registry.register_benchmark>`
  - ```{autodoc2-docstring} scmodelforge.eval.registry.register_benchmark
    :summary:
    ```
* - {py:obj}`get_benchmark <scmodelforge.eval.registry.get_benchmark>`
  - ```{autodoc2-docstring} scmodelforge.eval.registry.get_benchmark
    :summary:
    ```
* - {py:obj}`list_benchmarks <scmodelforge.eval.registry.list_benchmarks>`
  - ```{autodoc2-docstring} scmodelforge.eval.registry.list_benchmarks
    :summary:
    ```
````

### API

````{py:function} register_benchmark(name: str)
:canonical: scmodelforge.eval.registry.register_benchmark

```{autodoc2-docstring} scmodelforge.eval.registry.register_benchmark
```
````

````{py:function} get_benchmark(name: str, **kwargs: typing.Any) -> scmodelforge.eval.base.BaseBenchmark
:canonical: scmodelforge.eval.registry.get_benchmark

```{autodoc2-docstring} scmodelforge.eval.registry.get_benchmark
```
````

````{py:function} list_benchmarks() -> list[str]
:canonical: scmodelforge.eval.registry.list_benchmarks

```{autodoc2-docstring} scmodelforge.eval.registry.list_benchmarks
```
````
