# {py:mod}`scmodelforge.data.preprocessing`

```{py:module} scmodelforge.data.preprocessing
```

```{autodoc2-docstring} scmodelforge.data.preprocessing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PreprocessingPipeline <scmodelforge.data.preprocessing.PreprocessingPipeline>`
  - ```{autodoc2-docstring} scmodelforge.data.preprocessing.PreprocessingPipeline
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`select_highly_variable_genes <scmodelforge.data.preprocessing.select_highly_variable_genes>`
  - ```{autodoc2-docstring} scmodelforge.data.preprocessing.select_highly_variable_genes
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.data.preprocessing.logger>`
  - ```{autodoc2-docstring} scmodelforge.data.preprocessing.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.data.preprocessing.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.data.preprocessing.logger
```

````

````{py:class} PreprocessingPipeline(normalize: str | None = 'library_size', target_sum: float | None = 10000.0, log1p: bool = True)
:canonical: scmodelforge.data.preprocessing.PreprocessingPipeline

```{autodoc2-docstring} scmodelforge.data.preprocessing.PreprocessingPipeline
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.data.preprocessing.PreprocessingPipeline.__init__
```

````

````{py:function} select_highly_variable_genes(expressions: numpy.ndarray, n_top_genes: int) -> numpy.ndarray
:canonical: scmodelforge.data.preprocessing.select_highly_variable_genes

```{autodoc2-docstring} scmodelforge.data.preprocessing.select_highly_variable_genes
```
````
