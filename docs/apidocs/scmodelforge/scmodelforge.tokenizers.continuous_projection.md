# {py:mod}`scmodelforge.tokenizers.continuous_projection`

```{py:module} scmodelforge.tokenizers.continuous_projection
```

```{autodoc2-docstring} scmodelforge.tokenizers.continuous_projection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ContinuousProjectionTokenizer <scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer
    :summary:
    ```
````

### API

`````{py:class} ContinuousProjectionTokenizer(gene_vocab: scmodelforge.data.gene_vocab.GeneVocab, max_len: int = 2048, prepend_cls: bool = True, include_zero_genes: bool = True, log_transform: bool = False)
:canonical: scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer

Bases: {py:obj}`scmodelforge.tokenizers.base.BaseTokenizer`

```{autodoc2-docstring} scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer.__init__
```

````{py:property} vocab_size
:canonical: scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer.vocab_size
:type: int

````

````{py:property} strategy_name
:canonical: scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer.strategy_name
:type: str

````

````{py:method} tokenize(expression: numpy.ndarray | torch.Tensor, gene_indices: numpy.ndarray | torch.Tensor, metadata: dict[str, typing.Any] | None = None) -> scmodelforge.tokenizers.base.TokenizedCell
:canonical: scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer.tokenize

````

````{py:method} tokenize_batch(expressions: list[numpy.ndarray | torch.Tensor], gene_indices_list: list[numpy.ndarray | torch.Tensor], metadata_list: list[dict[str, typing.Any]] | None = None) -> dict[str, torch.Tensor]
:canonical: scmodelforge.tokenizers.continuous_projection.ContinuousProjectionTokenizer.tokenize_batch

````

`````
