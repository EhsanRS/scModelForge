# {py:mod}`scmodelforge.tokenizers.binned_expression`

```{py:module} scmodelforge.tokenizers.binned_expression
```

```{autodoc2-docstring} scmodelforge.tokenizers.binned_expression
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BinnedExpressionTokenizer <scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer
    :summary:
    ```
````

### API

`````{py:class} BinnedExpressionTokenizer(gene_vocab: scmodelforge.data.gene_vocab.GeneVocab, max_len: int = 2048, n_bins: int = 51, binning_method: str = 'uniform', bin_edges: numpy.ndarray | None = None, value_max: float = 10.0, prepend_cls: bool = True, include_zero_genes: bool = True)
:canonical: scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer

Bases: {py:obj}`scmodelforge.tokenizers.base.BaseTokenizer`

```{autodoc2-docstring} scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.__init__
```

````{py:method} fit(expression_values: numpy.ndarray) -> scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer
:canonical: scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.fit

```{autodoc2-docstring} scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.fit
```

````

````{py:property} bin_edges
:canonical: scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.bin_edges
:type: numpy.ndarray | None

```{autodoc2-docstring} scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.bin_edges
```

````

````{py:property} n_bin_tokens
:canonical: scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.n_bin_tokens
:type: int

```{autodoc2-docstring} scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.n_bin_tokens
```

````

````{py:property} vocab_size
:canonical: scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.vocab_size
:type: int

````

````{py:property} strategy_name
:canonical: scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.strategy_name
:type: str

````

````{py:method} tokenize(expression: numpy.ndarray | torch.Tensor, gene_indices: numpy.ndarray | torch.Tensor, metadata: dict[str, typing.Any] | None = None) -> scmodelforge.tokenizers.base.TokenizedCell
:canonical: scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.tokenize

````

````{py:method} tokenize_batch(expressions: list[numpy.ndarray | torch.Tensor], gene_indices_list: list[numpy.ndarray | torch.Tensor], metadata_list: list[dict[str, typing.Any]] | None = None) -> dict[str, torch.Tensor]
:canonical: scmodelforge.tokenizers.binned_expression.BinnedExpressionTokenizer.tokenize_batch

````

`````
