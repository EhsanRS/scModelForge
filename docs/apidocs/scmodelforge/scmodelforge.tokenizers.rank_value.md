# {py:mod}`scmodelforge.tokenizers.rank_value`

```{py:module} scmodelforge.tokenizers.rank_value
```

```{autodoc2-docstring} scmodelforge.tokenizers.rank_value
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RankValueTokenizer <scmodelforge.tokenizers.rank_value.RankValueTokenizer>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.rank_value.RankValueTokenizer
    :summary:
    ```
````

### API

`````{py:class} RankValueTokenizer(gene_vocab: scmodelforge.data.gene_vocab.GeneVocab, max_len: int = 2048, prepend_cls: bool = True)
:canonical: scmodelforge.tokenizers.rank_value.RankValueTokenizer

Bases: {py:obj}`scmodelforge.tokenizers.base.BaseTokenizer`

```{autodoc2-docstring} scmodelforge.tokenizers.rank_value.RankValueTokenizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.tokenizers.rank_value.RankValueTokenizer.__init__
```

````{py:property} vocab_size
:canonical: scmodelforge.tokenizers.rank_value.RankValueTokenizer.vocab_size
:type: int

````

````{py:property} strategy_name
:canonical: scmodelforge.tokenizers.rank_value.RankValueTokenizer.strategy_name
:type: str

````

````{py:method} tokenize(expression: numpy.ndarray | torch.Tensor, gene_indices: numpy.ndarray | torch.Tensor, metadata: dict[str, typing.Any] | None = None) -> scmodelforge.tokenizers.base.TokenizedCell
:canonical: scmodelforge.tokenizers.rank_value.RankValueTokenizer.tokenize

````

````{py:method} tokenize_batch(expressions: list[numpy.ndarray | torch.Tensor], gene_indices_list: list[numpy.ndarray | torch.Tensor], metadata_list: list[dict[str, typing.Any]] | None = None) -> dict[str, torch.Tensor]
:canonical: scmodelforge.tokenizers.rank_value.RankValueTokenizer.tokenize_batch

````

`````
