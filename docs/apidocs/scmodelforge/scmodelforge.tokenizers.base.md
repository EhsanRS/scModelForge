# {py:mod}`scmodelforge.tokenizers.base`

```{py:module} scmodelforge.tokenizers.base
```

```{autodoc2-docstring} scmodelforge.tokenizers.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizedCell <scmodelforge.tokenizers.base.TokenizedCell>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.base.TokenizedCell
    :summary:
    ```
* - {py:obj}`MaskedTokenizedCell <scmodelforge.tokenizers.base.MaskedTokenizedCell>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.base.MaskedTokenizedCell
    :summary:
    ```
* - {py:obj}`BaseTokenizer <scmodelforge.tokenizers.base.BaseTokenizer>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.base.BaseTokenizer
    :summary:
    ```
````

### API

`````{py:class} TokenizedCell
:canonical: scmodelforge.tokenizers.base.TokenizedCell

```{autodoc2-docstring} scmodelforge.tokenizers.base.TokenizedCell
```

````{py:attribute} input_ids
:canonical: scmodelforge.tokenizers.base.TokenizedCell.input_ids
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} scmodelforge.tokenizers.base.TokenizedCell.input_ids
```

````

````{py:attribute} attention_mask
:canonical: scmodelforge.tokenizers.base.TokenizedCell.attention_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} scmodelforge.tokenizers.base.TokenizedCell.attention_mask
```

````

````{py:attribute} values
:canonical: scmodelforge.tokenizers.base.TokenizedCell.values
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.tokenizers.base.TokenizedCell.values
```

````

````{py:attribute} bin_ids
:canonical: scmodelforge.tokenizers.base.TokenizedCell.bin_ids
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.tokenizers.base.TokenizedCell.bin_ids
```

````

````{py:attribute} gene_indices
:canonical: scmodelforge.tokenizers.base.TokenizedCell.gene_indices
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.tokenizers.base.TokenizedCell.gene_indices
```

````

````{py:attribute} metadata
:canonical: scmodelforge.tokenizers.base.TokenizedCell.metadata
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.tokenizers.base.TokenizedCell.metadata
```

````

`````

`````{py:class} MaskedTokenizedCell
:canonical: scmodelforge.tokenizers.base.MaskedTokenizedCell

Bases: {py:obj}`scmodelforge.tokenizers.base.TokenizedCell`

```{autodoc2-docstring} scmodelforge.tokenizers.base.MaskedTokenizedCell
```

````{py:attribute} labels
:canonical: scmodelforge.tokenizers.base.MaskedTokenizedCell.labels
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.tokenizers.base.MaskedTokenizedCell.labels
```

````

````{py:attribute} masked_positions
:canonical: scmodelforge.tokenizers.base.MaskedTokenizedCell.masked_positions
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.tokenizers.base.MaskedTokenizedCell.masked_positions
```

````

````{py:attribute} input_ids
:canonical: scmodelforge.tokenizers.base.MaskedTokenizedCell.input_ids
:type: torch.Tensor
:value: >
   None

````

````{py:attribute} attention_mask
:canonical: scmodelforge.tokenizers.base.MaskedTokenizedCell.attention_mask
:type: torch.Tensor
:value: >
   None

````

````{py:attribute} values
:canonical: scmodelforge.tokenizers.base.MaskedTokenizedCell.values
:type: torch.Tensor | None
:value: >
   None

````

````{py:attribute} bin_ids
:canonical: scmodelforge.tokenizers.base.MaskedTokenizedCell.bin_ids
:type: torch.Tensor | None
:value: >
   None

````

````{py:attribute} gene_indices
:canonical: scmodelforge.tokenizers.base.MaskedTokenizedCell.gene_indices
:type: torch.Tensor
:value: >
   'field(...)'

````

````{py:attribute} metadata
:canonical: scmodelforge.tokenizers.base.MaskedTokenizedCell.metadata
:type: dict[str, typing.Any]
:value: >
   'field(...)'

````

`````

`````{py:class} BaseTokenizer(gene_vocab: scmodelforge.data.gene_vocab.GeneVocab, max_len: int = 2048)
:canonical: scmodelforge.tokenizers.base.BaseTokenizer

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} scmodelforge.tokenizers.base.BaseTokenizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.tokenizers.base.BaseTokenizer.__init__
```

````{py:method} tokenize(expression: numpy.ndarray | torch.Tensor, gene_indices: numpy.ndarray | torch.Tensor, metadata: dict[str, typing.Any] | None = None) -> scmodelforge.tokenizers.base.TokenizedCell
:canonical: scmodelforge.tokenizers.base.BaseTokenizer.tokenize
:abstractmethod:

```{autodoc2-docstring} scmodelforge.tokenizers.base.BaseTokenizer.tokenize
```

````

````{py:property} vocab_size
:canonical: scmodelforge.tokenizers.base.BaseTokenizer.vocab_size
:abstractmethod:
:type: int

```{autodoc2-docstring} scmodelforge.tokenizers.base.BaseTokenizer.vocab_size
```

````

````{py:property} strategy_name
:canonical: scmodelforge.tokenizers.base.BaseTokenizer.strategy_name
:abstractmethod:
:type: str

```{autodoc2-docstring} scmodelforge.tokenizers.base.BaseTokenizer.strategy_name
```

````

````{py:method} tokenize_batch(expressions: list[numpy.ndarray | torch.Tensor], gene_indices_list: list[numpy.ndarray | torch.Tensor], metadata_list: list[dict[str, typing.Any]] | None = None) -> dict[str, torch.Tensor]
:canonical: scmodelforge.tokenizers.base.BaseTokenizer.tokenize_batch

```{autodoc2-docstring} scmodelforge.tokenizers.base.BaseTokenizer.tokenize_batch
```

````

`````
