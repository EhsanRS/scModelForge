# {py:mod}`scmodelforge.tokenizers.masking`

```{py:module} scmodelforge.tokenizers.masking
```

```{autodoc2-docstring} scmodelforge.tokenizers.masking
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MaskingStrategy <scmodelforge.tokenizers.masking.MaskingStrategy>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers.masking.MaskingStrategy
    :summary:
    ```
````

### API

`````{py:class} MaskingStrategy(mask_ratio: float = 0.15, mask_action_ratio: float = 0.8, random_replace_ratio: float = 0.1, vocab_size: int | None = None)
:canonical: scmodelforge.tokenizers.masking.MaskingStrategy

```{autodoc2-docstring} scmodelforge.tokenizers.masking.MaskingStrategy
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.tokenizers.masking.MaskingStrategy.__init__
```

````{py:method} apply(cell: scmodelforge.tokenizers.base.TokenizedCell, seed: int | None = None) -> scmodelforge.tokenizers.base.MaskedTokenizedCell
:canonical: scmodelforge.tokenizers.masking.MaskingStrategy.apply

```{autodoc2-docstring} scmodelforge.tokenizers.masking.MaskingStrategy.apply
```

````

`````
