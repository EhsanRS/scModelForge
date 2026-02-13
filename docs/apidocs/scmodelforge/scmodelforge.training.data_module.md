# {py:mod}`scmodelforge.training.data_module`

```{py:module} scmodelforge.training.data_module
```

```{autodoc2-docstring} scmodelforge.training.data_module
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizedCellDataset <scmodelforge.training.data_module.TokenizedCellDataset>`
  - ```{autodoc2-docstring} scmodelforge.training.data_module.TokenizedCellDataset
    :summary:
    ```
* - {py:obj}`CellDataModule <scmodelforge.training.data_module.CellDataModule>`
  - ```{autodoc2-docstring} scmodelforge.training.data_module.CellDataModule
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.training.data_module.logger>`
  - ```{autodoc2-docstring} scmodelforge.training.data_module.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.training.data_module.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.training.data_module.logger
```

````

````{py:class} TokenizedCellDataset(dataset: torch.utils.data.Dataset, tokenizer: scmodelforge.tokenizers.base.BaseTokenizer, masking: scmodelforge.tokenizers.masking.MaskingStrategy | None = None)
:canonical: scmodelforge.training.data_module.TokenizedCellDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} scmodelforge.training.data_module.TokenizedCellDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.training.data_module.TokenizedCellDataset.__init__
```

````

`````{py:class} CellDataModule(data_config: scmodelforge.config.schema.DataConfig, tokenizer_config: scmodelforge.config.schema.TokenizerConfig, training_batch_size: int = 64, num_workers: int = 4, val_split: float = 0.05, seed: int = 42, adata: object | None = None)
:canonical: scmodelforge.training.data_module.CellDataModule

```{autodoc2-docstring} scmodelforge.training.data_module.CellDataModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.training.data_module.CellDataModule.__init__
```

````{py:property} gene_vocab
:canonical: scmodelforge.training.data_module.CellDataModule.gene_vocab
:type: scmodelforge.data.gene_vocab.GeneVocab

```{autodoc2-docstring} scmodelforge.training.data_module.CellDataModule.gene_vocab
```

````

````{py:property} tokenizer
:canonical: scmodelforge.training.data_module.CellDataModule.tokenizer
:type: scmodelforge.tokenizers.base.BaseTokenizer

```{autodoc2-docstring} scmodelforge.training.data_module.CellDataModule.tokenizer
```

````

````{py:property} masking
:canonical: scmodelforge.training.data_module.CellDataModule.masking
:type: scmodelforge.tokenizers.masking.MaskingStrategy | None

```{autodoc2-docstring} scmodelforge.training.data_module.CellDataModule.masking
```

````

````{py:method} setup(stage: str | None = None) -> None
:canonical: scmodelforge.training.data_module.CellDataModule.setup

```{autodoc2-docstring} scmodelforge.training.data_module.CellDataModule.setup
```

````

````{py:method} train_dataloader() -> torch.utils.data.DataLoader
:canonical: scmodelforge.training.data_module.CellDataModule.train_dataloader

```{autodoc2-docstring} scmodelforge.training.data_module.CellDataModule.train_dataloader
```

````

````{py:method} val_dataloader() -> torch.utils.data.DataLoader
:canonical: scmodelforge.training.data_module.CellDataModule.val_dataloader

```{autodoc2-docstring} scmodelforge.training.data_module.CellDataModule.val_dataloader
```

````

`````
