# {py:mod}`scmodelforge.finetuning.data_module`

```{py:module} scmodelforge.finetuning.data_module
```

```{autodoc2-docstring} scmodelforge.finetuning.data_module
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LabelEncoder <scmodelforge.finetuning.data_module.LabelEncoder>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.data_module.LabelEncoder
    :summary:
    ```
* - {py:obj}`FineTuneDataModule <scmodelforge.finetuning.data_module.FineTuneDataModule>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.data_module.FineTuneDataModule
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.finetuning.data_module.logger>`
  - ```{autodoc2-docstring} scmodelforge.finetuning.data_module.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.finetuning.data_module.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.finetuning.data_module.logger
```

````

`````{py:class} LabelEncoder(labels: collections.abc.Sequence[str])
:canonical: scmodelforge.finetuning.data_module.LabelEncoder

```{autodoc2-docstring} scmodelforge.finetuning.data_module.LabelEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.finetuning.data_module.LabelEncoder.__init__
```

````{py:property} n_classes
:canonical: scmodelforge.finetuning.data_module.LabelEncoder.n_classes
:type: int

```{autodoc2-docstring} scmodelforge.finetuning.data_module.LabelEncoder.n_classes
```

````

````{py:property} classes
:canonical: scmodelforge.finetuning.data_module.LabelEncoder.classes
:type: list[str]

```{autodoc2-docstring} scmodelforge.finetuning.data_module.LabelEncoder.classes
```

````

````{py:method} encode(label: str) -> int
:canonical: scmodelforge.finetuning.data_module.LabelEncoder.encode

```{autodoc2-docstring} scmodelforge.finetuning.data_module.LabelEncoder.encode
```

````

````{py:method} decode(idx: int) -> str
:canonical: scmodelforge.finetuning.data_module.LabelEncoder.decode

```{autodoc2-docstring} scmodelforge.finetuning.data_module.LabelEncoder.decode
```

````

`````

`````{py:class} FineTuneDataModule(data_config: scmodelforge.config.schema.DataConfig, tokenizer_config: scmodelforge.config.schema.TokenizerConfig, finetune_config: scmodelforge.config.schema.FinetuneConfig, training_batch_size: int = 64, num_workers: int = 4, val_split: float = 0.1, seed: int = 42, adata: typing.Any | None = None)
:canonical: scmodelforge.finetuning.data_module.FineTuneDataModule

```{autodoc2-docstring} scmodelforge.finetuning.data_module.FineTuneDataModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} scmodelforge.finetuning.data_module.FineTuneDataModule.__init__
```

````{py:property} gene_vocab
:canonical: scmodelforge.finetuning.data_module.FineTuneDataModule.gene_vocab
:type: scmodelforge.data.gene_vocab.GeneVocab

```{autodoc2-docstring} scmodelforge.finetuning.data_module.FineTuneDataModule.gene_vocab
```

````

````{py:property} tokenizer
:canonical: scmodelforge.finetuning.data_module.FineTuneDataModule.tokenizer
:type: scmodelforge.tokenizers.base.BaseTokenizer

```{autodoc2-docstring} scmodelforge.finetuning.data_module.FineTuneDataModule.tokenizer
```

````

````{py:property} label_encoder
:canonical: scmodelforge.finetuning.data_module.FineTuneDataModule.label_encoder
:type: scmodelforge.finetuning.data_module.LabelEncoder | None

```{autodoc2-docstring} scmodelforge.finetuning.data_module.FineTuneDataModule.label_encoder
```

````

````{py:method} setup(stage: str | None = None) -> None
:canonical: scmodelforge.finetuning.data_module.FineTuneDataModule.setup

```{autodoc2-docstring} scmodelforge.finetuning.data_module.FineTuneDataModule.setup
```

````

````{py:method} train_dataloader() -> torch.utils.data.DataLoader
:canonical: scmodelforge.finetuning.data_module.FineTuneDataModule.train_dataloader

```{autodoc2-docstring} scmodelforge.finetuning.data_module.FineTuneDataModule.train_dataloader
```

````

````{py:method} val_dataloader() -> torch.utils.data.DataLoader
:canonical: scmodelforge.finetuning.data_module.FineTuneDataModule.val_dataloader

```{autodoc2-docstring} scmodelforge.finetuning.data_module.FineTuneDataModule.val_dataloader
```

````

`````
