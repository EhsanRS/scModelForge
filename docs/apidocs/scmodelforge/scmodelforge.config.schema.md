# {py:mod}`scmodelforge.config.schema`

```{py:module} scmodelforge.config.schema
```

```{autodoc2-docstring} scmodelforge.config.schema
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PreprocessingConfig <scmodelforge.config.schema.PreprocessingConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.PreprocessingConfig
    :summary:
    ```
* - {py:obj}`CensusConfig <scmodelforge.config.schema.CensusConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.CensusConfig
    :summary:
    ```
* - {py:obj}`DataConfig <scmodelforge.config.schema.DataConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.DataConfig
    :summary:
    ```
* - {py:obj}`MaskingConfig <scmodelforge.config.schema.MaskingConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.MaskingConfig
    :summary:
    ```
* - {py:obj}`TokenizerConfig <scmodelforge.config.schema.TokenizerConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.TokenizerConfig
    :summary:
    ```
* - {py:obj}`ModelConfig <scmodelforge.config.schema.ModelConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig
    :summary:
    ```
* - {py:obj}`OptimizerConfig <scmodelforge.config.schema.OptimizerConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.OptimizerConfig
    :summary:
    ```
* - {py:obj}`SchedulerConfig <scmodelforge.config.schema.SchedulerConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.SchedulerConfig
    :summary:
    ```
* - {py:obj}`TrainingConfig <scmodelforge.config.schema.TrainingConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig
    :summary:
    ```
* - {py:obj}`EvalConfig <scmodelforge.config.schema.EvalConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.EvalConfig
    :summary:
    ```
* - {py:obj}`TaskHeadConfig <scmodelforge.config.schema.TaskHeadConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.TaskHeadConfig
    :summary:
    ```
* - {py:obj}`LoRAConfig <scmodelforge.config.schema.LoRAConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.LoRAConfig
    :summary:
    ```
* - {py:obj}`FinetuneConfig <scmodelforge.config.schema.FinetuneConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig
    :summary:
    ```
* - {py:obj}`ScModelForgeConfig <scmodelforge.config.schema.ScModelForgeConfig>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.ScModelForgeConfig
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_config <scmodelforge.config.schema.load_config>`
  - ```{autodoc2-docstring} scmodelforge.config.schema.load_config
    :summary:
    ```
````

### API

`````{py:class} PreprocessingConfig
:canonical: scmodelforge.config.schema.PreprocessingConfig

```{autodoc2-docstring} scmodelforge.config.schema.PreprocessingConfig
```

````{py:attribute} normalize
:canonical: scmodelforge.config.schema.PreprocessingConfig.normalize
:type: str | None
:value: >
   'library_size'

```{autodoc2-docstring} scmodelforge.config.schema.PreprocessingConfig.normalize
```

````

````{py:attribute} target_sum
:canonical: scmodelforge.config.schema.PreprocessingConfig.target_sum
:type: float | None
:value: >
   10000.0

```{autodoc2-docstring} scmodelforge.config.schema.PreprocessingConfig.target_sum
```

````

````{py:attribute} hvg_selection
:canonical: scmodelforge.config.schema.PreprocessingConfig.hvg_selection
:type: int | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.PreprocessingConfig.hvg_selection
```

````

````{py:attribute} log1p
:canonical: scmodelforge.config.schema.PreprocessingConfig.log1p
:type: bool
:value: >
   True

```{autodoc2-docstring} scmodelforge.config.schema.PreprocessingConfig.log1p
```

````

`````

`````{py:class} CensusConfig
:canonical: scmodelforge.config.schema.CensusConfig

```{autodoc2-docstring} scmodelforge.config.schema.CensusConfig
```

````{py:attribute} organism
:canonical: scmodelforge.config.schema.CensusConfig.organism
:type: str
:value: >
   'Homo sapiens'

```{autodoc2-docstring} scmodelforge.config.schema.CensusConfig.organism
```

````

````{py:attribute} census_version
:canonical: scmodelforge.config.schema.CensusConfig.census_version
:type: str
:value: >
   'latest'

```{autodoc2-docstring} scmodelforge.config.schema.CensusConfig.census_version
```

````

````{py:attribute} obs_value_filter
:canonical: scmodelforge.config.schema.CensusConfig.obs_value_filter
:type: str | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.CensusConfig.obs_value_filter
```

````

````{py:attribute} var_value_filter
:canonical: scmodelforge.config.schema.CensusConfig.var_value_filter
:type: str | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.CensusConfig.var_value_filter
```

````

````{py:attribute} filters
:canonical: scmodelforge.config.schema.CensusConfig.filters
:type: dict[str, typing.Any] | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.CensusConfig.filters
```

````

````{py:attribute} obs_columns
:canonical: scmodelforge.config.schema.CensusConfig.obs_columns
:type: list[str] | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.CensusConfig.obs_columns
```

````

`````

`````{py:class} DataConfig
:canonical: scmodelforge.config.schema.DataConfig

```{autodoc2-docstring} scmodelforge.config.schema.DataConfig
```

````{py:attribute} source
:canonical: scmodelforge.config.schema.DataConfig.source
:type: str
:value: >
   'local'

```{autodoc2-docstring} scmodelforge.config.schema.DataConfig.source
```

````

````{py:attribute} paths
:canonical: scmodelforge.config.schema.DataConfig.paths
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.DataConfig.paths
```

````

````{py:attribute} gene_vocab
:canonical: scmodelforge.config.schema.DataConfig.gene_vocab
:type: str
:value: >
   'human_protein_coding'

```{autodoc2-docstring} scmodelforge.config.schema.DataConfig.gene_vocab
```

````

````{py:attribute} preprocessing
:canonical: scmodelforge.config.schema.DataConfig.preprocessing
:type: scmodelforge.config.schema.PreprocessingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.DataConfig.preprocessing
```

````

````{py:attribute} max_genes
:canonical: scmodelforge.config.schema.DataConfig.max_genes
:type: int
:value: >
   2048

```{autodoc2-docstring} scmodelforge.config.schema.DataConfig.max_genes
```

````

````{py:attribute} num_workers
:canonical: scmodelforge.config.schema.DataConfig.num_workers
:type: int
:value: >
   4

```{autodoc2-docstring} scmodelforge.config.schema.DataConfig.num_workers
```

````

````{py:attribute} census
:canonical: scmodelforge.config.schema.DataConfig.census
:type: scmodelforge.config.schema.CensusConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.DataConfig.census
```

````

`````

`````{py:class} MaskingConfig
:canonical: scmodelforge.config.schema.MaskingConfig

```{autodoc2-docstring} scmodelforge.config.schema.MaskingConfig
```

````{py:attribute} mask_ratio
:canonical: scmodelforge.config.schema.MaskingConfig.mask_ratio
:type: float
:value: >
   0.15

```{autodoc2-docstring} scmodelforge.config.schema.MaskingConfig.mask_ratio
```

````

````{py:attribute} random_replace_ratio
:canonical: scmodelforge.config.schema.MaskingConfig.random_replace_ratio
:type: float
:value: >
   0.1

```{autodoc2-docstring} scmodelforge.config.schema.MaskingConfig.random_replace_ratio
```

````

````{py:attribute} keep_ratio
:canonical: scmodelforge.config.schema.MaskingConfig.keep_ratio
:type: float
:value: >
   0.1

```{autodoc2-docstring} scmodelforge.config.schema.MaskingConfig.keep_ratio
```

````

`````

`````{py:class} TokenizerConfig
:canonical: scmodelforge.config.schema.TokenizerConfig

```{autodoc2-docstring} scmodelforge.config.schema.TokenizerConfig
```

````{py:attribute} strategy
:canonical: scmodelforge.config.schema.TokenizerConfig.strategy
:type: str
:value: >
   'rank_value'

```{autodoc2-docstring} scmodelforge.config.schema.TokenizerConfig.strategy
```

````

````{py:attribute} max_genes
:canonical: scmodelforge.config.schema.TokenizerConfig.max_genes
:type: int
:value: >
   2048

```{autodoc2-docstring} scmodelforge.config.schema.TokenizerConfig.max_genes
```

````

````{py:attribute} gene_vocab
:canonical: scmodelforge.config.schema.TokenizerConfig.gene_vocab
:type: str
:value: >
   'human_protein_coding'

```{autodoc2-docstring} scmodelforge.config.schema.TokenizerConfig.gene_vocab
```

````

````{py:attribute} prepend_cls
:canonical: scmodelforge.config.schema.TokenizerConfig.prepend_cls
:type: bool
:value: >
   True

```{autodoc2-docstring} scmodelforge.config.schema.TokenizerConfig.prepend_cls
```

````

````{py:attribute} n_bins
:canonical: scmodelforge.config.schema.TokenizerConfig.n_bins
:type: int
:value: >
   51

```{autodoc2-docstring} scmodelforge.config.schema.TokenizerConfig.n_bins
```

````

````{py:attribute} binning_method
:canonical: scmodelforge.config.schema.TokenizerConfig.binning_method
:type: str
:value: >
   'uniform'

```{autodoc2-docstring} scmodelforge.config.schema.TokenizerConfig.binning_method
```

````

````{py:attribute} masking
:canonical: scmodelforge.config.schema.TokenizerConfig.masking
:type: scmodelforge.config.schema.MaskingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.TokenizerConfig.masking
```

````

`````

`````{py:class} ModelConfig
:canonical: scmodelforge.config.schema.ModelConfig

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig
```

````{py:attribute} architecture
:canonical: scmodelforge.config.schema.ModelConfig.architecture
:type: str
:value: >
   'transformer_encoder'

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.architecture
```

````

````{py:attribute} hidden_dim
:canonical: scmodelforge.config.schema.ModelConfig.hidden_dim
:type: int
:value: >
   512

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.hidden_dim
```

````

````{py:attribute} num_layers
:canonical: scmodelforge.config.schema.ModelConfig.num_layers
:type: int
:value: >
   12

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.num_layers
```

````

````{py:attribute} num_heads
:canonical: scmodelforge.config.schema.ModelConfig.num_heads
:type: int
:value: >
   8

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.num_heads
```

````

````{py:attribute} ffn_dim
:canonical: scmodelforge.config.schema.ModelConfig.ffn_dim
:type: int | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.ffn_dim
```

````

````{py:attribute} dropout
:canonical: scmodelforge.config.schema.ModelConfig.dropout
:type: float
:value: >
   0.1

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.dropout
```

````

````{py:attribute} max_seq_len
:canonical: scmodelforge.config.schema.ModelConfig.max_seq_len
:type: int
:value: >
   2048

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.max_seq_len
```

````

````{py:attribute} pooling
:canonical: scmodelforge.config.schema.ModelConfig.pooling
:type: str
:value: >
   'cls'

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.pooling
```

````

````{py:attribute} activation
:canonical: scmodelforge.config.schema.ModelConfig.activation
:type: str
:value: >
   'gelu'

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.activation
```

````

````{py:attribute} use_expression_values
:canonical: scmodelforge.config.schema.ModelConfig.use_expression_values
:type: bool
:value: >
   True

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.use_expression_values
```

````

````{py:attribute} pretraining_task
:canonical: scmodelforge.config.schema.ModelConfig.pretraining_task
:type: str
:value: >
   'masked_gene_prediction'

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.pretraining_task
```

````

````{py:attribute} mask_ratio
:canonical: scmodelforge.config.schema.ModelConfig.mask_ratio
:type: float
:value: >
   0.15

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.mask_ratio
```

````

````{py:attribute} vocab_size
:canonical: scmodelforge.config.schema.ModelConfig.vocab_size
:type: int | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.vocab_size
```

````

````{py:attribute} n_bins
:canonical: scmodelforge.config.schema.ModelConfig.n_bins
:type: int
:value: >
   51

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.n_bins
```

````

````{py:attribute} gene_loss_weight
:canonical: scmodelforge.config.schema.ModelConfig.gene_loss_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.gene_loss_weight
```

````

````{py:attribute} expression_loss_weight
:canonical: scmodelforge.config.schema.ModelConfig.expression_loss_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.expression_loss_weight
```

````

````{py:attribute} decoder_dim
:canonical: scmodelforge.config.schema.ModelConfig.decoder_dim
:type: int | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.decoder_dim
```

````

````{py:attribute} decoder_layers
:canonical: scmodelforge.config.schema.ModelConfig.decoder_layers
:type: int
:value: >
   4

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.decoder_layers
```

````

````{py:attribute} decoder_heads
:canonical: scmodelforge.config.schema.ModelConfig.decoder_heads
:type: int | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.ModelConfig.decoder_heads
```

````

`````

`````{py:class} OptimizerConfig
:canonical: scmodelforge.config.schema.OptimizerConfig

```{autodoc2-docstring} scmodelforge.config.schema.OptimizerConfig
```

````{py:attribute} name
:canonical: scmodelforge.config.schema.OptimizerConfig.name
:type: str
:value: >
   'adamw'

```{autodoc2-docstring} scmodelforge.config.schema.OptimizerConfig.name
```

````

````{py:attribute} lr
:canonical: scmodelforge.config.schema.OptimizerConfig.lr
:type: float
:value: >
   0.0001

```{autodoc2-docstring} scmodelforge.config.schema.OptimizerConfig.lr
```

````

````{py:attribute} weight_decay
:canonical: scmodelforge.config.schema.OptimizerConfig.weight_decay
:type: float
:value: >
   0.01

```{autodoc2-docstring} scmodelforge.config.schema.OptimizerConfig.weight_decay
```

````

`````

`````{py:class} SchedulerConfig
:canonical: scmodelforge.config.schema.SchedulerConfig

```{autodoc2-docstring} scmodelforge.config.schema.SchedulerConfig
```

````{py:attribute} name
:canonical: scmodelforge.config.schema.SchedulerConfig.name
:type: str
:value: >
   'cosine_warmup'

```{autodoc2-docstring} scmodelforge.config.schema.SchedulerConfig.name
```

````

````{py:attribute} warmup_steps
:canonical: scmodelforge.config.schema.SchedulerConfig.warmup_steps
:type: int
:value: >
   2000

```{autodoc2-docstring} scmodelforge.config.schema.SchedulerConfig.warmup_steps
```

````

````{py:attribute} total_steps
:canonical: scmodelforge.config.schema.SchedulerConfig.total_steps
:type: int
:value: >
   100000

```{autodoc2-docstring} scmodelforge.config.schema.SchedulerConfig.total_steps
```

````

`````

`````{py:class} TrainingConfig
:canonical: scmodelforge.config.schema.TrainingConfig

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig
```

````{py:attribute} batch_size
:canonical: scmodelforge.config.schema.TrainingConfig.batch_size
:type: int
:value: >
   64

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.batch_size
```

````

````{py:attribute} max_epochs
:canonical: scmodelforge.config.schema.TrainingConfig.max_epochs
:type: int
:value: >
   10

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.max_epochs
```

````

````{py:attribute} seed
:canonical: scmodelforge.config.schema.TrainingConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.seed
```

````

````{py:attribute} strategy
:canonical: scmodelforge.config.schema.TrainingConfig.strategy
:type: str
:value: >
   'ddp'

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.strategy
```

````

````{py:attribute} num_gpus
:canonical: scmodelforge.config.schema.TrainingConfig.num_gpus
:type: int | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.num_gpus
```

````

````{py:attribute} precision
:canonical: scmodelforge.config.schema.TrainingConfig.precision
:type: str
:value: >
   'bf16-mixed'

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.precision
```

````

````{py:attribute} optimizer
:canonical: scmodelforge.config.schema.TrainingConfig.optimizer
:type: scmodelforge.config.schema.OptimizerConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.optimizer
```

````

````{py:attribute} scheduler
:canonical: scmodelforge.config.schema.TrainingConfig.scheduler
:type: scmodelforge.config.schema.SchedulerConfig | None
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.scheduler
```

````

````{py:attribute} gradient_clip
:canonical: scmodelforge.config.schema.TrainingConfig.gradient_clip
:type: float
:value: >
   1.0

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.gradient_clip
```

````

````{py:attribute} gradient_accumulation
:canonical: scmodelforge.config.schema.TrainingConfig.gradient_accumulation
:type: int
:value: >
   1

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.gradient_accumulation
```

````

````{py:attribute} logger
:canonical: scmodelforge.config.schema.TrainingConfig.logger
:type: str
:value: >
   'wandb'

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.logger
```

````

````{py:attribute} wandb_project
:canonical: scmodelforge.config.schema.TrainingConfig.wandb_project
:type: str
:value: >
   'scmodelforge'

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.wandb_project
```

````

````{py:attribute} run_name
:canonical: scmodelforge.config.schema.TrainingConfig.run_name
:type: str | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.run_name
```

````

````{py:attribute} log_dir
:canonical: scmodelforge.config.schema.TrainingConfig.log_dir
:type: str
:value: >
   'logs'

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.log_dir
```

````

````{py:attribute} log_every_n_steps
:canonical: scmodelforge.config.schema.TrainingConfig.log_every_n_steps
:type: int
:value: >
   50

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.log_every_n_steps
```

````

````{py:attribute} checkpoint_dir
:canonical: scmodelforge.config.schema.TrainingConfig.checkpoint_dir
:type: str
:value: >
   'checkpoints'

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.checkpoint_dir
```

````

````{py:attribute} save_top_k
:canonical: scmodelforge.config.schema.TrainingConfig.save_top_k
:type: int
:value: >
   3

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.save_top_k
```

````

````{py:attribute} num_workers
:canonical: scmodelforge.config.schema.TrainingConfig.num_workers
:type: int
:value: >
   4

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.num_workers
```

````

````{py:attribute} val_split
:canonical: scmodelforge.config.schema.TrainingConfig.val_split
:type: float
:value: >
   0.05

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.val_split
```

````

````{py:attribute} resume_from
:canonical: scmodelforge.config.schema.TrainingConfig.resume_from
:type: str | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.TrainingConfig.resume_from
```

````

`````

`````{py:class} EvalConfig
:canonical: scmodelforge.config.schema.EvalConfig

```{autodoc2-docstring} scmodelforge.config.schema.EvalConfig
```

````{py:attribute} every_n_epochs
:canonical: scmodelforge.config.schema.EvalConfig.every_n_epochs
:type: int
:value: >
   2

```{autodoc2-docstring} scmodelforge.config.schema.EvalConfig.every_n_epochs
```

````

````{py:attribute} batch_size
:canonical: scmodelforge.config.schema.EvalConfig.batch_size
:type: int
:value: >
   256

```{autodoc2-docstring} scmodelforge.config.schema.EvalConfig.batch_size
```

````

````{py:attribute} benchmarks
:canonical: scmodelforge.config.schema.EvalConfig.benchmarks
:type: list[typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.EvalConfig.benchmarks
```

````

`````

`````{py:class} TaskHeadConfig
:canonical: scmodelforge.config.schema.TaskHeadConfig

```{autodoc2-docstring} scmodelforge.config.schema.TaskHeadConfig
```

````{py:attribute} task
:canonical: scmodelforge.config.schema.TaskHeadConfig.task
:type: str
:value: >
   'classification'

```{autodoc2-docstring} scmodelforge.config.schema.TaskHeadConfig.task
```

````

````{py:attribute} n_classes
:canonical: scmodelforge.config.schema.TaskHeadConfig.n_classes
:type: int | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.TaskHeadConfig.n_classes
```

````

````{py:attribute} output_dim
:canonical: scmodelforge.config.schema.TaskHeadConfig.output_dim
:type: int
:value: >
   1

```{autodoc2-docstring} scmodelforge.config.schema.TaskHeadConfig.output_dim
```

````

````{py:attribute} hidden_dim
:canonical: scmodelforge.config.schema.TaskHeadConfig.hidden_dim
:type: int | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.TaskHeadConfig.hidden_dim
```

````

````{py:attribute} dropout
:canonical: scmodelforge.config.schema.TaskHeadConfig.dropout
:type: float
:value: >
   0.1

```{autodoc2-docstring} scmodelforge.config.schema.TaskHeadConfig.dropout
```

````

`````

`````{py:class} LoRAConfig
:canonical: scmodelforge.config.schema.LoRAConfig

```{autodoc2-docstring} scmodelforge.config.schema.LoRAConfig
```

````{py:attribute} enabled
:canonical: scmodelforge.config.schema.LoRAConfig.enabled
:type: bool
:value: >
   False

```{autodoc2-docstring} scmodelforge.config.schema.LoRAConfig.enabled
```

````

````{py:attribute} rank
:canonical: scmodelforge.config.schema.LoRAConfig.rank
:type: int
:value: >
   8

```{autodoc2-docstring} scmodelforge.config.schema.LoRAConfig.rank
```

````

````{py:attribute} alpha
:canonical: scmodelforge.config.schema.LoRAConfig.alpha
:type: int
:value: >
   16

```{autodoc2-docstring} scmodelforge.config.schema.LoRAConfig.alpha
```

````

````{py:attribute} dropout
:canonical: scmodelforge.config.schema.LoRAConfig.dropout
:type: float
:value: >
   0.05

```{autodoc2-docstring} scmodelforge.config.schema.LoRAConfig.dropout
```

````

````{py:attribute} target_modules
:canonical: scmodelforge.config.schema.LoRAConfig.target_modules
:type: list[str] | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.LoRAConfig.target_modules
```

````

````{py:attribute} bias
:canonical: scmodelforge.config.schema.LoRAConfig.bias
:type: str
:value: >
   'none'

```{autodoc2-docstring} scmodelforge.config.schema.LoRAConfig.bias
```

````

`````

`````{py:class} FinetuneConfig
:canonical: scmodelforge.config.schema.FinetuneConfig

```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig
```

````{py:attribute} checkpoint_path
:canonical: scmodelforge.config.schema.FinetuneConfig.checkpoint_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig.checkpoint_path
```

````

````{py:attribute} freeze_backbone
:canonical: scmodelforge.config.schema.FinetuneConfig.freeze_backbone
:type: bool
:value: >
   False

```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig.freeze_backbone
```

````

````{py:attribute} freeze_backbone_epochs
:canonical: scmodelforge.config.schema.FinetuneConfig.freeze_backbone_epochs
:type: int
:value: >
   0

```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig.freeze_backbone_epochs
```

````

````{py:attribute} label_key
:canonical: scmodelforge.config.schema.FinetuneConfig.label_key
:type: str
:value: >
   'cell_type'

```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig.label_key
```

````

````{py:attribute} head
:canonical: scmodelforge.config.schema.FinetuneConfig.head
:type: scmodelforge.config.schema.TaskHeadConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig.head
```

````

````{py:attribute} backbone_lr
:canonical: scmodelforge.config.schema.FinetuneConfig.backbone_lr
:type: float | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig.backbone_lr
```

````

````{py:attribute} head_lr
:canonical: scmodelforge.config.schema.FinetuneConfig.head_lr
:type: float | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig.head_lr
```

````

````{py:attribute} lora
:canonical: scmodelforge.config.schema.FinetuneConfig.lora
:type: scmodelforge.config.schema.LoRAConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.FinetuneConfig.lora
```

````

`````

`````{py:class} ScModelForgeConfig
:canonical: scmodelforge.config.schema.ScModelForgeConfig

```{autodoc2-docstring} scmodelforge.config.schema.ScModelForgeConfig
```

````{py:attribute} data
:canonical: scmodelforge.config.schema.ScModelForgeConfig.data
:type: scmodelforge.config.schema.DataConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.ScModelForgeConfig.data
```

````

````{py:attribute} tokenizer
:canonical: scmodelforge.config.schema.ScModelForgeConfig.tokenizer
:type: scmodelforge.config.schema.TokenizerConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.ScModelForgeConfig.tokenizer
```

````

````{py:attribute} model
:canonical: scmodelforge.config.schema.ScModelForgeConfig.model
:type: scmodelforge.config.schema.ModelConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.ScModelForgeConfig.model
```

````

````{py:attribute} training
:canonical: scmodelforge.config.schema.ScModelForgeConfig.training
:type: scmodelforge.config.schema.TrainingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.ScModelForgeConfig.training
```

````

````{py:attribute} eval
:canonical: scmodelforge.config.schema.ScModelForgeConfig.eval
:type: scmodelforge.config.schema.EvalConfig
:value: >
   'field(...)'

```{autodoc2-docstring} scmodelforge.config.schema.ScModelForgeConfig.eval
```

````

````{py:attribute} finetune
:canonical: scmodelforge.config.schema.ScModelForgeConfig.finetune
:type: scmodelforge.config.schema.FinetuneConfig | None
:value: >
   None

```{autodoc2-docstring} scmodelforge.config.schema.ScModelForgeConfig.finetune
```

````

`````

````{py:function} load_config(path: str | pathlib.Path) -> scmodelforge.config.schema.ScModelForgeConfig
:canonical: scmodelforge.config.schema.load_config

```{autodoc2-docstring} scmodelforge.config.schema.load_config
```
````
