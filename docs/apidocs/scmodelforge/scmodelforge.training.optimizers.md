# {py:mod}`scmodelforge.training.optimizers`

```{py:module} scmodelforge.training.optimizers
```

```{autodoc2-docstring} scmodelforge.training.optimizers
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_optimizer <scmodelforge.training.optimizers.build_optimizer>`
  - ```{autodoc2-docstring} scmodelforge.training.optimizers.build_optimizer
    :summary:
    ```
* - {py:obj}`build_scheduler <scmodelforge.training.optimizers.build_scheduler>`
  - ```{autodoc2-docstring} scmodelforge.training.optimizers.build_scheduler
    :summary:
    ```
````

### API

````{py:function} build_optimizer(model: torch.nn.Module, config: scmodelforge.config.schema.OptimizerConfig) -> torch.optim.Optimizer
:canonical: scmodelforge.training.optimizers.build_optimizer

```{autodoc2-docstring} scmodelforge.training.optimizers.build_optimizer
```
````

````{py:function} build_scheduler(optimizer: torch.optim.Optimizer, config: scmodelforge.config.schema.SchedulerConfig) -> dict
:canonical: scmodelforge.training.optimizers.build_scheduler

```{autodoc2-docstring} scmodelforge.training.optimizers.build_scheduler
```
````
