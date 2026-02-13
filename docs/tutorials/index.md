# Tutorials

Hands-on guides for using scModelForge in your single-cell research. Tutorials progress from basic workflows to advanced customization.

## Getting Started

New to scModelForge? Start here. These tutorials assume familiarity with single-cell RNA-seq analysis (scanpy, AnnData) but not with transformer models or pretraining.

```{toctree}
:maxdepth: 1

quickstart
data_loading
pretraining
finetuning_cell_type
evaluation
```

## Intermediate Workflows

Deeper dives into specific capabilities for users comfortable with the basics.

```{toctree}
:maxdepth: 1

tokenization_guide
hub_models
multi_species
perturbation_prediction
large_scale_data
```

## Advanced Topics

For model developers, ML engineers, and power users who want to extend scModelForge.

```{toctree}
:maxdepth: 1

custom_tokenizer
custom_model
custom_benchmark
distributed_training
plugin_development
scverse_integration
```

## Which tutorial should I read?

| I want to... | Start with |
|---|---|
| Get my first model running quickly | [Quick Start](quickstart.md) |
| Understand data loading from H5AD and Census | [Data Loading & Preprocessing](data_loading.md) |
| Pretrain a model on my own dataset | [Pretraining a Foundation Model](pretraining.md) |
| Classify cell types with a pretrained model | [Fine-tuning for Cell Type Annotation](finetuning_cell_type.md) |
| Evaluate how good my model is | [Model Evaluation & Benchmarking](evaluation.md) |
| Choose the right tokenization strategy | [Tokenization Strategies](tokenization_guide.md) |
| Share my model on HuggingFace Hub | [HuggingFace Hub Integration](hub_models.md) |
| Train on human + mouse data together | [Multi-species Analysis](multi_species.md) |
| Predict perturbation responses | [Perturbation Response Prediction](perturbation_prediction.md) |
| Scale to millions of cells | [Large-scale Data Handling](large_scale_data.md) |
| Implement a new tokenizer | [Building Custom Tokenizers](custom_tokenizer.md) |
| Implement a new model architecture | [Building Custom Models](custom_model.md) |
| Add a new evaluation metric | [Building Custom Benchmarks](custom_benchmark.md) |
| Train across multiple GPUs | [Distributed Training with FSDP](distributed_training.md) |
| Create a pip-installable plugin | [Plugin Development](plugin_development.md) |
| Integrate with scanpy and scverse | [scverse Ecosystem Integration](scverse_integration.md) |
