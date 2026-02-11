# Getting Started

## Installation

Install scModelForge from PyPI:

```bash
pip install scModelForge
```

For development:

```bash
git clone https://github.com/EhsanRS/scModelForge.git
cd scModelForge
pip install -e ".[dev]"
```

### Optional dependencies

```bash
# CELLxGENE Census data source
pip install "scModelForge[census]"

# Evaluation benchmarks (scIB, pertpy)
pip install "scModelForge[eval]"

# Weights & Biases logging
pip install "scModelForge[wandb]"

# Everything
pip install "scModelForge[all]"
```

## Quick Start

*Coming soon — the training pipeline will be implemented in Stage 4.*

Create a YAML config file and run:

```bash
scmodelforge train --config your_config.yaml
```

See `configs/examples/geneformer_basic.yaml` for a reference configuration.
