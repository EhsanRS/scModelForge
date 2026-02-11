# scModelForge

**A pretraining toolkit for single-cell foundation models**

*Democratising the training of cell foundation models — train your own in 200 lines of config, not 2,000 lines of bespoke code.*

[![CI](https://github.com/EhsanRS/scModelForge/actions/workflows/ci.yml/badge.svg)](https://github.com/EhsanRS/scModelForge/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

---

## What is scModelForge?

scModelForge is an open-source Python toolkit that makes training and fine-tuning single-cell foundation models accessible to researchers who are not infrastructure specialists. It provides the missing layer between raw biological data (AnnData / CELLxGENE) and production-quality model training:

- **Standardised tokenisation strategies** — Geneformer-style rank ordering, scGPT-style binning, TranscriptFormer-style continuous projection, all behind a common interface.
- **GPU-accelerated data streaming** — AnnData-native data pipeline with lazy loading, sharding, and on-the-fly preprocessing.
- **Reference model architectures** — Clean, interoperable implementations of the major architecture families.
- **Config-driven training** — PyTorch Lightning-based training loop with DDP/FSDP, mixed precision, and WandB logging.
- **Integrated evaluation** — scIB metrics, linear probes, and perturbation benchmarks as training callbacks.

Built natively on the [scverse](https://scverse.org/) ecosystem.

## Installation

```bash
pip install scModelForge
```

For development:

```bash
git clone https://github.com/EhsanRS/scModelForge.git
cd scModelForge
pip install -e ".[dev]"
```

## Quick Start

Create a YAML config file:

```yaml
# geneformer_basic.yaml
data:
  source: local
  paths:
    - ./data/my_dataset.h5ad
  preprocessing:
    normalize: library_size
    hvg_selection: 2000

tokenizer:
  strategy: rank_value
  max_genes: 2048

model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 12
  num_heads: 8
  pretraining_task: masked_gene_prediction
  mask_ratio: 0.15

training:
  batch_size: 64
  max_epochs: 10
  precision: bf16-mixed
  strategy: ddp
  num_gpus: 4

eval:
  every_n_epochs: 2
  benchmarks:
    - cell_type_annotation:
        dataset: tabula_sapiens
```

Then launch training:

```bash
scmodelforge train --config geneformer_basic.yaml
```

## Architecture

scModelForge is organised into five core modules:

| Module | Responsibility |
|---|---|
| `scmodelforge.data` | AnnData streaming, preprocessing, gene vocabularies |
| `scmodelforge.tokenizers` | Pluggable tokenisation strategies |
| `scmodelforge.models` | Reference transformer architectures |
| `scmodelforge.training` | Config-driven training loop (Lightning) |
| `scmodelforge.eval` | Integrated benchmarks (scIB, perturbation, GRN) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
