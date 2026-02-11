# Stage 0: Repository Scaffolding

## Overview

Set up the project infrastructure: directory layout, build system, CI/CD, linting, testing framework, documentation skeleton, and CLI entry point. This stage produces no functional code but establishes the foundation everything else builds on.

**Estimated time:** Weeks 1–2
**Dependencies:** None (this is the foundation)
**Blocks:** All other stages

---

## 1. Directory Structure

```
scModelForge/
├── src/
│   └── scmodelforge/
│       ├── __init__.py              # Version, top-level imports
│       ├── _constants.py            # Shared constants (default vocab size, etc.)
│       ├── _types.py                # Shared type aliases and protocols
│       ├── config/
│       │   ├── __init__.py
│       │   └── schema.py            # Pydantic/dataclass config schema
│       ├── data/
│       │   └── __init__.py
│       ├── tokenizers/
│       │   └── __init__.py
│       ├── models/
│       │   └── __init__.py
│       ├── training/
│       │   └── __init__.py
│       ├── eval/
│       │   └── __init__.py
│       └── cli.py                   # CLI entry point (click or typer)
├── tests/
│   ├── conftest.py                  # Shared fixtures (mini AnnData, gene vocab, etc.)
│   ├── data/                        # Test fixtures (small .h5ad files, gene lists)
│   ├── test_data/
│   ├── test_tokenizers/
│   ├── test_models/
│   ├── test_training/
│   └── test_eval/
├── docs/
│   ├── conf.py                      # Sphinx configuration
│   ├── index.md                     # Landing page
│   ├── getting_started.md
│   ├── api/                         # Auto-generated API docs
│   └── tutorials/                   # Step-by-step guides (Phase 2+)
├── configs/
│   └── examples/
│       └── geneformer_basic.yaml    # Reference config for docs/tests
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Lint + test on PR
│       ├── docs.yml                 # Build & deploy docs
│       └── release.yml              # PyPI publish on tag
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── .pre-commit-config.yaml
├── .gitignore                       # Already exists
└── LICENSE                          # Already exists (Apache 2.0)
```

**Key decisions:**
- `src/` layout (not flat) — prevents accidental imports of uninstalled code.
- Package name is `scmodelforge` (lowercase, no hyphens) for PEP 8 compliance. Project name remains `scModelForge`.
- Each module gets its own subdirectory under `src/scmodelforge/` and mirrored test directory.

---

## 2. Build System — `pyproject.toml`

Use `hatchling` as the build backend (lightweight, well-supported, used by many scverse packages).

### Core dependencies (Phase 1)

```toml
[project]
name = "scModelForge"
version = "0.1.0"
description = "A pretraining toolkit for single-cell foundation models"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [
    { name = "Ehsan", email = "TBD" },
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "anndata>=0.10",
    "scanpy>=1.9",
    "numpy>=1.23",
    "scipy>=1.9",
    "pandas>=1.5",
    "torch>=2.0",
    "pytorch-lightning>=2.0",
    "pyyaml>=6.0",
    "omegaconf>=2.3",
    "rich>=13.0",
    "click>=8.0",
    "tqdm>=4.60",
]

[project.optional-dependencies]
census = ["cellxgene-census>=1.0"]
eval = ["scib>=1.1", "pertpy>=0.6"]
wandb = ["wandb>=0.15"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.0",
    "ruff>=0.4",
    "mypy>=1.5",
    "pre-commit>=3.0",
]
docs = [
    "sphinx>=7.0",
    "myst-parser>=2.0",
    "sphinx-autodoc2>=0.5",
    "sphinx-book-theme>=1.0",
    "sphinx-copybutton>=0.5",
]
all = ["scModelForge[census,eval,wandb,dev,docs]"]

[project.scripts]
scmodelforge = "scmodelforge.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Tool configuration (in pyproject.toml)

```toml
[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "TCH"]
ignore = ["E501"]  # line length handled by formatter

[tool.ruff.lint.isort]
known-first-party = ["scmodelforge"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Relax initially, tighten over time

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
]
```

---

## 3. CI/CD — GitHub Actions

### `ci.yml` — Runs on every PR and push to main

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python-version }} }
      - run: pip install -e ".[dev]"
      - run: pytest -m "not slow and not gpu" --cov=scmodelforge --cov-report=xml
      - uses: codecov/codecov-action@v4  # Optional: upload coverage

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -e ".[dev]"
      - run: mypy src/scmodelforge/ --ignore-missing-imports
```

### `release.yml` — Publish to PyPI on GitHub Release

```yaml
name: Release
on:
  release:
    types: [published]
jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # Trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

---

## 4. Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=500]
```

---

## 5. Shared Infrastructure Code

### `src/scmodelforge/__init__.py`

```python
"""scModelForge: A pretraining toolkit for single-cell foundation models."""

__version__ = "0.1.0"
```

### `src/scmodelforge/_types.py`

Shared type aliases and protocols used across modules. Initially minimal:

```python
from typing import Protocol, Any
import torch
from anndata import AnnData

class TokenizedCell(Protocol):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    metadata: dict[str, Any]
```

### `src/scmodelforge/config/schema.py`

Top-level config schema using dataclasses + OmegaConf:

```python
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig

@dataclass
class DataConfig:
    source: str = "local"
    paths: list[str] = field(default_factory=list)
    # ... expanded in Stage 1

@dataclass
class TokenizerConfig:
    strategy: str = "rank_value"
    max_genes: int = 2048
    # ... expanded in Stage 2

@dataclass
class ModelConfig:
    architecture: str = "transformer_encoder"
    hidden_dim: int = 512
    # ... expanded in Stage 3

@dataclass
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 10
    # ... expanded in Stage 4

@dataclass
class EvalConfig:
    every_n_epochs: int = 2
    benchmarks: list[str] = field(default_factory=list)
    # ... expanded in Stage 5

@dataclass
class ScModelForgeConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
```

### `src/scmodelforge/cli.py`

Minimal CLI skeleton using `click`:

```python
import click

@click.group()
@click.version_option()
def main():
    """scModelForge: Train single-cell foundation models."""
    pass

@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def train(config: str):
    """Train a model from a YAML config."""
    click.echo(f"Training with config: {config}")
    # Will be implemented in Stage 4

@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def evaluate(config: str):
    """Evaluate a trained model."""
    click.echo(f"Evaluating with config: {config}")
    # Will be implemented in Stage 5
```

---

## 6. Test Infrastructure

### `tests/conftest.py`

Shared fixtures that all module tests will use:

```python
import pytest
import anndata as ad
import numpy as np
import scipy.sparse as sp

@pytest.fixture
def mini_adata():
    """A small AnnData object for testing (100 cells x 200 genes)."""
    n_obs, n_vars = 100, 200
    X = sp.random(n_obs, n_vars, density=0.3, format="csr")
    obs = pd.DataFrame({
        "cell_type": np.random.choice(["T cell", "B cell", "Monocyte"], n_obs),
        "batch": np.random.choice(["batch1", "batch2"], n_obs),
    }, index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame({
        "gene_name": [f"GENE_{i}" for i in range(n_vars)],
        "ensembl_id": [f"ENSG{i:011d}" for i in range(n_vars)],
    }, index=[f"GENE_{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)

@pytest.fixture
def gene_vocab(mini_adata):
    """A simple gene vocabulary mapping."""
    return {g: i for i, g in enumerate(mini_adata.var_names)}

@pytest.fixture
def tmp_h5ad(mini_adata, tmp_path):
    """Write mini_adata to a temporary .h5ad file."""
    path = tmp_path / "test.h5ad"
    mini_adata.write_h5ad(path)
    return path
```

### Initial smoke test

```python
# tests/test_smoke.py
def test_import():
    import scmodelforge
    assert hasattr(scmodelforge, "__version__")

def test_cli_help(cli_runner):
    from click.testing import CliRunner
    from scmodelforge.cli import main
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
```

---

## 7. Documentation Skeleton

### `docs/conf.py`

```python
project = "scModelForge"
extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx_copybutton",
]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/TBD/scModelForge",
    "use_repository_button": True,
}
myst_enable_extensions = ["colon_fence", "deflist"]
autodoc2_packages = ["../src/scmodelforge"]
```

### `docs/index.md`

```markdown
# scModelForge

A pretraining toolkit for single-cell foundation models.

## Quick Start
(Coming soon — see Phase 1 roadmap)

## Modules
- {doc}`api/data`
- {doc}`api/tokenizers`
- {doc}`api/models`
- {doc}`api/training`
- {doc}`api/eval`
```

---

## 8. Example Config

### `configs/examples/geneformer_basic.yaml`

```yaml
# Example: Train a Geneformer-style model on local data
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
  gene_vocab: human_protein_coding

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
  lr: 1.0e-4
  optimizer: adam
  scheduler: cosine

eval:
  every_n_epochs: 2
  benchmarks:
    - cell_type_annotation:
        dataset: tabula_sapiens
```

---

## 9. README.md

Write a concise README with:
- Project name and tagline
- Badges (CI, PyPI version, Python versions, license)
- One-paragraph description
- Installation instructions (`pip install scModelForge`)
- Quick start example (the YAML config + CLI command)
- Link to docs
- Contributing section (link to CONTRIBUTING.md)
- License (Apache 2.0)

---

## Checklist

- [ ] Create directory structure
- [ ] Write `pyproject.toml` with all dependency groups
- [ ] Write `src/scmodelforge/__init__.py` with version
- [ ] Write `src/scmodelforge/_types.py` with shared protocols
- [ ] Write `src/scmodelforge/config/schema.py` with config dataclasses
- [ ] Write `src/scmodelforge/cli.py` with skeleton CLI
- [ ] Write `tests/conftest.py` with shared fixtures
- [ ] Write `tests/test_smoke.py` with import and CLI tests
- [ ] Write `.pre-commit-config.yaml`
- [ ] Write `.github/workflows/ci.yml`
- [ ] Write `.github/workflows/release.yml`
- [ ] Write `docs/conf.py` and `docs/index.md`
- [ ] Write `configs/examples/geneformer_basic.yaml`
- [ ] Write `README.md`
- [ ] Write `CONTRIBUTING.md`
- [ ] Write `CHANGELOG.md`
- [ ] Verify `pip install -e ".[dev]"` works
- [ ] Verify `pytest` passes
- [ ] Verify `ruff check` and `ruff format` pass
- [ ] Verify `scmodelforge --help` works
- [ ] Publish placeholder to PyPI (optional, can defer)
