# Plugin Development

Learn how to create pip-installable plugins that extend scModelForge with custom tokenizers, models, and benchmarks.

## Overview

scModelForge's plugin system uses Python entry points, the standard mechanism for package extensibility. When users `pip install` your plugin, its components automatically become available in scModelForge's registries without requiring source code modification.

scModelForge defines three entry-point groups:

- `scmodelforge.tokenizers` for custom tokenization strategies
- `scmodelforge.models` for new model architectures
- `scmodelforge.benchmarks` for evaluation tasks

Your plugin's `pyproject.toml` declares entry points mapping names to classes. On first call to `get_*()` or `list_*()`, scModelForge scans installed packages and loads plugins. Discovery is lazy, built-in components take precedence, and loading errors in one plugin do not prevent others from loading

## Creating a Tokenizer Plugin

### Step 1: Package Structure

Standard Python package with src-layout:

```
scmodelforge-my-tokenizer/
  pyproject.toml
  src/my_tokenizer/__init__.py, tokenizer.py
  tests/test_tokenizer.py
```

### Step 2: Implement Your Tokenizer

Create your tokenizer in `src/my_tokenizer/tokenizer.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell

if TYPE_CHECKING:
    import numpy as np


class MyCustomTokenizer(BaseTokenizer):
    """Custom tokenization strategy."""

    def __init__(self, gene_vocab, max_genes: int = 2048, custom_param: float = 1.0) -> None:
        super().__init__(gene_vocab=gene_vocab, max_genes=max_genes)
        self.custom_param = custom_param

    @property
    def vocab_size(self) -> int:
        return len(self.gene_vocab) + self._special_tokens_count

    @property
    def strategy_name(self) -> str:
        return "my_custom"

    def tokenize(self, expression: np.ndarray, gene_indices: np.ndarray, metadata=None) -> TokenizedCell:
        # Filter zero/low expression
        nonzero_mask = expression > 0.1
        expr_nz = expression[nonzero_mask]
        gene_idx_nz = gene_indices[nonzero_mask]

        # Apply custom ranking
        scores = expr_nz * self.custom_param
        top_k_indices = torch.topk(
            torch.from_numpy(scores), k=min(len(scores), self.max_genes), sorted=True
        ).indices

        selected_genes = gene_idx_nz[top_k_indices.numpy()]
        selected_values = expr_nz[top_k_indices.numpy()]

        return TokenizedCell(
            input_ids=torch.from_numpy(selected_genes).long(),
            attention_mask=torch.ones(len(selected_genes)),
            values=torch.from_numpy(selected_values).float(),
            gene_indices=torch.from_numpy(selected_genes).long(),
            metadata=metadata or {},
        )
```

Key points: Inherit from `BaseTokenizer`, implement `vocab_size`, `strategy_name`, and `tokenize()`. Do NOT use `@register_tokenizer` (entry points handle registration). Follow conventions: `from __future__ import annotations`, `TYPE_CHECKING` blocks

### Step 3: Configure Entry Points

Create `pyproject.toml` with entry-point declarations:

```toml
[project]
name = "scmodelforge-my-tokenizer"
version = "0.1.0"
description = "Custom tokenization strategy for scModelForge"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "scModelForge>=0.1",
    "torch>=2.0",
    "numpy>=1.24",
]

[project.entry-points."scmodelforge.tokenizers"]
my_custom = "my_tokenizer.tokenizer:MyCustomTokenizer"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

The entry-point format is: `name = "package.module:ClassName"`

- `name`: Registry key users will reference in configs
- `package.module`: Import path to the module
- `ClassName`: The class to load

### Step 4: Test Locally

Install in editable mode and verify:

```bash
pip install -e .
python -c "from scmodelforge.tokenizers import list_tokenizers; print(list_tokenizers())"
# Should include 'my_custom'

python -c "from scmodelforge.tokenizers import get_tokenizer; from scmodelforge.data import GeneVocab; tok = get_tokenizer('my_custom', gene_vocab=GeneVocab.from_genes(['A']), max_genes=100); print(tok.strategy_name)"
```

### Step 5: Write Tests

Create `tests/test_tokenizer.py`:

```python
from __future__ import annotations

import numpy as np
from scmodelforge.data import GeneVocab
from scmodelforge.tokenizers import get_tokenizer, list_tokenizers
from scmodelforge.tokenizers.base import TokenizedCell


def test_standalone():
    """Test standalone import."""
    from my_tokenizer.tokenizer import MyCustomTokenizer

    vocab = GeneVocab.from_genes(["A", "B", "C", "D", "E"])
    tok = MyCustomTokenizer(gene_vocab=vocab, max_genes=3)
    result = tok.tokenize(np.array([0.5, 2.0, 0.1, 3.0, 1.5]), np.array([0, 1, 2, 3, 4]))

    assert isinstance(result, TokenizedCell)
    assert len(result.input_ids) <= 3


def test_via_registry():
    """Test via scModelForge registry."""
    assert "my_custom" in list_tokenizers()
    tok = get_tokenizer("my_custom", gene_vocab=GeneVocab.from_genes(["A", "B"]), max_genes=10)
    assert tok.strategy_name == "my_custom"
```

Run: `pytest tests/ -v`

### Step 6: Use in YAML Configuration

Once installed, your tokenizer works seamlessly in YAML configs:

```yaml
data:
  path: /path/to/data.h5ad
  obs_key_label: cell_type

tokenizer:
  strategy: my_custom
  max_genes: 2048
  custom_param: 1.5

model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 6

training:
  max_epochs: 100
  batch_size: 32
```

Run training:

```bash
scmodelforge train --config my_config.yaml
```

### Step 7: Publish

Build and upload: `pip install build twine && python -m build && twine upload dist/*`

Users install with: `pip install scmodelforge-my-tokenizer`

## Creating a Model Plugin

Models require a `from_config()` class method that accepts `ModelConfig`. Example:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from scmodelforge.models.protocol import ModelOutput

if TYPE_CHECKING:
    import torch
    from scmodelforge.config.schema import ModelConfig


class MyCustomModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, custom_layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=custom_layers,
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> MyCustomModel:
        """Required factory method."""
        return cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            custom_layers=getattr(config, "custom_layers", 4),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> ModelOutput:
        x = self.embedding(input_ids)
        encoded = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        return ModelOutput(last_hidden_state=encoded, pooled_output=encoded[:, 0, :])

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for evaluation."""
        return self.forward(input_ids, attention_mask).pooled_output
```

Entry point: `[project.entry-points."scmodelforge.models"]` with `my_model = "my_package.model:MyCustomModel"`

## Creating a Benchmark Plugin

Benchmarks evaluate model quality. Example:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


class MyCustomBenchmark(BaseBenchmark):
    def __init__(self, obs_key_label: str = "cell_type", custom_threshold: float = 0.5) -> None:
        super().__init__()
        self.obs_key_label = obs_key_label
        self.custom_threshold = custom_threshold

    @property
    def name(self) -> str:
        return "my_custom"

    @property
    def required_obs_keys(self) -> list[str]:
        return [self.obs_key_label]

    def run(self, embeddings: np.ndarray, adata: AnnData, dataset_name: str = "unknown") -> BenchmarkResult:
        """Run evaluation on precomputed embeddings."""
        labels = adata.obs[self.obs_key_label].values

        # Compute metrics (your logic here)
        metric1 = self._compute_metric(embeddings, labels)

        return BenchmarkResult(
            benchmark_name="my_custom",
            dataset_name=dataset_name,
            metrics={"custom_metric": float(metric1)},
            metadata={"n_cells": len(adata), "threshold": self.custom_threshold},
        )

    def _compute_metric(self, embeddings, labels) -> float:
        return 0.85  # Your evaluation logic
```

Entry point: `[project.entry-points."scmodelforge.benchmarks"]` with `my_custom = "my_package.benchmark:MyCustomBenchmark"`

Use in YAML: `benchmarks: [{name: my_custom, obs_key_label: cell_type, custom_threshold: 0.75}]`

## Best Practices

Choose unique names to avoid collisions with built-ins. Consider organization prefixes: `mylab_tokenizer` instead of `custom`.

Declare `scModelForge>=0.1` as a dependency with minimum versions:

```toml
dependencies = ["scModelForge>=0.1", "torch>=2.0", "numpy>=1.24"]
[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.1"]
```

Test both standalone import and registry loading. Run full pipelines (train, finetune, benchmark) for integration testing.

Document with clear docstrings. Include README with installation, usage examples, supported versions, and citations.

Follow semantic versioning: `0.1.0` for initial release, `0.2.0` for backward-compatible features, `1.0.0` for breaking changes

## Troubleshooting

Plugin not appearing: Check `pip list | grep scmodelforge` and verify entry points with:

```python
from importlib.metadata import entry_points
for ep in entry_points(group='scmodelforge.tokenizers'):
    print(f'{ep.name}: {ep.value}')
```

Import errors: Enable debug logging with `logging.basicConfig(level=logging.DEBUG)` before importing. Check for typos in entry-point paths.

Name collisions: If your plugin name matches a built-in, it is skipped with a warning. Choose a different name.

## Advanced Topics

Register multiple components from one package:

```toml
[project.entry-points."scmodelforge.tokenizers"]
strategy_a = "my_package:TokenizerA"
strategy_b = "my_package:TokenizerB"
```

For complex parameters, extend config dataclasses with defaults for backward compatibility.

Lazy-load heavy dependencies inside methods to keep registration fast: `import heavy_lib` inside `run()` rather than at module level

## Summary

Plugin system enables distributed development without core code changes. Steps: create package with src-layout, implement component class, declare entry points in `pyproject.toml`, test locally, publish to PyPI. Plugins integrate seamlessly with YAML configs and CLI commands.
