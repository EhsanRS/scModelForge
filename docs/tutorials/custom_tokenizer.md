# Building Custom Tokenizers

This tutorial demonstrates how to implement, register, and test a custom tokenization strategy for scModelForge. If you have researched a novel tokenization approach or need specialized handling of gene expression data, the registry pattern makes it straightforward to integrate your custom tokenizer with the existing training pipeline, CLI, and configuration system.

## Prerequisites

This is an advanced tutorial. You should be comfortable with:

- Python object-oriented programming and abstract base classes
- PyTorch tensor operations
- The basics of scModelForge architecture and configuration

If you are new to scModelForge, start with the quickstart guide first.

## Tokenizer Architecture Overview

scModelForge uses a modular tokenization system built around three core components:

1. **BaseTokenizer** (abstract base class): Defines the interface all tokenizers must implement
2. **TokenizedCell dataclass**: The output format returned by `tokenize()`, containing `input_ids`, `attention_mask`, `values`, and optional fields
3. **Registry pattern**: The `@register_tokenizer("name")` decorator and `get_tokenizer("name")` lookup function enable seamless integration

### The BaseTokenizer Interface

Every tokenizer must subclass `BaseTokenizer` and implement three abstract members:

```python
class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(
        self,
        expression: np.ndarray | torch.Tensor,
        gene_indices: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        """Convert a single cell's expression vector into model input."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size for the embedding layer."""
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name for this tokenization strategy."""
        ...
```

The base class provides `tokenize_batch()` and `_collate()` methods for batch processing and padding, which work for most use cases without modification.

### TokenizedCell Output Format

The `tokenize()` method returns a `TokenizedCell` dataclass with the following fields:

- `input_ids`: 1-D long tensor of token indices (shape: `(seq_len,)`)
- `attention_mask`: 1-D long tensor, `1` for real tokens, `0` for padding (shape: `(seq_len,)`)
- `values`: Optional 1-D float tensor of continuous expression values (shape: `(seq_len,)`)
- `bin_ids`: Optional 1-D long tensor of discrete bin indices (shape: `(seq_len,)`)
- `gene_indices`: 1-D long tensor of gene vocabulary indices for each position (shape: `(seq_len,)`)
- `metadata`: Pass-through dictionary for cell-level metadata

Batching and padding are handled automatically by the base class `_collate()` method.

## Example: Top-K Expression Tokenizer

We will build a simple but complete tokenizer that selects the top K most highly expressed genes and uses their expression values directly as continuous features. This demonstrates all the key patterns you need for any custom tokenizer.

### Step 1: Create the Implementation File

Create a new file in the tokenizers package:

```bash
touch src/scmodelforge/tokenizers/top_k_expression.py
```

### Step 2: Implement the Tokenizer Class

Here is the complete implementation:

```python
"""Top-K expression tokenizer — selects most highly expressed genes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from scmodelforge._constants import CLS_TOKEN_ID, PAD_TOKEN_ID
from scmodelforge.tokenizers._utils import ensure_tensor
from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell
from scmodelforge.tokenizers.registry import register_tokenizer

if TYPE_CHECKING:
    import numpy as np

    from scmodelforge.data.gene_vocab import GeneVocab


@register_tokenizer("top_k_expression")
class TopKExpressionTokenizer(BaseTokenizer):
    """Selects the top-K most highly expressed genes.

    Unlike rank-value tokenization (which ranks all non-zero genes),
    this strategy strictly limits the sequence to K genes, even when
    more genes are expressed. This can be useful for fixed-length
    models or memory-constrained training.

    Parameters
    ----------
    gene_vocab
        Gene vocabulary mapping gene names to indices.
    max_len
        Maximum sequence length (including CLS if enabled).
    k
        Number of top genes to select (default: 512).
    prepend_cls
        Whether to prepend a ``[CLS]`` token.
    """

    def __init__(
        self,
        gene_vocab: GeneVocab,
        max_len: int = 2048,
        k: int = 512,
        prepend_cls: bool = True,
    ) -> None:
        super().__init__(gene_vocab, max_len)
        self.k = k
        self.prepend_cls = prepend_cls

    @property
    def vocab_size(self) -> int:
        """Vocabulary size is the gene vocab size (special tokens included)."""
        return len(self.gene_vocab)

    @property
    def strategy_name(self) -> str:
        """Strategy identifier for logging and configuration."""
        return "top_k_expression"

    def tokenize(
        self,
        expression: np.ndarray | torch.Tensor,
        gene_indices: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        """Tokenize a single cell by selecting top-K expressed genes.

        Parameters
        ----------
        expression
            1-D array of expression values (same length as gene_indices).
        gene_indices
            1-D array of gene vocabulary indices.
        metadata
            Optional metadata dictionary (passed through).

        Returns
        -------
        TokenizedCell
            Tokenized representation with top-K genes, optionally
            prepended with ``[CLS]``.
        """
        # Convert inputs to tensors
        expr_t = ensure_tensor(expression, torch.float32)
        genes_t = ensure_tensor(gene_indices, torch.long)

        # Filter to non-zero expression
        nonzero_mask = expr_t > 0
        nonzero_expr = expr_t[nonzero_mask]
        nonzero_genes = genes_t[nonzero_mask]

        # Select top-K by expression value (descending)
        effective_k = min(self.k, len(nonzero_expr))
        if effective_k > 0:
            top_k_indices = torch.topk(nonzero_expr, effective_k, largest=True, sorted=True).indices
            selected_genes = nonzero_genes[top_k_indices]
            selected_values = nonzero_expr[top_k_indices]
        else:
            # No expressed genes — empty sequence
            selected_genes = torch.tensor([], dtype=torch.long)
            selected_values = torch.tensor([], dtype=torch.float32)

        # Prepend CLS token if enabled
        if self.prepend_cls:
            cls_id = torch.tensor([CLS_TOKEN_ID], dtype=torch.long)
            input_ids = torch.cat([cls_id, selected_genes])
            values = torch.cat([torch.zeros(1, dtype=torch.float32), selected_values])
        else:
            input_ids = selected_genes
            values = selected_values

        # Attention mask: 1 for all real tokens (no padding yet)
        attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long)

        return TokenizedCell(
            input_ids=input_ids,
            attention_mask=attention_mask,
            values=values,
            gene_indices=input_ids.clone(),
            metadata=metadata or {},
        )
```

### Step 3: Register the Tokenizer

The `@register_tokenizer("top_k_expression")` decorator automatically registers the class when the module is imported. The registry name must be unique across all tokenizers.

To make the tokenizer discoverable, import it in the package `__init__.py`:

```python
# src/scmodelforge/tokenizers/__init__.py

from scmodelforge.tokenizers.top_k_expression import TopKExpressionTokenizer

__all__ = [
    # ... existing exports ...
    "TopKExpressionTokenizer",
]
```

Once imported, the tokenizer is available via `get_tokenizer("top_k_expression", ...)`.

### Step 4: Understanding Collation

The base class `_collate()` method handles batch padding automatically:

- Pads sequences to the maximum length in the batch (not `max_len`)
- Pads `input_ids` with `PAD_TOKEN_ID`
- Pads `attention_mask` with `0`
- Pads `values` with `0.0`
- Stacks all tensors into batch format with shape `(batch_size, seq_len)`

You only need to override `_collate()` if you have non-standard batch processing requirements.

### Step 5: Add Configuration Support

If your tokenizer has custom parameters (like our `k` parameter), you can add them to `TokenizerConfig` in `src/scmodelforge/config/schema.py`:

```python
@dataclass
class TokenizerConfig:
    """Tokenization configuration."""

    strategy: str = "rank_value"
    max_len: int = 2048
    prepend_cls: bool = True
    # ... existing fields ...
    k: int = 512  # For top_k_expression strategy
```

Alternatively, pass custom parameters directly via `get_tokenizer()`:

```python
tokenizer = get_tokenizer(
    "top_k_expression",
    gene_vocab=vocab,
    max_len=2048,
    k=256,
    prepend_cls=True,
)
```

## Testing Your Tokenizer

Follow the project testing patterns to ensure correctness. Here is a minimal test suite:

```python
"""Tests for TopKExpressionTokenizer."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from scmodelforge._constants import CLS_TOKEN_ID, NUM_SPECIAL_TOKENS
from scmodelforge.tokenizers import get_tokenizer


class TestTopKTokenizer:
    def test_selects_top_k(self, small_vocab):
        """Should select exactly K most expressed genes."""
        tokenizer = get_tokenizer(
            "top_k_expression",
            gene_vocab=small_vocab,
            max_len=2048,
            k=3,
            prepend_cls=False,
        )
        # 10 genes with varying expression
        expression = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 0.0, 4.0, 0.0, 6.0, 7.0], dtype=np.float32)
        gene_indices = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)

        result = tokenizer.tokenize(expression, gene_indices)

        # Should select exactly 3 genes
        assert result.input_ids.shape[0] == 3
        # Top 3 values are 8.0, 7.0, 6.0
        assert result.values[0].item() == 8.0
        assert result.values[1].item() == 7.0
        assert result.values[2].item() == 6.0

    def test_prepend_cls(self, small_vocab):
        """CLS token should be prepended when enabled."""
        tokenizer = get_tokenizer(
            "top_k_expression",
            gene_vocab=small_vocab,
            max_len=2048,
            k=5,
            prepend_cls=True,
        )
        expression = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gene_indices = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 5, dtype=np.int64)

        result = tokenizer.tokenize(expression, gene_indices)

        # 5 genes + 1 CLS = 6
        assert result.input_ids.shape[0] == 6
        assert result.input_ids[0].item() == CLS_TOKEN_ID
        assert result.values[0].item() == 0.0

    def test_empty_cell(self, small_vocab):
        """All-zero expression should return empty sequence (or just CLS)."""
        tokenizer = get_tokenizer(
            "top_k_expression",
            gene_vocab=small_vocab,
            max_len=2048,
            k=10,
            prepend_cls=True,
        )
        expression = np.zeros(10, dtype=np.float32)
        gene_indices = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)

        result = tokenizer.tokenize(expression, gene_indices)

        # Only CLS token
        assert result.input_ids.shape[0] == 1
        assert result.input_ids[0].item() == CLS_TOKEN_ID

    def test_batch_collation(self, small_vocab):
        """Batch should be padded to longest sequence in batch."""
        tokenizer = get_tokenizer(
            "top_k_expression",
            gene_vocab=small_vocab,
            max_len=2048,
            k=5,
            prepend_cls=True,
        )
        # Cell 1: 3 non-zero genes
        expr1 = np.array([1.0, 2.0, 3.0, 0.0, 0.0], dtype=np.float32)
        # Cell 2: 5 non-zero genes
        expr2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 5, dtype=np.int64)

        batch = tokenizer.tokenize_batch([expr1, expr2], [genes, genes])

        # Cell 1: CLS + 3 = 4, Cell 2: CLS + 5 = 6 → pad to 6
        assert batch["input_ids"].shape == (2, 6)
        assert batch["attention_mask"][0].sum().item() == 4  # Cell 1 has 2 padding
        assert batch["attention_mask"][1].sum().item() == 6  # Cell 2 fully attended

    def test_vocab_size(self, small_vocab):
        """vocab_size should match the gene vocabulary."""
        tokenizer = get_tokenizer("top_k_expression", gene_vocab=small_vocab, k=10)
        assert tokenizer.vocab_size == len(small_vocab)

    def test_strategy_name(self, small_vocab):
        """strategy_name should match registry key."""
        tokenizer = get_tokenizer("top_k_expression", gene_vocab=small_vocab, k=10)
        assert tokenizer.strategy_name == "top_k_expression"
```

Run the tests:

```bash
.venv/bin/python -m pytest tests/test_tokenizers/test_top_k_expression.py -v
```

## Using Your Custom Tokenizer

### From Python API

```python
from scmodelforge.data import GeneVocab
from scmodelforge.tokenizers import get_tokenizer

# Load or create gene vocabulary
vocab = GeneVocab.from_anndata(adata)

# Instantiate your custom tokenizer
tokenizer = get_tokenizer(
    "top_k_expression",
    gene_vocab=vocab,
    max_len=2048,
    k=512,
    prepend_cls=True,
)

# Tokenize a cell
result = tokenizer.tokenize(expression, gene_indices)
```

### From YAML Configuration

```yaml
data:
  gene_vocab_file: data/gene_vocab.json

tokenizer:
  strategy: top_k_expression
  max_len: 2048
  prepend_cls: true
  k: 512

model:
  architecture: transformer_encoder
  # ...

training:
  # ...
```

Then run training:

```bash
scmodelforge train --config configs/my_config.yaml
```

The training pipeline will automatically instantiate your tokenizer from the config.

## Distributing as a Plugin

To distribute your custom tokenizer as a pip-installable package, use Python entry points. In your plugin package `setup.py` or `pyproject.toml`:

```toml
[project.entry-points."scmodelforge.tokenizers"]
top_k_expression = "my_tokenizers:TopKExpressionTokenizer"
```

When installed, scModelForge will automatically discover and register your tokenizer, making it available via `get_tokenizer("top_k_expression")` without any imports.

For more details on plugin development, see the plugin development guide.

## Key Design Principles

When implementing custom tokenizers, keep these principles in mind:

1. **Always return TokenizedCell**: The `tokenize()` method must return a properly formatted `TokenizedCell` with all required fields (`input_ids`, `attention_mask`). Optional fields like `values` and `bin_ids` can be `None`.

2. **Handle CLS token correctly**: If `prepend_cls=True`, always prepend `CLS_TOKEN_ID` to `input_ids` and a corresponding zero or placeholder value to other fields like `values`.

3. **Pad to max_len in tokenize()**: The single-cell `tokenize()` method should NOT pad to `max_len`. Return the natural sequence length. The `_collate()` method handles batch-level padding.

4. **Use unique registry names**: Choose a descriptive, unique name for `@register_tokenizer("name")` to avoid conflicts with other tokenizers.

5. **Leverage base class utilities**: The default `_collate()` implementation works for most use cases. Only override if you need custom batch processing logic.

6. **Ensure tensor types**: Use `ensure_tensor()` from `scmodelforge.tokenizers._utils` to handle both numpy arrays and torch tensors uniformly.

7. **Document your strategy**: Provide clear docstrings explaining the tokenization logic, parameters, and use cases.

## Advanced Topics

### Custom Collation

If you need non-standard batch processing (for example, dynamic masking per batch or special handling of metadata), override `_collate()`:

```python
def _collate(self, cells: list[TokenizedCell]) -> dict[str, torch.Tensor]:
    """Custom collation with per-batch processing."""
    # Call parent implementation first
    batch = super()._collate(cells)

    # Add custom batch-level tensors
    batch["custom_feature"] = torch.tensor([...])

    return batch
```

### Integration with Masking

Your tokenizer output will be passed to `MaskingStrategy` during pretraining. Ensure your `TokenizedCell` contains the fields required by the masking strategy:

- `input_ids`: Required for BERT-style masking
- `bin_ids`: Required if using binned expression prediction
- `values`: Required if using continuous expression prediction

### Multi-field Tokenization

If your tokenization strategy produces multiple token types (for example, gene tokens and expression bin tokens as separate vocabularies), you can add custom fields to `TokenizedCell` or create a subclass:

```python
@dataclass
class MultiFieldTokenizedCell(TokenizedCell):
    """Extended cell with secondary token field."""
    secondary_ids: torch.Tensor | None = None
```

Then update `_collate()` to handle the new field.

## Summary

You have learned how to:

- Subclass `BaseTokenizer` and implement the required abstract methods
- Register your tokenizer using the `@register_tokenizer("name")` decorator
- Handle CLS tokens, padding, and batch collation
- Write comprehensive tests following project conventions
- Use your custom tokenizer from Python and YAML configuration
- Distribute tokenizers as pip-installable plugins

The registry pattern and abstract base class make it straightforward to experiment with novel tokenization strategies while maintaining full compatibility with scModelForge's training infrastructure.

For more examples, review the built-in tokenizers in `src/scmodelforge/tokenizers/`:

- `rank_value.py`: Rank-based gene selection
- `binned_expression.py`: Discrete binning
- `continuous_projection.py`: Learned continuous projection
- `gene_embedding.py`: Pretrained gene embeddings

Happy tokenizing!
