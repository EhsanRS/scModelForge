# Implement a New Tokenizer

Follow this guide to add a new tokenization strategy to scModelForge. Every tokenizer converts raw gene expression vectors into `TokenizedCell` objects that the training pipeline can consume.

## Overview

**Files to create:**
- `src/scmodelforge/tokenizers/<tokenizer_name>.py` — Tokenizer implementation
- `tests/test_tokenizers/test_<tokenizer_name>.py` — Tests

**Files to modify:**
- `src/scmodelforge/tokenizers/__init__.py` — Import + `__all__`
- `src/scmodelforge/config/schema.py` — New `TokenizerConfig` fields (if needed)
- `docs/api/tokenizers.md` — Documentation

## Background: Existing Tokenizers

| Name | Registry key | Strategy |
|------|-------------|----------|
| `RankValueTokenizer` | `"rank_value"` | Geneformer-style: sort genes by expression rank, use gene IDs as tokens |
| `BinnedExpressionTokenizer` | `"binned_expression"` | scGPT-style: discretize expression into bins, gene IDs + bin IDs |
| `ContinuousProjectionTokenizer` | `"continuous_projection"` | TranscriptFormer-style: pass continuous expression values directly |

## Step 1: Implementation File

Create `src/scmodelforge/tokenizers/<tokenizer_name>.py`:

```python
"""<Paper name>-style <description> tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from scmodelforge._constants import CLS_TOKEN_ID
from scmodelforge.tokenizers._utils import ensure_tensor  # and other utils as needed
from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell
from scmodelforge.tokenizers.registry import register_tokenizer

if TYPE_CHECKING:
    import numpy as np

    from scmodelforge.data.gene_vocab import GeneVocab


@register_tokenizer("<registry_name>")
class <TokenizerClassName>(BaseTokenizer):
    """<One-line description>.

    Parameters
    ----------
    gene_vocab : GeneVocab
        Gene vocabulary mapping gene names to token IDs.
    max_len : int
        Maximum sequence length (including CLS token if prepended).
    prepend_cls : bool
        Whether to prepend a CLS token.
    """

    def __init__(
        self,
        gene_vocab: GeneVocab,
        max_len: int = 2048,
        prepend_cls: bool = True,
        # Add tokenizer-specific parameters here
    ) -> None:
        super().__init__(gene_vocab, max_len)
        self.prepend_cls = prepend_cls
        # Store additional parameters

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.gene_vocab)

    @property
    def strategy_name(self) -> str:
        """Return registry name of this tokenizer."""
        return "<registry_name>"

    def tokenize(
        self,
        expression: np.ndarray | torch.Tensor,
        gene_indices: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        """Tokenize a single cell's expression vector.

        Parameters
        ----------
        expression : np.ndarray | torch.Tensor
            Raw expression values for each gene, shape ``(n_genes,)``.
        gene_indices : np.ndarray | torch.Tensor
            Token IDs for each gene (from GeneVocab), shape ``(n_genes,)``.
        metadata : dict | None
            Optional metadata to attach to the tokenized cell.

        Returns
        -------
        TokenizedCell
            Tokenized representation with input_ids, attention_mask, values, etc.
        """
        expr = ensure_tensor(expression, dtype=torch.float32)
        genes = ensure_tensor(gene_indices, dtype=torch.long)

        # ──────────────────────────────────────────
        # YOUR TOKENIZATION LOGIC HERE
        # ──────────────────────────────────────────
        # Common steps:
        # 1. Filter (e.g., remove zero-expression genes)
        # 2. Sort/order genes
        # 3. Truncate to max_len (accounting for CLS if prepended)
        # 4. Compute values tensor
        # 5. Optionally compute bin_ids

        # Example: filter non-zero, sort by expression descending
        nonzero_mask = expr > 0
        expr = expr[nonzero_mask]
        genes = genes[nonzero_mask]

        sorted_idx = torch.argsort(expr, descending=True)
        expr = expr[sorted_idx]
        genes = genes[sorted_idx]

        # Truncate
        effective_max = self.max_len - (1 if self.prepend_cls else 0)
        if len(genes) > effective_max:
            genes = genes[:effective_max]
            expr = expr[:effective_max]

        # Build input_ids and values
        if self.prepend_cls:
            input_ids = torch.cat([torch.tensor([CLS_TOKEN_ID], dtype=torch.long), genes])
            values = torch.cat([torch.tensor([0.0]), expr])
        else:
            input_ids = genes
            values = expr

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        return TokenizedCell(
            input_ids=input_ids,
            attention_mask=attention_mask,
            values=values,
            # bin_ids=bin_ids,  # Set if your tokenizer computes discrete bins
            gene_indices=input_ids.clone(),
            metadata=metadata or {},
        )
```

### Key design points

- `ensure_tensor()` from `_utils.py` converts numpy arrays to torch tensors
- Filter zero-expression genes (most tokenizers do this)
- Always respect `self.max_len` — truncate to fit
- If `prepend_cls=True`, the CLS token counts toward `max_len`
- `attention_mask` is all 1s — padding happens in `_collate()` during batching
- `TokenizedCell.bin_ids` is optional — only set if your tokenizer discretizes expressions
- The parent `BaseTokenizer._collate()` handles padding and batching automatically

### Available utilities in `_utils.py`

```python
from scmodelforge.tokenizers._utils import (
    ensure_tensor,               # np.ndarray | Tensor -> Tensor
    rank_genes_by_expression,    # Sort indices by expression descending
    compute_bin_edges,           # Compute bin edges from training data
    digitize_expression,         # Map continuous values to bin IDs
)
```

## Step 2: Register in `__init__.py`

Add to `src/scmodelforge/tokenizers/__init__.py`:

```python
# Add import (alphabetical order)
from scmodelforge.tokenizers.<tokenizer_name> import <TokenizerClassName>

# Add to __all__ (alphabetical order)
__all__ = [
    ...
    "<TokenizerClassName>",
    ...
]
```

## Step 3: Config Fields (if needed)

Add new fields to `TokenizerConfig` in `src/scmodelforge/config/schema.py`:

```python
@dataclass
class TokenizerConfig:
    strategy: str = "rank_value"
    ...
    # <TokenizerName>-specific options
    new_param: int = sensible_default
```

## Step 4: Tests

Create `tests/test_tokenizers/test_<tokenizer_name>.py`:

```python
"""Tests for <TokenizerClassName>."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scmodelforge._constants import CLS_TOKEN_ID, NUM_SPECIAL_TOKENS, PAD_TOKEN_ID
from scmodelforge.tokenizers.<tokenizer_name> import <TokenizerClassName>
from scmodelforge.tokenizers.base import TokenizedCell


# ── Fixtures (or use conftest.py fixtures: small_vocab, sample_expression, sample_gene_indices) ──


class TestTokenization:
    def test_returns_tokenized_cell(self, small_vocab, sample_expression, sample_gene_indices):
        tok = <TokenizerClassName>(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert isinstance(cell, TokenizedCell)

    def test_input_ids_dtype(self, small_vocab, sample_expression, sample_gene_indices):
        tok = <TokenizerClassName>(gene_vocab=small_vocab, max_len=20)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids.dtype == torch.long

    def test_attention_mask_all_ones(self, small_vocab, sample_expression, sample_gene_indices):
        tok = <TokenizerClassName>(gene_vocab=small_vocab, max_len=20)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert (cell.attention_mask == 1).all()

    def test_values_shape_matches_input_ids(self, small_vocab, sample_expression, sample_gene_indices):
        tok = <TokenizerClassName>(gene_vocab=small_vocab, max_len=20)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.values.shape == cell.input_ids.shape


class TestCLS:
    def test_prepend_cls(self, small_vocab, sample_expression, sample_gene_indices):
        tok = <TokenizerClassName>(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids[0].item() == CLS_TOKEN_ID

    def test_no_cls(self, small_vocab, sample_expression, sample_gene_indices):
        tok = <TokenizerClassName>(gene_vocab=small_vocab, max_len=20, prepend_cls=False)
        cell = tok.tokenize(sample_expression, sample_gene_indices)
        assert cell.input_ids[0].item() != CLS_TOKEN_ID


class TestTruncation:
    def test_respects_max_len(self, large_vocab, large_expression, large_gene_indices):
        max_len = 20
        tok = <TokenizerClassName>(gene_vocab=large_vocab, max_len=max_len, prepend_cls=True)
        cell = tok.tokenize(large_expression, large_gene_indices)
        assert cell.input_ids.shape[0] <= max_len


class TestBatching:
    def test_collate_pads_correctly(self, small_vocab, sample_expression, sample_gene_indices):
        tok = <TokenizerClassName>(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell1 = tok.tokenize(sample_expression, sample_gene_indices)
        # Shorter cell
        short_expr = sample_expression[:3]
        short_genes = sample_gene_indices[:3]
        cell2 = tok.tokenize(short_expr, short_genes)
        batch = tok._collate([cell1, cell2])
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[1] == cell1.input_ids.shape[0]  # padded to longest
        assert batch["attention_mask"][1, -1].item() == 0  # padding position


class TestEdgeCases:
    def test_all_zero_expression(self, small_vocab):
        expr = np.zeros(10, dtype=np.float32)
        genes = np.arange(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + 10, dtype=np.int64)
        tok = <TokenizerClassName>(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(expr, genes)
        assert isinstance(cell, TokenizedCell)

    def test_single_gene(self, small_vocab):
        expr = np.array([5.0], dtype=np.float32)
        genes = np.array([NUM_SPECIAL_TOKENS], dtype=np.int64)
        tok = <TokenizerClassName>(gene_vocab=small_vocab, max_len=20, prepend_cls=True)
        cell = tok.tokenize(expr, genes)
        assert isinstance(cell, TokenizedCell)


class TestProperties:
    def test_vocab_size(self, small_vocab):
        tok = <TokenizerClassName>(gene_vocab=small_vocab)
        assert tok.vocab_size == len(small_vocab)

    def test_strategy_name(self, small_vocab):
        tok = <TokenizerClassName>(gene_vocab=small_vocab)
        assert tok.strategy_name == "<registry_name>"

    def test_registry_registered(self):
        from scmodelforge.tokenizers.registry import list_tokenizers
        assert "<registry_name>" in list_tokenizers()
```

## Step 5: Verification

```bash
# Lint
.venv/bin/ruff check src/scmodelforge/tokenizers/<tokenizer_name>.py tests/test_tokenizers/test_<tokenizer_name>.py

# New tests
.venv/bin/python -m pytest tests/test_tokenizers/test_<tokenizer_name>.py -v

# Tokenizer module regression
.venv/bin/python -m pytest tests/test_tokenizers/ -v

# Full suite
.venv/bin/python -m pytest tests/ -v
```

## Tokenizer Contract Summary

Every tokenizer **must** provide:

| Member | Type | Description |
|--------|------|-------------|
| `__init__(gene_vocab, max_len, ...)` | Constructor | Store vocab and config |
| `tokenize(expression, gene_indices, metadata)` | Method | Convert one cell → `TokenizedCell` |
| `vocab_size` | Property | Size of gene vocabulary |
| `strategy_name` | Property | Registry name string |

Inherited from `BaseTokenizer` (no need to override):
- `tokenize_batch(expressions, gene_indices_list, metadata_list)` → batched `dict[str, Tensor]`
- `_collate(cells)` → pads and stacks `TokenizedCell` list into batch dict

### `TokenizedCell` fields

| Field | Type | Description |
|-------|------|-------------|
| `input_ids` | `Tensor (S,)` | Gene token IDs |
| `attention_mask` | `Tensor (S,)` | 1 for real tokens, 0 for padding |
| `values` | `Tensor (S,) \| None` | Expression values |
| `bin_ids` | `Tensor (S,) \| None` | Discrete bin IDs (optional) |
| `gene_indices` | `Tensor (S,)` | Original gene indices |
| `metadata` | `dict[str, Any]` | Arbitrary metadata |

The training pipeline's `TokenizedCellDataset.__getitem__` calls `tokenizer.tokenize()` on each cell, then the `MaskingStrategy` applies masking to produce a `MaskedTokenizedCell`.
