# Implement a New Sub-Component

Follow this guide to add a new reusable component to `src/scmodelforge/models/components/`. Components are building blocks that models compose: embeddings, prediction heads, attention mechanisms, and pooling strategies.

## Overview

**Existing components:**

| File | Components | Used by |
|------|-----------|---------|
| `embeddings.py` | `GeneExpressionEmbedding` | All 3 models |
| `heads.py` | `MaskedGenePredictionHead`, `BinPredictionHead`, `ExpressionPredictionHead` | Various models |
| `pooling.py` | `cls_pool()`, `mean_pool()` | All 3 models |
| `attention.py` | `generate_causal_mask()` | AutoregressiveTransformer |

**Decision: New file or extend existing?**

| If your component is... | Then... |
|------------------------|---------|
| A new embedding strategy | Add to `embeddings.py` or create new file |
| A new prediction head | Add to `heads.py` |
| A new pooling strategy | Add to `pooling.py` |
| A new attention mechanism (RoPE, flash, linear, etc.) | Add to `attention.py` or create new file |
| Something entirely new (e.g., Mamba block, perceiver cross-attn) | Create a new file |

## Step 1: Implementation

### Option A: Add to existing file

For simple additions (a new head, pooling function, etc.), add to the relevant file.

**Example — new prediction head in `heads.py`:**

```python
class ContrastivePredictionHead(nn.Module):
    """Projects hidden states for contrastive loss."""

    def __init__(self, hidden_dim: int, projection_dim: int = 128) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.projection = nn.Linear(hidden_dim, projection_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dense(hidden_states))
        return self.projection(x)
```

**Example — new pooling function in `pooling.py`:**

```python
def weighted_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted mean pooling over the sequence dimension."""
    if weights is None:
        return mean_pool(hidden_states, attention_mask)
    w = weights.unsqueeze(-1).float()
    if attention_mask is not None:
        w = w * attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * w).sum(dim=1)
    total = w.sum(dim=1).clamp(min=1)
    return summed / total
```

### Option B: Create a new file

For larger components that don't fit existing categories.

Create `src/scmodelforge/models/components/<component_name>.py`:

```python
"""<Description> component."""

from __future__ import annotations

import torch
import torch.nn as nn


class <ComponentClassName>(nn.Module):
    """<One-line description>.

    Parameters
    ----------
    <param> : <type>
        <Description>.
    """

    def __init__(self, ...) -> None:
        super().__init__()
        # Define layers

    def forward(self, ...) -> torch.Tensor:
        """<Description of forward pass>.

        Parameters
        ----------
        ...

        Returns
        -------
        torch.Tensor
            <Shape and description>.
        """
        ...
```

**File conventions:**
- `from __future__ import annotations` at the top
- Use `TYPE_CHECKING` for type-hint-only imports
- Each component is a standalone `nn.Module` or a pure function
- No registry needed — components are used directly by model classes
- Weight initialization handled by the model's `self.apply(init_weights)`

## Step 2: Export in `__init__.py`

Add to `src/scmodelforge/models/components/__init__.py`:

```python
# Add import
from scmodelforge.models.components.<file> import <ComponentClassName>

# Add to __all__
__all__ = [
    ...
    "<ComponentClassName>",
    ...
]
```

Also add to `src/scmodelforge/models/__init__.py` if you want top-level access:

```python
from scmodelforge.models.components import <ComponentClassName>

__all__ = [
    ...
    "<ComponentClassName>",
    ...
]
```

## Step 3: Tests

Add tests to the appropriate test file or create a new one.

**For a new head — add to `tests/test_models/test_heads.py`:**

```python
class TestContrastivePredictionHead:
    def test_output_shape(self):
        head = ContrastivePredictionHead(hidden_dim=64, projection_dim=128)
        x = torch.randn(2, 10, 64)
        out = head(x)
        assert out.shape == (2, 10, 128)

    def test_gradient_flows(self):
        head = ContrastivePredictionHead(hidden_dim=64, projection_dim=128)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
```

**For a new file — create `tests/test_models/test_<component_name>.py`:**

```python
"""Tests for <ComponentClassName>."""

from __future__ import annotations

import pytest
import torch

from scmodelforge.models.components.<file> import <ComponentClassName>


class Test<ComponentClassName>:
    def test_output_shape(self):
        component = <ComponentClassName>(...)
        x = torch.randn(2, 10, 64)  # (batch, seq, hidden)
        out = component(x)
        assert out.shape == (2, 10, 64)  # expected output shape

    def test_handles_variable_seq_len(self):
        component = <ComponentClassName>(...)
        for seq_len in [1, 5, 20, 100]:
            x = torch.randn(2, seq_len, 64)
            out = component(x)
            assert out.shape[1] == seq_len

    def test_gradient_flows(self):
        component = <ComponentClassName>(...)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = component(x)
        out.sum().backward()
        assert x.grad is not None

    def test_deterministic(self):
        component = <ComponentClassName>(...)
        component.eval()
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            out1 = component(x)
            out2 = component(x)
        assert torch.equal(out1, out2)
```

## Step 4: Use in a Model

Once the component exists, use it in your model:

```python
from scmodelforge.models.components.<file> import <ComponentClassName>

class MyNewModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.my_component = <ComponentClassName>(...)
        # ...
        self.apply(init_weights)

    def forward(self, ...):
        # Use the component
        out = self.my_component(x)
        ...
```

## Step 5: Verification

```bash
# Lint
.venv/bin/ruff check src/scmodelforge/models/components/ tests/test_models/

# Component tests
.venv/bin/python -m pytest tests/test_models/test_<component>.py -v

# Model module regression (components are tested through models too)
.venv/bin/python -m pytest tests/test_models/ -v
```

## Common Component Patterns

### Rotary Position Embedding (RoPE)

```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin
```

### Flash Attention Wrapper

```python
class FlashSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(attn_out.transpose(1, 2).reshape(B, S, -1))
```

### Custom Transformer Block

```python
class CustomTransformerBlock(nn.Module):
    """Pre-norm transformer block with custom attention."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = FlashSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

These patterns can be composed into full model architectures following `prompts/implement_model.md`.
