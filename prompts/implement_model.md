# Implement a New Model Architecture

Follow this guide to add a new model to scModelForge. Every model must follow the same contract so it works with the existing training pipeline, fine-tuning, and assessment.

## Overview

**Files to create:**
- `src/scmodelforge/models/<model_name>.py` — Model implementation
- `tests/test_models/test_<model_name>.py` — Tests

**Files to modify:**
- `src/scmodelforge/models/__init__.py` — Import + `__all__`
- `src/scmodelforge/config/schema.py` — New config fields (if needed)
- `docs/api/models.md` — Documentation

## Step 1: Implementation File

Create `src/scmodelforge/models/<model_name>.py`:

```python
"""<Paper name>-style <description> model for single-cell gene expression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from scmodelforge.models._utils import count_parameters, init_weights
from scmodelforge.models.components.embeddings import GeneExpressionEmbedding
from scmodelforge.models.components.heads import MaskedGenePredictionHead  # or other heads
from scmodelforge.models.components.pooling import cls_pool, mean_pool
from scmodelforge.models.protocol import ModelOutput
from scmodelforge.models.registry import register_model

if TYPE_CHECKING:
    from scmodelforge.config.schema import ModelConfig


@register_model("<registry_name>")
class <ModelClassName>(nn.Module):
    """<One-line description>.

    Parameters
    ----------
    vocab_size : int
        Gene vocabulary size (including special tokens).
    hidden_dim : int
        Hidden dimension of the model.
    num_layers : int
        Number of layers.
    num_heads : int
        Number of attention heads (if applicable).
    ffn_dim : int | None
        Feed-forward dimension. Defaults to 4 * hidden_dim.
    dropout : float
        Dropout rate.
    max_seq_len : int
        Maximum sequence length.
    pooling : str
        Pooling strategy ("cls" or "mean").
    activation : str
        Activation function name.
    use_expression_values : bool
        Whether to incorporate continuous expression values.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pooling: str = "cls",
        activation: str = "gelu",
        *,
        use_expression_values: bool = True,
        # Add model-specific parameters here
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self._pooling_strategy = pooling

        ffn_dim = ffn_dim or 4 * hidden_dim

        # 1. Embedding layer (reuse existing component)
        self.embedding = GeneExpressionEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_expression_values=use_expression_values,
        )

        # 2. Core model layers
        # TODO: Replace with your architecture's layers
        # Example for transformer:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # 3. Prediction head
        self.head = MaskedGenePredictionHead(hidden_dim, vocab_size)

        # 4. Initialize weights (MUST be last)
        self.apply(init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Gene token IDs, shape ``(B, S)``.
        attention_mask : torch.Tensor
            1 for real tokens, 0 for padding, shape ``(B, S)``.
        values : torch.Tensor | None
            Expression values, shape ``(B, S)``.
        labels : torch.Tensor | None
            Target gene IDs at masked positions, -100 elsewhere, shape ``(B, S)``.

        Returns
        -------
        ModelOutput
            With loss (if labels provided), logits, and embeddings.
        """
        # Embed
        hidden = self.embedding(input_ids, values)

        # Encode
        # Convert attention_mask to the format your architecture needs
        src_key_padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)

        # Pool for embeddings
        embeddings = self._pool(hidden, attention_mask)

        # Predict
        logits = self.head(hidden)

        # Loss
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(loss=loss, logits=logits, embeddings=embeddings)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract embeddings without prediction head.

        Returns
        -------
        torch.Tensor
            Cell embeddings, shape ``(B, hidden_dim)``.
        """
        hidden = self.embedding(input_ids, values)
        src_key_padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)
        return self._pool(hidden, attention_mask)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply the configured pooling strategy."""
        if self._pooling_strategy == "cls":
            return cls_pool(hidden, attention_mask)
        if self._pooling_strategy == "mean":
            return mean_pool(hidden, attention_mask)
        msg = f"Unknown pooling strategy: {self._pooling_strategy!r}"
        raise ValueError(msg)

    @classmethod
    def from_config(cls, config: ModelConfig) -> <ModelClassName>:
        """Create model from a :class:`ModelConfig`."""
        if config.vocab_size is None:
            msg = "ModelConfig.vocab_size must be set before constructing a model."
            raise ValueError(msg)
        return cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            pooling=config.pooling,
            activation=config.activation,
            use_expression_values=config.use_expression_values,
            # Map any model-specific config fields here
        )

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        """Return number of parameters."""
        return count_parameters(self, trainable_only=trainable_only)
```

## Step 2: Register in `__init__.py`

Add to `src/scmodelforge/models/__init__.py`:

```python
# Add import (alphabetical order)
from scmodelforge.models.<model_name> import <ModelClassName>

# Add to __all__ (alphabetical order)
__all__ = [
    ...
    "<ModelClassName>",
    ...
]
```

**Critical:** The import in `__init__.py` triggers the `@register_model` decorator. Without it, the model won't appear in `list_models()`.

## Step 3: Config Fields (if needed)

Add new fields to `ModelConfig` in `src/scmodelforge/config/schema.py`:

```python
@dataclass
class ModelConfig:
    ...
    # <ModelName>-specific options
    new_param: int = sensible_default  # Brief description
```

**Rules:**
- Always provide sensible defaults so existing configs aren't broken
- Use `field(default_factory=...)` for mutable defaults (lists, dicts)
- Optional nested configs use `ConfigType | None = None`

## Step 4: Tests

Create `tests/test_models/test_<model_name>.py`:

```python
"""Tests for <ModelClassName>."""

from __future__ import annotations

import pytest
import torch

from scmodelforge.config.schema import ModelConfig
from scmodelforge.models.<model_name> import <ModelClassName>
from scmodelforge.models.protocol import ModelOutput


@pytest.fixture()
def config() -> ModelConfig:
    """Tiny config for fast tests."""
    return ModelConfig(
        architecture="<registry_name>",
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        dropout=0.0,
        max_seq_len=32,
        pooling="cls",
        activation="gelu",
        use_expression_values=True,
        # Set model-specific fields
    )


@pytest.fixture()
def batch() -> dict[str, torch.Tensor]:
    """Minimal batch: B=2, S=10."""
    torch.manual_seed(42)
    B, S, V = 2, 10, 100
    input_ids = torch.randint(1, V, (B, S))
    attention_mask = torch.ones(B, S, dtype=torch.long)
    attention_mask[1, -2:] = 0
    input_ids[1, -2:] = 0
    values = torch.rand(B, S)
    labels = torch.full((B, S), -100, dtype=torch.long)
    labels[0, 1] = input_ids[0, 1].item()
    labels[0, 3] = input_ids[0, 3].item()
    labels[1, 2] = input_ids[1, 2].item()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "values": values,
        "labels": labels,
    }


class TestForward:
    def test_returns_model_output(self, config, batch):
        model = <ModelClassName>.from_config(config)
        out = model(**batch)
        assert isinstance(out, ModelOutput)

    def test_embeddings_shape(self, config, batch):
        model = <ModelClassName>.from_config(config)
        out = model(**batch)
        assert out.embeddings.shape == (2, config.hidden_dim)

    def test_logits_shape(self, config, batch):
        model = <ModelClassName>.from_config(config)
        out = model(**batch)
        S = batch["input_ids"].size(1)
        assert out.logits.shape == (2, S, config.vocab_size)

    def test_with_labels_has_loss(self, config, batch):
        model = <ModelClassName>.from_config(config)
        out = model(**batch)
        assert out.loss is not None
        assert out.loss.shape == ()
        assert out.loss.item() > 0

    def test_without_labels_no_loss(self, config, batch):
        model = <ModelClassName>.from_config(config)
        batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
        out = model(**batch_no_labels)
        assert out.loss is None


class TestEncode:
    def test_output_shape(self, config, batch):
        model = <ModelClassName>.from_config(config)
        emb = model.encode(batch["input_ids"], batch["attention_mask"], values=batch["values"])
        assert emb.shape == (2, config.hidden_dim)

    def test_no_grad_accumulation(self, config, batch):
        model = <ModelClassName>.from_config(config)
        model.eval()
        with torch.no_grad():
            emb = model.encode(batch["input_ids"], batch["attention_mask"])
        assert emb.shape == (2, config.hidden_dim)


class TestFromConfig:
    def test_creates_model(self, config):
        model = <ModelClassName>.from_config(config)
        assert isinstance(model, <ModelClassName>)

    def test_vocab_size_none_raises(self):
        config = ModelConfig(vocab_size=None)
        with pytest.raises(ValueError, match="vocab_size must be set"):
            <ModelClassName>.from_config(config)


class TestGradients:
    def test_gradient_flows(self, config, batch):
        model = <ModelClassName>.from_config(config)
        out = model(**batch)
        out.loss.backward()
        assert model.embedding.gene_embedding.weight.grad is not None


class TestPooling:
    def test_cls_pooling(self, config, batch):
        config_cls = ModelConfig(**{**config.__dict__, "pooling": "cls"})
        model = <ModelClassName>.from_config(config_cls)
        out = model(**batch)
        assert out.embeddings.shape == (2, config.hidden_dim)

    def test_mean_pooling(self, config, batch):
        config_mean = ModelConfig(**{**config.__dict__, "pooling": "mean"})
        model = <ModelClassName>.from_config(config_mean)
        out = model(**batch)
        assert out.embeddings.shape == (2, config.hidden_dim)

    def test_unknown_pooling_raises(self, config, batch):
        config_bad = ModelConfig(**{**config.__dict__, "pooling": "bad"})
        model = <ModelClassName>.from_config(config_bad)
        with pytest.raises(ValueError, match="Unknown pooling"):
            model(**batch)


class TestMisc:
    def test_num_parameters(self, config):
        model = <ModelClassName>.from_config(config)
        n = model.num_parameters()
        assert n > 0

    def test_registry_registered(self):
        from scmodelforge.models.registry import list_models
        assert "<registry_name>" in list_models()
```

## Step 5: Verification

```bash
# Lint
.venv/bin/ruff check src/scmodelforge/models/<model_name>.py tests/test_models/test_<model_name>.py

# New tests
.venv/bin/python -m pytest tests/test_models/test_<model_name>.py -v

# Model module regression
.venv/bin/python -m pytest tests/test_models/ -v

# Full suite
.venv/bin/python -m pytest tests/ -v
```

## Model Contract Summary

Every model **must** provide:

| Method | Signature | Returns |
|--------|-----------|---------|
| `__init__` | Model-specific params | — |
| `forward` | `(input_ids, attention_mask, values=None, labels=None, **kwargs)` | `ModelOutput` |
| `encode` | `(input_ids, attention_mask, values=None, **kwargs)` | `torch.Tensor (B, H)` |
| `from_config` | `(cls, config: ModelConfig)` | Instance |
| `num_parameters` | `(trainable_only=True)` | `int` |

`ModelOutput` is a frozen dataclass with: `loss`, `logits`, `embeddings`, `hidden_states` (all optional).

The training pipeline calls `model(**batch)` and expects `ModelOutput` back. The assessment pipeline calls `model.encode(...)` for embeddings. Fine-tuning wraps the model as a backbone and calls `encode()`.
