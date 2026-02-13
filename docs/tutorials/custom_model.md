# Building Custom Model Architectures

This tutorial shows how to implement a custom model architecture that integrates with scModelForge's training pipeline, configuration system, and assessment harness.

## Target Audience

Model developers and ML engineers who want to experiment with novel architectures for single-cell foundation models. We assume familiarity with PyTorch and neural network architectures.

## Introduction

scModelForge provides three built-in architectures:

- `TransformerEncoder` — BERT-style bidirectional transformer with masked gene prediction
- `AutoregressiveTransformer` — scGPT-style causal transformer with dual heads
- `MaskedAutoencoder` — asymmetric encoder-decoder for reconstruction

However, single-cell foundation model research is rapidly evolving, and you may want to implement your own architecture. This tutorial demonstrates how to build a custom model that works seamlessly with the existing training pipeline, CLI commands, and YAML configuration.

## Architecture Requirements

Every scModelForge model must satisfy these requirements:

1. Inherit from `torch.nn.Module`
2. Implement a `from_config(cls, config: ModelConfig)` classmethod
3. Return a `ModelOutput` from the `forward()` method
4. Implement an `encode(batch)` method that returns cell embeddings
5. Be registered with the `@register_model("name")` decorator

## The ModelOutput Protocol

All models must return a `ModelOutput` instance from their `forward()` method. This is a frozen dataclass that standardizes model outputs:

```python
from scmodelforge.models.protocol import ModelOutput

@dataclass(frozen=True)
class ModelOutput:
    """Standard output container for all scModelForge models.

    Attributes
    ----------
    loss
        Scalar training loss (present only when labels are provided).
    logits
        Token-level predictions of shape (B, S, V) where V is vocab size.
    embeddings
        Cell-level embeddings of shape (B, H) from pooling.
    hidden_states
        Per-layer hidden states, each of shape (B, S, H).
    """
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    embeddings: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
```

The training pipeline expects `loss` for optimization and `logits` for masked prediction tasks. The assessment harness uses `encode()` to extract `embeddings` for downstream benchmarks.

## Example: GenePoolingModel

We will implement a simple baseline model called `GenePoolingModel` that uses mean pooling over gene embeddings instead of attention. This is computationally cheaper than transformers and serves as a reasonable baseline for cell representation learning.

### Step 1: Implement the Model

Create a new file at `src/scmodelforge/models/gene_pooling.py`:

```python
"""Simple gene pooling baseline model for single-cell representation learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from scmodelforge.models._utils import count_parameters, init_weights
from scmodelforge.models.components.embeddings import GeneExpressionEmbedding
from scmodelforge.models.components.heads import MaskedGenePredictionHead
from scmodelforge.models.components.pooling import mean_pool
from scmodelforge.models.protocol import ModelOutput
from scmodelforge.models.registry import register_model

if TYPE_CHECKING:
    from scmodelforge.config.schema import ModelConfig


@register_model("gene_pooling")
class GenePoolingModel(nn.Module):
    """Baseline model using mean pooling over gene embeddings without attention.

    This architecture processes gene expression inputs through embeddings and a
    simple feed-forward layer, then pools to create cell-level representations.
    It is much faster than transformer-based models and serves as a useful baseline.

    Parameters
    ----------
    vocab_size
        Gene vocabulary size (including special tokens).
    hidden_dim
        Hidden dimension for embeddings and feed-forward layers.
    num_layers
        Number of feed-forward transformation layers.
    dropout
        Dropout probability.
    max_seq_len
        Maximum sequence length.
    use_expression_values
        Whether to use expression value projection in embeddings.
    layer_norm_eps
        Epsilon for LayerNorm layers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        *,
        use_expression_values: bool = True,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Embedding layer (reuses existing component)
        self.embedding = GeneExpressionEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_expression_values=use_expression_values,
            layer_norm_eps=layer_norm_eps,
        )

        # Stack of feed-forward layers with residual connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
            )
            for _ in range(num_layers)
        ])

        # Prediction head (reuses existing component)
        self.head = MaskedGenePredictionHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps,
        )

        # Initialize weights using the standard utility
        self.apply(init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Forward pass with optional masked gene prediction loss.

        Parameters
        ----------
        input_ids
            Token IDs of shape (B, S).
        attention_mask
            Mask of shape (B, S) — 1 for real tokens, 0 for padding.
        values
            Optional expression values of shape (B, S).
        labels
            Optional target token IDs of shape (B, S) for computing loss.
            Positions with value -100 are ignored.
        **kwargs
            Additional batch keys (e.g., masked_positions, bin_ids) — ignored.

        Returns
        -------
        ModelOutput
            Contains loss (if labels provided), logits, and embeddings.
        """
        # Compute embeddings
        emb = self.embedding(input_ids, values=values)  # (B, S, H)

        # Apply feed-forward layers with residual connections
        hidden = emb
        for layer in self.layers:
            hidden = hidden + layer(hidden)  # Residual connection

        # Pool to get cell-level embeddings (mean over non-padding tokens)
        embeddings = mean_pool(hidden, attention_mask)  # (B, H)

        # Predict gene tokens at each position
        logits = self.head(hidden)  # (B, S, V)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

        return ModelOutput(loss=loss, logits=logits, embeddings=embeddings)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract cell embeddings of shape (B, hidden_dim).

        This method is used by the assessment harness to extract embeddings
        for downstream tasks like cell type classification and batch integration.

        Parameters
        ----------
        input_ids
            Token IDs of shape (B, S).
        attention_mask
            Mask of shape (B, S).
        values
            Optional expression values of shape (B, S).
        **kwargs
            Additional batch keys — ignored.

        Returns
        -------
        torch.Tensor
            Cell embeddings of shape (B, H).
        """
        emb = self.embedding(input_ids, values=values)
        hidden = emb
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return mean_pool(hidden, attention_mask)

    @classmethod
    def from_config(cls, config: ModelConfig) -> GenePoolingModel:
        """Create a GenePoolingModel from a ModelConfig.

        Parameters
        ----------
        config
            Model configuration. vocab_size must be set (not None).

        Returns
        -------
        GenePoolingModel

        Raises
        ------
        ValueError
            If config.vocab_size is None.
        """
        if config.vocab_size is None:
            msg = "ModelConfig.vocab_size must be set before constructing a model."
            raise ValueError(msg)

        return cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            use_expression_values=config.use_expression_values,
        )

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count the number of parameters in this model.

        Parameters
        ----------
        trainable_only
            If True (default), count only trainable parameters.

        Returns
        -------
        int
            Total number of (trainable) parameters.
        """
        return count_parameters(self, trainable_only=trainable_only)
```

### Step 2: Register the Model

The `@register_model("gene_pooling")` decorator automatically adds the model to the registry. To ensure it is available at runtime, you must import it in the models package `__init__.py`:

```python
# src/scmodelforge/models/__init__.py

from scmodelforge.models.autoregressive import AutoregressiveTransformer
from scmodelforge.models.gene_pooling import GenePoolingModel
from scmodelforge.models.masked_autoencoder import MaskedAutoencoder
from scmodelforge.models.transformer_encoder import TransformerEncoder

__all__ = [
    "AutoregressiveTransformer",
    "GenePoolingModel",
    "MaskedAutoencoder",
    "TransformerEncoder",
]
```

Importing the module triggers the decorator and registers your model. Now `scmodelforge train` and other CLI commands can discover it by name.

### Step 3: Understand the Batch Dictionary

The training pipeline provides the following keys in the batch dictionary:

| Key | Shape | Description |
|-----|-------|-------------|
| `input_ids` | `(B, S)` | Gene token IDs (may include MASK tokens) |
| `attention_mask` | `(B, S)` | 1 for real tokens, 0 for padding |
| `values` | `(B, S)` | Expression values (continuous or binned) |
| `labels` | `(B, S)` | Original gene IDs at masked positions, -100 elsewhere |
| `masked_positions` | `(B, S)` | Boolean mask indicating which tokens were masked |
| `bin_ids` | `(B, S)` | Optional expression bin IDs (for binned tokenization) |

Your `forward()` method must accept `input_ids`, `attention_mask`, and optionally `values` and `labels`. The `**kwargs` pattern allows your model to ignore extra keys like `masked_positions` and `bin_ids` if you do not need them.

### Step 4: Reuse Existing Components

scModelForge provides reusable components in `src/scmodelforge/models/components/`:

**Embeddings**:
- `GeneExpressionEmbedding` — combines gene token embeddings, positional embeddings, and optional expression value projection.

**Heads**:
- `MaskedGenePredictionHead` — predicts gene tokens from hidden states.
- `BinPredictionHead` — predicts expression bin IDs (for binned tokenization).
- `ExpressionPredictionHead` — predicts continuous expression values (MAE-style).

**Pooling**:
- `cls_pool(hidden, mask)` — returns the first token (CLS) embedding.
- `mean_pool(hidden, mask)` — mean-pools over non-padding tokens.

Reusing these components ensures consistency across models and reduces code duplication.

### Step 5: Write Tests

Follow the testing patterns in `tests/test_models/`. A typical test suite includes:

```python
"""Tests for GenePoolingModel."""

from __future__ import annotations

import pytest
import torch

from scmodelforge.config.schema import ModelConfig
from scmodelforge.models.gene_pooling import GenePoolingModel
from scmodelforge.models.protocol import ModelOutput


class TestGenePoolingForward:
    """Tests for the forward pass."""

    def test_forward_returns_model_output(self, tiny_config, dummy_batch):
        model = GenePoolingModel.from_config(tiny_config)
        out = model(**dummy_batch)
        assert isinstance(out, ModelOutput)

    def test_forward_embeddings_shape(self, tiny_config, dummy_batch):
        model = GenePoolingModel.from_config(tiny_config)
        out = model(**dummy_batch)
        batch_size = dummy_batch["input_ids"].size(0)
        assert out.embeddings.shape == (batch_size, tiny_config.hidden_dim)

    def test_forward_logits_shape(self, tiny_config, dummy_batch):
        model = GenePoolingModel.from_config(tiny_config)
        out = model(**dummy_batch)
        batch_size, seq_len = dummy_batch["input_ids"].shape
        assert out.logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

    def test_forward_with_labels_has_loss(self, tiny_config, dummy_batch):
        model = GenePoolingModel.from_config(tiny_config)
        out = model(**dummy_batch)
        assert out.loss is not None
        assert out.loss.shape == ()
        assert out.loss.item() > 0

    def test_forward_without_labels_no_loss(self, tiny_config, dummy_batch):
        model = GenePoolingModel.from_config(tiny_config)
        batch_no_labels = {k: v for k, v in dummy_batch.items() if k != "labels"}
        out = model(**batch_no_labels)
        assert out.loss is None


class TestGenePoolingEncode:
    """Tests for the encode method."""

    def test_encode_output_shape(self, tiny_config, dummy_batch):
        model = GenePoolingModel.from_config(tiny_config)
        emb = model.encode(
            dummy_batch["input_ids"],
            dummy_batch["attention_mask"],
            values=dummy_batch["values"],
        )
        batch_size = dummy_batch["input_ids"].size(0)
        assert emb.shape == (batch_size, tiny_config.hidden_dim)

    def test_encode_matches_forward_embeddings(self, tiny_config, dummy_batch):
        model = GenePoolingModel.from_config(tiny_config)
        model.training = False
        with torch.no_grad():
            emb = model.encode(
                dummy_batch["input_ids"],
                dummy_batch["attention_mask"],
                values=dummy_batch["values"],
            )
            out = model(**dummy_batch)
        assert torch.allclose(emb, out.embeddings, atol=1e-6)


class TestGenePoolingConfig:
    """Tests for configuration and initialization."""

    def test_from_config_success(self, tiny_config):
        model = GenePoolingModel.from_config(tiny_config)
        assert model.vocab_size == tiny_config.vocab_size
        assert model.hidden_dim == tiny_config.hidden_dim

    def test_from_config_missing_vocab_size_raises(self):
        config = ModelConfig(vocab_size=None)
        with pytest.raises(ValueError, match="vocab_size must be set"):
            GenePoolingModel.from_config(config)

    def test_num_parameters(self, tiny_config):
        model = GenePoolingModel.from_config(tiny_config)
        total = model.num_parameters(trainable_only=False)
        trainable = model.num_parameters(trainable_only=True)
        assert total > 0
        assert trainable == total  # All params trainable by default


class TestGenePoolingTraining:
    """Tests for training behavior."""

    def test_loss_decreases_with_training(self, tiny_config, dummy_batch):
        model = GenePoolingModel.from_config(tiny_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initial loss
        out1 = model(**dummy_batch)
        initial_loss = out1.loss.item()

        # Training steps
        for _ in range(10):
            optimizer.zero_grad()
            out = model(**dummy_batch)
            out.loss.backward()
            optimizer.step()

        # Final loss
        out2 = model(**dummy_batch)
        final_loss = out2.loss.item()

        # Loss should decrease
        assert final_loss < initial_loss
```

Run tests with:

```bash
.venv/bin/python -m pytest tests/test_models/test_gene_pooling.py -v
```

### Step 6: Use in Configuration

Create a YAML config file for your model at `configs/examples/gene_pooling.yaml`:

```yaml
data:
  source: local
  paths:
    - /path/to/dataset.h5ad
  gene_vocab: human_protein_coding
  max_genes: 2048

tokenizer:
  strategy: rank_value
  masking:
    mask_ratio: 0.15
    mask_token_prob: 0.8
    random_token_prob: 0.1

model:
  architecture: gene_pooling
  hidden_dim: 256
  num_layers: 4
  dropout: 0.1
  max_seq_len: 2048
  use_expression_values: true
  pooling: mean

training:
  batch_size: 32
  max_epochs: 50
  optimizer:
    name: adamw
    lr: 1e-4
    weight_decay: 0.01
  scheduler:
    name: cosine_warmup
    warmup_steps: 1000

output:
  log_dir: ./logs/gene_pooling
  checkpoint_dir: ./checkpoints/gene_pooling
  save_top_k: 3
```

Then train your model:

```bash
scmodelforge train --config configs/examples/gene_pooling.yaml
```

The training pipeline will:
1. Load your configuration
2. Instantiate `GenePoolingModel` via the registry
3. Set up data loaders and tokenization
4. Train with the specified optimizer and scheduler
5. Save checkpoints to `./checkpoints/gene_pooling/`

### Step 7: Weight Initialization

The `init_weights()` utility from `scmodelforge.models._utils` provides sensible defaults:

- `nn.Linear`: Xavier uniform initialization for weights, zeros for biases
- `nn.Embedding`: Normal distribution (mean=0, std=0.02), zeros for padding index
- `nn.LayerNorm`: Ones for weight, zeros for bias

Apply it with:

```python
self.apply(init_weights)
```

This ensures consistent initialization across all models in the toolkit.

## Advanced Topics

### Multi-Head Architectures

If your model needs to predict multiple targets (e.g., gene tokens and expression bins), you can use multiple heads:

```python
from scmodelforge.models.components.heads import (
    MaskedGenePredictionHead,
    BinPredictionHead,
)

self.gene_head = MaskedGenePredictionHead(hidden_dim, vocab_size)
self.bin_head = BinPredictionHead(hidden_dim, n_bins)

def forward(self, input_ids, attention_mask, labels=None, bin_labels=None, **kwargs):
    hidden = self.embedding(input_ids)
    # ... process hidden states ...

    gene_logits = self.gene_head(hidden)
    bin_logits = self.bin_head(hidden)

    # Compute weighted loss
    loss = None
    if labels is not None and bin_labels is not None:
        gene_loss = nn.CrossEntropyLoss(ignore_index=-100)(
            gene_logits.view(-1, self.vocab_size), labels.view(-1)
        )
        bin_loss = nn.CrossEntropyLoss(ignore_index=-100)(
            bin_logits.view(-1, self.n_bins), bin_labels.view(-1)
        )
        loss = self.gene_loss_weight * gene_loss + self.bin_loss_weight * bin_loss

    return ModelOutput(loss=loss, logits=gene_logits, embeddings=embeddings)
```

See `AutoregressiveTransformer` for a complete example.

### Custom Attention Mechanisms

You can implement custom attention patterns:

```python
from scmodelforge.models.components.attention import generate_causal_mask

# Causal (autoregressive) attention
causal_mask = generate_causal_mask(seq_len, device=input_ids.device)
hidden = self.encoder(emb, mask=causal_mask)

# Windowed attention (custom)
def generate_window_mask(seq_len, window_size, device):
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 0.0
    return mask
```

### Continuous Expression Prediction

For MAE-style reconstruction of expression values:

```python
from scmodelforge.models.components.heads import ExpressionPredictionHead

self.expr_head = ExpressionPredictionHead(hidden_dim)

def forward(self, input_ids, values, labels=None, **kwargs):
    hidden = # ... forward pass ...

    # Predict expression values
    pred_values = self.expr_head(hidden)  # (B, S)

    # MSE loss at masked positions
    loss = None
    if labels is not None:
        masked_positions = labels != -100
        if masked_positions.any():
            loss = nn.MSELoss()(
                pred_values[masked_positions],
                values[masked_positions],
            )

    return ModelOutput(loss=loss, logits=None, embeddings=embeddings)
```

## Summary

To implement a custom model architecture:

1. Create a new file in `src/scmodelforge/models/`
2. Inherit from `nn.Module` and add `@register_model("name")`
3. Implement `__init__`, `forward`, `encode`, and `from_config`
4. Return `ModelOutput` from `forward` with `.loss` and `.logits`
5. Import in `models/__init__.py` and add to `__all__`
6. Write tests following existing patterns
7. Use in YAML config with `model.architecture: your_name`

By following these conventions, your model will seamlessly integrate with scModelForge's training pipeline, assessment harness, HuggingFace Hub export, and CLI commands.
