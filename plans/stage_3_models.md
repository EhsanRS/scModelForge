# Stage 3: scModelForge.models

## Overview

The models module provides reference implementations of major single-cell foundation model architectures. These are not replicas of the original codebases but clean, standardised, and interoperable implementations that work with any scModelForge tokenizer and data pipeline.

**Core responsibility:** Define model architectures with a shared protocol so they can be trained, evaluated, and shared interchangeably.

**Dependencies:** Stage 0 (scaffolding), Stage 2 (tokenizers — models consume `TokenizedCell`)
**Blocks:** Stage 4 (training wraps models), Stage 5 (eval uses model outputs)

---

## Phase 1: Foundation (Months 1–3)

### Goals
- Define `ScModelForgeModel` protocol — the contract all models must satisfy
- Implement `TransformerEncoder` (BERT-style masked language model, Geneformer pattern)
- Make architecture fully configurable via YAML
- Reproduce Geneformer cell-type annotation results within 2% on Tabula Sapiens

### Architecture

```
TokenizedCell (from tokenizer)
     │
     ▼
┌─────────────────────────────┐
│ ScModelForgeModel protocol  │
│  .forward()                 │  ← training (returns loss + logits)
│  .encode()                  │  ← inference (returns cell embeddings)
│  .from_config()             │  ← construction from YAML config
└──────────┬──────────────────┘
           │
      ┌────┴────┐
      │         │
      ▼         ▼
 TransformerEncoder    (Phase 2: AutoregressiveTransformer, MaskedAutoencoder)
```

### File Structure

```
src/scmodelforge/models/
├── __init__.py              # Public API: ScModelForgeModel, TransformerEncoder, get_model
├── protocol.py              # ScModelForgeModel protocol definition
├── transformer_encoder.py   # BERT-style masked LM (Geneformer pattern)
├── components/
│   ├── __init__.py
│   ├── embeddings.py        # Gene embeddings, positional encodings
│   ├── attention.py         # Multi-head attention (standard + future: flash)
│   ├── heads.py             # Pretraining heads (masked gene prediction, etc.)
│   └── pooling.py           # CLS pooling, mean pooling, etc.
├── registry.py              # Model registry (string name → class)
└── _utils.py                # Weight initialisation, parameter counting
```

### Key Classes and Interfaces

#### `ScModelForgeModel` Protocol

```python
from typing import Protocol, Any

class ModelOutput:
    """Standard output from model forward pass."""
    loss: torch.Tensor | None           # Training loss (None during inference)
    logits: torch.Tensor | None         # Prediction logits
    embeddings: torch.Tensor | None     # Cell embeddings (from encoder)
    hidden_states: tuple[torch.Tensor, ...] | None  # All layer outputs (optional)

class ScModelForgeModel(Protocol):
    """Protocol that all scModelForge models must implement."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> ModelOutput:
        """Forward pass for training. Returns loss when labels provided."""
        ...

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Extract cell embeddings. Returns (batch_size, hidden_dim)."""
        ...

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ScModelForgeModel":
        """Construct model from a config object."""
        ...

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        ...
```

#### `TransformerEncoder` (Geneformer-style)

```python
class TransformerEncoder(nn.Module):
    """BERT-style transformer encoder for single-cell data.

    Architecture:
    - Gene embedding layer (gene vocab → hidden_dim)
    - Positional encoding (learned or sinusoidal)
    - N transformer encoder layers
    - CLS token pooling for cell embeddings
    - Masked gene prediction head for pretraining
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int | None = None,       # Default: 4 * hidden_dim
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pooling: str = "cls",              # "cls", "mean"
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        use_expression_values: bool = True,  # Add expression as continuous feature
        pretraining_task: str = "masked_gene_prediction",
    ):
        super().__init__()

        # Embeddings
        self.gene_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Optional: project expression values and add to gene embedding
        if use_expression_values:
            self.expression_projection = nn.Linear(1, hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim or 4 * hidden_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling
        self.pooling_strategy = pooling

        # Pretraining head
        self.pretraining_head = MaskedGenePredictionHead(hidden_dim, vocab_size)

        # Initialisation
        self._init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, S) gene indices
        attention_mask: torch.Tensor,    # (B, S)
        values: torch.Tensor | None = None,  # (B, S) expression values
        labels: torch.Tensor | None = None,  # (B, S) targets for masked positions
        **kwargs,
    ) -> ModelOutput:
        # 1. Embed genes
        x = self.gene_embedding(input_ids)

        # 2. Add positional encoding
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        # 3. Optionally add expression value information
        if values is not None and hasattr(self, "expression_projection"):
            expr_embed = self.expression_projection(values.unsqueeze(-1))
            x = x + expr_embed

        # 4. Create attention mask (True = ignore for nn.TransformerEncoder)
        src_key_padding_mask = ~attention_mask.bool()

        # 5. Encode
        hidden = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 6. Compute loss if labels provided
        loss = None
        logits = None
        if labels is not None:
            logits = self.pretraining_head(hidden)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # 7. Cell embedding via pooling
        embeddings = self._pool(hidden, attention_mask)

        return ModelOutput(
            loss=loss,
            logits=logits,
            embeddings=embeddings,
            hidden_states=None,
        )

    def encode(self, input_ids, attention_mask, values=None, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask, values)
            return output.embeddings

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "cls":
            return hidden[:, 0]  # CLS token
        elif self.pooling_strategy == "mean":
            mask_expanded = mask.unsqueeze(-1).float()
            return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1)

    def _init_weights(self):
        """Xavier uniform for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "TransformerEncoder":
        return cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )
```

#### Pretraining Heads

```python
class MaskedGenePredictionHead(nn.Module):
    """Predict masked gene identities from hidden states."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        return self.decoder(x)


class ExpressionPredictionHead(nn.Module):
    """Predict continuous expression values (for future MAE-style models)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        return self.output(x).squeeze(-1)
```

#### Embedding Components

```python
class GeneExpressionEmbedding(nn.Module):
    """Combined gene identity + expression value embedding."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_seq_len: int = 2048,
        use_expression: bool = True,
    ):
        super().__init__()
        self.gene_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        if use_expression:
            self.expression_proj = nn.Linear(1, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        gene_ids: torch.Tensor,
        positions: torch.Tensor,
        expression_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.gene_embedding(gene_ids) + self.position_embedding(positions)
        if expression_values is not None and hasattr(self, "expression_proj"):
            x = x + self.expression_proj(expression_values.unsqueeze(-1))
        return self.dropout(self.layer_norm(x))
```

### Config Integration

```yaml
model:
  architecture: transformer_encoder
  hidden_dim: 512
  num_layers: 12
  num_heads: 8
  ffn_dim: 2048               # Optional, default 4*hidden_dim
  dropout: 0.1
  max_seq_len: 2048
  pooling: cls                 # cls | mean
  activation: gelu
  use_expression_values: true
  pretraining_task: masked_gene_prediction
  # vocab_size is inferred from gene_vocab
```

### Model Registry

```python
_MODEL_REGISTRY: dict[str, type] = {}

def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str, config: ModelConfig) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name].from_config(config)

# Usage:
@register_model("transformer_encoder")
class TransformerEncoder(nn.Module):
    ...
```

### Tests (Phase 1)

- `test_protocol.py`: Verify TransformerEncoder satisfies ScModelForgeModel protocol.
- `test_transformer_encoder.py`:
  - Forward pass with correct shapes (input → output).
  - Loss computation with masked labels.
  - `encode()` returns correct embedding shape.
  - `from_config()` construction.
  - Gradient flow through all parameters.
- `test_embeddings.py`: Gene embedding, positional encoding, expression projection.
- `test_heads.py`: MaskedGenePredictionHead output shape, ExpressionPredictionHead.
- `test_registry.py`: Registration, lookup, error on unknown.
- `test_param_count.py`: Parameter counting for reference configurations.

### Reference Model Sizes

| Config | Layers | Hidden | Heads | Params | Notes |
|---|---|---|---|---|---|
| tiny | 2 | 128 | 4 | ~2M | For testing/debugging |
| small | 6 | 256 | 8 | ~12M | Quick experiments |
| base | 12 | 512 | 8 | ~85M | Geneformer-scale |
| large | 24 | 1024 | 16 | ~350M | scGPT-scale |

---

## Phase 2: Breadth (Months 4–6)

### AutoregressiveTransformer (scGPT-style)

GPT-style next-token prediction for gene expression.

```python
class AutoregressiveTransformer(nn.Module):
    """GPT-style autoregressive model for single-cell data.

    Key differences from TransformerEncoder:
    - Uses causal (lower-triangular) attention mask
    - Predicts next gene + expression value autoregressively
    - Two prediction heads: gene identity + expression value
    - Compatible with BinnedExpressionTokenizer
    """

    def __init__(
        self,
        vocab_size: int,
        n_bins: int = 51,           # For expression value prediction
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ...
    ): ...
```

### MaskedAutoencoder (scFoundation-style)

Predicts raw expression values from masked inputs.

```python
class MaskedAutoencoder(nn.Module):
    """MAE-style model that predicts continuous expression values.

    Key differences:
    - Encoder processes only unmasked tokens (efficiency)
    - Lightweight decoder reconstructs masked positions
    - Loss is MSE on expression values, not cross-entropy on token IDs
    - Compatible with ContinuousProjectionTokenizer
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 512,
        decoder_dim: int = 256,
        encoder_layers: int = 12,
        decoder_layers: int = 4,
        ...
    ): ...
```

### HuggingFace Hub Compatibility

- Save/load methods compatible with `huggingface_hub`.
- `model.save_pretrained(path)` and `Model.from_pretrained(path_or_hub_id)`.
- Stores config alongside weights for full reproducibility.

### Additional Phase 2 Files

```
src/scmodelforge/models/
├── ...existing...
├── autoregressive.py        # AutoregressiveTransformer
├── masked_autoencoder.py    # MaskedAutoencoder
└── hub.py                   # HuggingFace Hub save/load utilities
```

---

## Phase 3: Community & Scale (Months 7–12)

### FSDP Support

- Wrap models with PyTorch FSDP for >1B parameter training.
- Automatic sharding policy based on transformer layers.
- Activation checkpointing for memory efficiency.

### Custom Model API

- Clear documentation for implementing custom models.
- `ScModelForgeModel` protocol is the only requirement.
- Example: "Bring your own model in 50 lines of code."

### Model Zoo

- Collection of pretrained model weights on HuggingFace Hub.
- `scmodelforge://geneformer-base`, `scmodelforge://scgpt-base`, etc.
- Standardized model cards with training details and evaluation results.

---

## Checklist

### Phase 1
- [ ] Define `ModelOutput` dataclass
- [ ] Define `ScModelForgeModel` protocol
- [ ] Implement `GeneExpressionEmbedding` (gene + position + expression)
- [ ] Implement `MaskedGenePredictionHead`
- [ ] Implement `ExpressionPredictionHead`
- [ ] Implement `TransformerEncoder` with full forward/encode/from_config
- [ ] Implement model registry
- [ ] Define reference model size configs (tiny, small, base, large)
- [ ] Add config parsing for `model:` section
- [ ] Write comprehensive tests
- [ ] Verify correct gradient flow and training dynamics
- [ ] Write docstrings and API documentation

### Phase 2
- [ ] Implement `AutoregressiveTransformer` (scGPT-style)
- [ ] Implement `MaskedAutoencoder` (scFoundation-style)
- [ ] Add HuggingFace Hub save/load compatibility
- [ ] Ensure all models work with all tokenizers

### Phase 3
- [ ] Add FSDP wrapping utilities
- [ ] Create custom model documentation and examples
- [ ] Publish pretrained model zoo on HuggingFace Hub
