# `scmodelforge.models` — Model Architectures

Reference implementations of single-cell foundation model architectures.

## Overview

The `scmodelforge.models` module provides three distinct pretraining paradigms for single-cell gene expression data, each optimized for different learning objectives and data characteristics. All models share a common output protocol via `ModelOutput`, use reusable components from the `scmodelforge.models.components` subpackage, and integrate seamlessly with the training pipeline through a centralized registry system.

**Three Architecture Paradigms:**

1. **Masked Language Modeling (MLM)** — `TransformerEncoder` implements the BERT-style approach where random tokens are masked and the model learns to predict them from bidirectional context. This is the most widely applicable pretraining strategy, learning rich gene co-expression patterns and cell state representations. Best for general-purpose foundation models.

2. **Autoregressive Generation** — `AutoregressiveTransformer` follows the scGPT design with causal (left-to-right) attention and dual prediction heads for gene identity and expression bins. This architecture enables generative modeling of gene expression programs and sequential dependencies. Best when modeling gene regulatory cascades or generating synthetic cells.

3. **Masked Autoencoding (MAE)** — `MaskedAutoencoder` uses an asymmetric encoder-decoder design where the encoder processes only unmasked tokens (dense representation) and a lightweight decoder reconstructs expression values at masked positions using MSE loss. This approach is computationally efficient for high masking ratios and emphasizes continuous expression value prediction. Best for large-scale pretraining with limited compute.

**Shared Components:**

All models leverage reusable building blocks from `scmodelforge.models.components`:
- `GeneExpressionEmbedding` — Combines gene tokens, positional embeddings, and optional expression value projections
- Prediction heads — `MaskedGenePredictionHead`, `BinPredictionHead`, `ExpressionPredictionHead`
- Pooling functions — `cls_pool` (first token) and `mean_pool` (masked average)
- Attention utilities — `generate_causal_mask` for autoregressive models

**Model Selection Guide:**

- **TransformerEncoder**: Default choice for most pretraining tasks. Proven effectiveness in scBERT and Geneformer. Supports both rank-value and binned tokenization.
- **AutoregressiveTransformer**: Use when you need generative capabilities or want to model gene expression as a sequential process. Requires binned expression tokenization.
- **MaskedAutoencoder**: Best for very large datasets (millions of cells) where training speed is critical. Asymmetric design reduces computational cost during pretraining.

All models expose a unified API with `forward()` for training (returns `ModelOutput` with loss) and `encode()` for inference (returns cell embeddings). Models are registered by name and can be instantiated via `get_model()` with a `ModelConfig`.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| **Protocol & Output** | |
| `ModelOutput` | Frozen dataclass container for model outputs (loss, logits, embeddings, hidden states) |
| **Model Architectures** | |
| `TransformerEncoder` | BERT-style bidirectional transformer with masked language modeling |
| `AutoregressiveTransformer` | scGPT-style causal transformer with dual gene+expression prediction heads |
| `MaskedAutoencoder` | Asymmetric encoder-decoder with MSE reconstruction loss |
| **Components (scmodelforge.models.components)** | |
| `GeneExpressionEmbedding` | Gene token + positional + expression value embedding layer |
| `MaskedGenePredictionHead` | Vocabulary-size classification head for masked gene prediction |
| `BinPredictionHead` | Discrete bin classification head for expression levels |
| `ExpressionPredictionHead` | Single-value regression head for continuous expression prediction |
| `cls_pool()` | Extract first (CLS) token as cell embedding |
| `mean_pool()` | Mean-pool non-padding tokens as cell embedding |
| `generate_causal_mask()` | Create autoregressive attention mask (lower triangular) |
| **Registry** | |
| `register_model()` | Class decorator to register a model under a string name |
| `get_model()` | Instantiate a registered model by name from a `ModelConfig` |
| `list_models()` | Return list of registered model names |
| **Utilities** | |
| `init_weights()` | Xavier uniform initialization for linear layers, normal for embeddings |
| `count_parameters()` | Count total or trainable parameters in a model |

---

## Protocol & Output

### `ModelOutput`

```python
from scmodelforge.models import ModelOutput
```

Frozen dataclass that serves as the standard output container for all scModelForge models. Provides a consistent interface for accessing model predictions, embeddings, and training loss.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `loss` | `torch.Tensor \| None` | Scalar training loss (present only when `labels` are provided during training) |
| `logits` | `torch.Tensor \| None` | Token-level predictions of shape `(B, S, V)` where V is vocab size (MLM/autoregressive) or `(B, S)` for continuous values (MAE) |
| `embeddings` | `torch.Tensor \| None` | Cell-level embeddings of shape `(B, H)` from pooling hidden states |
| `hidden_states` | `tuple[torch.Tensor, ...] \| None` | Per-layer hidden states, each of shape `(B, S, H)` (currently unused, reserved for future features) |

**Example:**

```python
import torch
from scmodelforge.models import TransformerEncoder, ModelOutput

model = TransformerEncoder(vocab_size=5000, hidden_dim=512, num_layers=6, num_heads=8)

# Training mode (with labels)
input_ids = torch.randint(0, 5000, (4, 128))
attention_mask = torch.ones(4, 128)
labels = torch.randint(0, 5000, (4, 128))

output: ModelOutput = model(input_ids, attention_mask, labels=labels)
print(output.loss)        # Scalar tensor
print(output.logits.shape)  # (4, 128, 5000)
print(output.embeddings.shape)  # (4, 512)

# Inference mode (no labels)
output = model(input_ids, attention_mask)
print(output.loss)  # None
```

---

## Model Architectures

### `TransformerEncoder`

```python
from scmodelforge.models import TransformerEncoder
```

BERT-style bidirectional transformer encoder for single-cell gene expression. Uses pre-normalization (norm-first) transformer layers with full self-attention over the input sequence. Predicts masked gene tokens via cross-entropy loss and supports both CLS and mean pooling for cell embeddings.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | *required* | Gene vocabulary size (including special tokens like CLS, PAD, MASK) |
| `hidden_dim` | `int` | *required* | Hidden dimension of the transformer |
| `num_layers` | `int` | *required* | Number of transformer encoder layers |
| `num_heads` | `int` | *required* | Number of attention heads |
| `ffn_dim` | `int \| None` | `None` | Feed-forward intermediate dimension (defaults to `4 * hidden_dim`) |
| `dropout` | `float` | `0.1` | Dropout probability applied throughout the model |
| `max_seq_len` | `int` | `2048` | Maximum sequence length for positional embeddings |
| `pooling` | `str` | `"cls"` | Pooling strategy: `"cls"` (first token) or `"mean"` (average over non-padding) |
| `activation` | `str` | `"gelu"` | Activation function name for feed-forward layers |
| `use_expression_values` | `bool` | `True` | Whether to use expression value projection in embeddings |
| `layer_norm_eps` | `float` | `1e-12` | Epsilon for LayerNorm stability |

**Key Methods:**

#### `forward(input_ids, attention_mask, values=None, labels=None, **kwargs)`

Forward pass with optional masked gene prediction loss.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_ids` | `torch.Tensor` | Token IDs of shape `(B, S)` |
| `attention_mask` | `torch.Tensor` | Mask of shape `(B, S)` — 1 for real tokens, 0 for padding |
| `values` | `torch.Tensor \| None` | Optional expression values of shape `(B, S)` |
| `labels` | `torch.Tensor \| None` | Optional target token IDs of shape `(B, S)` for computing loss. Positions with value `-100` are ignored |

**Returns:** `ModelOutput` with `loss`, `logits` (shape `(B, S, vocab_size)`), and `embeddings` (shape `(B, hidden_dim)`)

#### `encode(input_ids, attention_mask, values=None, **kwargs)`

Extract cell embeddings of shape `(B, hidden_dim)` without computing predictions or loss. Use this method for downstream tasks like clustering, classification, or embedding visualization.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_ids` | `torch.Tensor` | Token IDs of shape `(B, S)` |
| `attention_mask` | `torch.Tensor` | Mask of shape `(B, S)` |
| `values` | `torch.Tensor \| None` | Optional expression values of shape `(B, S)` |

**Returns:** `torch.Tensor` of shape `(B, hidden_dim)`

#### `from_config(config)`

Class method to create a `TransformerEncoder` from a `ModelConfig` object.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ModelConfig` | Model configuration with `vocab_size` set (not `None`) |

**Returns:** `TransformerEncoder`

**Raises:** `ValueError` if `config.vocab_size` is `None`

#### `num_parameters(trainable_only=True)`

Count the number of parameters in this model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainable_only` | `bool` | `True` | If `True`, count only trainable parameters |

**Returns:** `int`

**Example:**

```python
from scmodelforge.models import TransformerEncoder
from scmodelforge.config.schema import ModelConfig
import torch

# Create from config (recommended)
config = ModelConfig(
    architecture="transformer_encoder",
    vocab_size=5000,
    hidden_dim=512,
    num_layers=12,
    num_heads=8,
    dropout=0.1,
    pooling="mean",
)
model = TransformerEncoder.from_config(config)

# Or instantiate directly
model = TransformerEncoder(
    vocab_size=5000,
    hidden_dim=512,
    num_layers=12,
    num_heads=8,
    pooling="mean",
)

print(f"Model has {model.num_parameters():,} parameters")
# Model has 45,234,560 parameters

# Training forward pass
batch = {
    "input_ids": torch.randint(0, 5000, (16, 256)),
    "attention_mask": torch.ones(16, 256),
    "values": torch.randn(16, 256),
    "labels": torch.randint(0, 5000, (16, 256)),
}

output = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    values=batch["values"],
    labels=batch["labels"],
)

loss = output.loss  # Scalar tensor
loss.backward()

# Inference - extract cell embeddings
embeddings = model.encode(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    values=batch["values"],
)
print(embeddings.shape)  # (16, 512)
```

---

### `AutoregressiveTransformer`

```python
from scmodelforge.models import AutoregressiveTransformer
```

Autoregressive (causal) transformer for single-cell gene expression following the scGPT architecture. Uses causal attention masking (left-to-right) and dual prediction heads for gene identity and expression bins. The combined loss is a weighted sum of both cross-entropy objectives.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | *required* | Gene vocabulary size (including special tokens) |
| `n_bins` | `int` | `51` | Number of expression bins for the bin prediction head |
| `hidden_dim` | `int` | `512` | Hidden dimension of the transformer |
| `num_layers` | `int` | `12` | Number of transformer encoder layers |
| `num_heads` | `int` | `8` | Number of attention heads |
| `ffn_dim` | `int \| None` | `None` | Feed-forward intermediate dimension (defaults to `4 * hidden_dim`) |
| `dropout` | `float` | `0.1` | Dropout probability |
| `max_seq_len` | `int` | `2048` | Maximum sequence length |
| `pooling` | `str` | `"cls"` | Pooling strategy: `"cls"` or `"mean"` |
| `activation` | `str` | `"gelu"` | Activation function name for feed-forward layers |
| `use_expression_values` | `bool` | `True` | Whether to use expression value projection in embeddings |
| `layer_norm_eps` | `float` | `1e-12` | Epsilon for LayerNorm |
| `gene_loss_weight` | `float` | `1.0` | Weight for the gene prediction loss component |
| `expression_loss_weight` | `float` | `1.0` | Weight for the expression bin prediction loss component |

**Key Methods:**

#### `forward(input_ids, attention_mask, values=None, labels=None, **kwargs)`

Forward pass with causal attention and dual prediction heads.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_ids` | `torch.Tensor` | Token IDs of shape `(B, S)` |
| `attention_mask` | `torch.Tensor` | Mask of shape `(B, S)` — 1 for real tokens, 0 for padding |
| `values` | `torch.Tensor \| None` | Optional expression values of shape `(B, S)` |
| `labels` | `torch.Tensor \| None` | Optional target token IDs of shape `(B, S)` for gene prediction loss. Positions with value `-100` are ignored |
| `**kwargs` | | Extra batch keys. `bin_ids` of shape `(B, S)` used as targets for expression bin prediction |

**Returns:** `ModelOutput` with `loss` (combined weighted loss), `logits` (gene predictions of shape `(B, S, vocab_size)`), and `embeddings` (shape `(B, hidden_dim)`)

**Note:** The loss combines gene and expression bin predictions when both `labels` and `bin_ids` are provided:
```
loss = gene_loss_weight * CrossEntropy(gene_logits, labels) +
       expression_loss_weight * CrossEntropy(bin_logits, bin_ids)
```

#### `encode(input_ids, attention_mask, values=None, **kwargs)`

Extract cell embeddings without causal masking. This method runs the full encoder on all tokens (no causal mask) and pools to produce cell embeddings. Use this for inference and downstream tasks.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_ids` | `torch.Tensor` | Token IDs of shape `(B, S)` |
| `attention_mask` | `torch.Tensor` | Mask of shape `(B, S)` |
| `values` | `torch.Tensor \| None` | Optional expression values of shape `(B, S)` |

**Returns:** `torch.Tensor` of shape `(B, hidden_dim)`

#### `from_config(config)`

Class method to create an `AutoregressiveTransformer` from a `ModelConfig`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ModelConfig` | Model configuration with `vocab_size` set (not `None`) |

**Returns:** `AutoregressiveTransformer`

**Raises:** `ValueError` if `config.vocab_size` is `None`

#### `num_parameters(trainable_only=True)`

Count the number of parameters in this model.

**Example:**

```python
from scmodelforge.models import AutoregressiveTransformer
from scmodelforge.config.schema import ModelConfig
import torch

# Create from config
config = ModelConfig(
    architecture="autoregressive_transformer",
    vocab_size=5000,
    n_bins=51,
    hidden_dim=512,
    num_layers=12,
    num_heads=8,
    gene_loss_weight=1.0,
    expression_loss_weight=0.5,
)
model = AutoregressiveTransformer.from_config(config)

# Training with dual heads
batch = {
    "input_ids": torch.randint(0, 5000, (8, 200)),
    "attention_mask": torch.ones(8, 200),
    "values": torch.randn(8, 200),
    "labels": torch.randint(0, 5000, (8, 200)),
    "bin_ids": torch.randint(0, 51, (8, 200)),
}

output = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    values=batch["values"],
    labels=batch["labels"],
    bin_ids=batch["bin_ids"],
)

print(output.loss)  # Combined loss: 1.0 * gene_loss + 0.5 * bin_loss
print(output.logits.shape)  # (8, 200, 5000) - gene predictions

# Inference - extract embeddings (no causal mask)
embeddings = model.encode(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    values=batch["values"],
)
print(embeddings.shape)  # (8, 512)
```

---

### `MaskedAutoencoder`

```python
from scmodelforge.models import MaskedAutoencoder
```

Asymmetric encoder-decoder masked autoencoder for single-cell expression following the scFoundation design. The encoder processes only unmasked tokens (dense representation) for computational efficiency. The decoder receives encoder outputs at unmasked positions and learnable mask tokens at masked positions, then predicts continuous expression values with MSE loss.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | *required* | Gene vocabulary size (including special tokens) |
| `encoder_dim` | `int` | `512` | Encoder hidden dimension |
| `decoder_dim` | `int \| None` | `None` | Decoder hidden dimension (defaults to `encoder_dim // 2`) |
| `encoder_layers` | `int` | `12` | Number of encoder transformer layers |
| `decoder_layers` | `int` | `4` | Number of decoder transformer layers |
| `encoder_heads` | `int` | `8` | Number of encoder attention heads |
| `decoder_heads` | `int \| None` | `None` | Number of decoder attention heads (defaults to `encoder_heads`) |
| `ffn_dim` | `int \| None` | `None` | Encoder feed-forward dimension (defaults to `4 * encoder_dim`) |
| `dropout` | `float` | `0.1` | Dropout probability |
| `max_seq_len` | `int` | `2048` | Maximum sequence length |
| `pooling` | `str` | `"mean"` | Pooling strategy: `"cls"` or `"mean"` |
| `activation` | `str` | `"gelu"` | Activation function name for feed-forward layers |
| `use_expression_values` | `bool` | `True` | Whether to use expression value projection in embeddings |
| `layer_norm_eps` | `float` | `1e-12` | Epsilon for LayerNorm |

**Key Methods:**

#### `forward(input_ids, attention_mask, values=None, labels=None, **kwargs)`

Forward pass with asymmetric encode-decode and MSE loss at masked positions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_ids` | `torch.Tensor` | Token IDs of shape `(B, S)` |
| `attention_mask` | `torch.Tensor` | Mask of shape `(B, S)` — 1 for real tokens, 0 for padding |
| `values` | `torch.Tensor \| None` | Expression values of shape `(B, S)` (used as reconstruction targets) |
| `labels` | `torch.Tensor \| None` | Not used directly for MAE loss. Presence triggers training mode; the loss target comes from `values` at masked positions |
| `**kwargs` | | Extra batch keys. `masked_positions` of shape `(B, S)` (bool or 0/1) indicates which positions are masked. If absent, inferred from `labels != -100` |

**Returns:** `ModelOutput` with `loss` (MSE at masked positions), `logits` (predicted expression values of shape `(B, S)`), and `embeddings` (shape `(B, encoder_dim)`)

**Note:** The encoder only processes unmasked tokens for efficiency. The decoder uses learnable mask tokens at masked positions and encoder outputs at unmasked positions.

#### `encode(input_ids, attention_mask, values=None, **kwargs)`

Extract cell embeddings using the full encoder without masking.

Processes all tokens (no masking) and pools to produce cell embeddings of shape `(B, encoder_dim)`. Use this for inference and downstream tasks.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_ids` | `torch.Tensor` | Token IDs of shape `(B, S)` |
| `attention_mask` | `torch.Tensor` | Mask of shape `(B, S)` |
| `values` | `torch.Tensor \| None` | Optional expression values of shape `(B, S)` |

**Returns:** `torch.Tensor` of shape `(B, encoder_dim)`

#### `from_config(config)`

Class method to create a `MaskedAutoencoder` from a `ModelConfig`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ModelConfig` | Model configuration with `vocab_size` set (not `None`) |

**Returns:** `MaskedAutoencoder`

**Raises:** `ValueError` if `config.vocab_size` is `None`

#### `num_parameters(trainable_only=True)`

Count the number of parameters in this model.

**Example:**

```python
from scmodelforge.models import MaskedAutoencoder
from scmodelforge.config.schema import ModelConfig
import torch

# Create asymmetric encoder-decoder
config = ModelConfig(
    architecture="masked_autoencoder",
    vocab_size=5000,
    hidden_dim=768,  # encoder_dim
    decoder_dim=384,  # half of encoder_dim
    num_layers=12,   # encoder_layers
    decoder_layers=4,
    num_heads=12,    # encoder_heads
    pooling="mean",
)
model = MaskedAutoencoder.from_config(config)

print(f"Encoder dim: {model.encoder_dim}, Decoder dim: {model.decoder_dim}")
# Encoder dim: 768, Decoder dim: 384

# Training with masked positions
batch_size, seq_len = 16, 300
batch = {
    "input_ids": torch.randint(0, 5000, (batch_size, seq_len)),
    "attention_mask": torch.ones(batch_size, seq_len),
    "values": torch.randn(batch_size, seq_len),
    "labels": torch.full((batch_size, seq_len), -100),  # trigger training mode
    "masked_positions": torch.rand(batch_size, seq_len) < 0.15,  # 15% masking
}

output = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    values=batch["values"],
    labels=batch["labels"],
    masked_positions=batch["masked_positions"],
)

print(output.loss)  # MSE loss at masked positions only
print(output.logits.shape)  # (16, 300) - predicted expression values

# Inference - extract embeddings
embeddings = model.encode(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    values=batch["values"],
)
print(embeddings.shape)  # (16, 768)
```

---

## Components

### `GeneExpressionEmbedding`

```python
from scmodelforge.models.components import GeneExpressionEmbedding
```

Combined embedding layer that sums gene token embeddings, learned positional embeddings, and optional expression value projections. Applied before the transformer encoder in all model architectures.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | *required* | Number of tokens in the gene vocabulary (including special tokens) |
| `hidden_dim` | `int` | *required* | Embedding dimension |
| `max_seq_len` | `int` | `2048` | Maximum sequence length for learned positional embeddings |
| `dropout` | `float` | `0.1` | Dropout probability applied after LayerNorm |
| `use_expression_values` | `bool` | `True` | If `True`, project scalar expression values to `hidden_dim` and add to embeddings |
| `layer_norm_eps` | `float` | `1e-12` | Epsilon for LayerNorm stability |

**Key Methods:**

#### `forward(input_ids, values=None)`

Compute embeddings for a batch of tokenized cells.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_ids` | `torch.Tensor` | Token IDs of shape `(B, S)` |
| `values` | `torch.Tensor \| None` | Optional expression values of shape `(B, S)` |

**Returns:** `torch.Tensor` of shape `(B, S, hidden_dim)`

**Example:**

```python
from scmodelforge.models.components import GeneExpressionEmbedding
import torch

embedding = GeneExpressionEmbedding(
    vocab_size=5000,
    hidden_dim=512,
    max_seq_len=2048,
    use_expression_values=True,
)

input_ids = torch.randint(0, 5000, (8, 256))
values = torch.randn(8, 256)

# With expression values
emb = embedding(input_ids, values=values)
print(emb.shape)  # (8, 256, 512)

# Without expression values (uses zeros)
emb = embedding(input_ids)
print(emb.shape)  # (8, 256, 512)
```

---

### `MaskedGenePredictionHead`

```python
from scmodelforge.models.components import MaskedGenePredictionHead
```

Prediction head for masked language modeling (MLM). Predicts gene token IDs from hidden states using a two-layer MLP followed by a linear projection to vocabulary size.

Architecture: `Linear → GELU → LayerNorm → Linear(vocab_size)`

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | *required* | Input hidden dimension |
| `vocab_size` | `int` | *required* | Output vocabulary size (number of gene tokens) |
| `layer_norm_eps` | `float` | `1e-12` | Epsilon for LayerNorm |

**Key Methods:**

#### `forward(hidden_states)`

Predict token logits from hidden states.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_states` | `torch.Tensor` | Shape `(B, S, hidden_dim)` |

**Returns:** `torch.Tensor` of shape `(B, S, vocab_size)` with unnormalized logits

**Example:**

```python
from scmodelforge.models.components import MaskedGenePredictionHead
import torch

head = MaskedGenePredictionHead(hidden_dim=512, vocab_size=5000)

hidden_states = torch.randn(4, 128, 512)
logits = head(hidden_states)
print(logits.shape)  # (4, 128, 5000)

# Compute cross-entropy loss for masked positions
labels = torch.randint(0, 5000, (4, 128))
labels[labels < 2500] = -100  # ignore unmasked positions

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fn(logits.view(-1, 5000), labels.view(-1))
```

---

### `BinPredictionHead`

```python
from scmodelforge.models.components import BinPredictionHead
```

Prediction head for discrete expression bin classification. Used in autoregressive models (scGPT-style) to predict which bin a gene's expression level falls into.

Architecture: `Linear → GELU → LayerNorm → Linear(n_bins)`

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | *required* | Input hidden dimension |
| `n_bins` | `int` | *required* | Number of expression bins (output classes) |
| `layer_norm_eps` | `float` | `1e-12` | Epsilon for LayerNorm |

**Key Methods:**

#### `forward(hidden_states)`

Predict bin logits from hidden states.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_states` | `torch.Tensor` | Shape `(B, S, hidden_dim)` |

**Returns:** `torch.Tensor` of shape `(B, S, n_bins)` with unnormalized logits

**Example:**

```python
from scmodelforge.models.components import BinPredictionHead
import torch

head = BinPredictionHead(hidden_dim=512, n_bins=51)

hidden_states = torch.randn(8, 200, 512)
bin_logits = head(hidden_states)
print(bin_logits.shape)  # (8, 200, 51)

# Compute loss
bin_labels = torch.randint(0, 51, (8, 200))
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(bin_logits.view(-1, 51), bin_labels.view(-1))
```

---

### `ExpressionPredictionHead`

```python
from scmodelforge.models.components import ExpressionPredictionHead
```

Regression head for predicting continuous expression values. Used in masked autoencoders (MAE) to reconstruct expression levels at masked positions.

Architecture: `Linear → GELU → Linear(1)`

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | *required* | Input hidden dimension |

**Key Methods:**

#### `forward(hidden_states)`

Predict expression values from hidden states.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_states` | `torch.Tensor` | Shape `(B, S, hidden_dim)` |

**Returns:** `torch.Tensor` of shape `(B, S)` with predicted expression values

**Example:**

```python
from scmodelforge.models.components import ExpressionPredictionHead
import torch

head = ExpressionPredictionHead(hidden_dim=512)

hidden_states = torch.randn(16, 300, 512)
predicted_values = head(hidden_states)
print(predicted_values.shape)  # (16, 300)

# Compute MSE loss at masked positions
target_values = torch.randn(16, 300)
masked_positions = torch.rand(16, 300) < 0.15

loss = torch.nn.functional.mse_loss(
    predicted_values[masked_positions],
    target_values[masked_positions],
)
```

---

### `cls_pool()`

```python
from scmodelforge.models.components import cls_pool
```

Extract cell embeddings by returning the hidden state of the first (CLS) token. Commonly used in BERT-style models where the CLS token is prepended to the sequence and learns to aggregate sequence-level information.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_states` | `torch.Tensor` | Shape `(B, S, H)` |
| `attention_mask` | `torch.Tensor \| None` | Unused — accepted for API symmetry with `mean_pool()` |

**Returns:** `torch.Tensor` of shape `(B, H)` containing the CLS token embeddings

**Example:**

```python
from scmodelforge.models.components import cls_pool
import torch

hidden_states = torch.randn(8, 256, 512)
cell_embeddings = cls_pool(hidden_states)
print(cell_embeddings.shape)  # (8, 512)
```

---

### `mean_pool()`

```python
from scmodelforge.models.components import mean_pool
```

Extract cell embeddings by mean-pooling over non-padding tokens. This provides a more robust representation than CLS pooling when the model is not explicitly trained with a CLS token.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_states` | `torch.Tensor` | Shape `(B, S, H)` |
| `attention_mask` | `torch.Tensor \| None` | Shape `(B, S)` with 1 for real tokens and 0 for padding. If `None`, all positions are treated as real tokens |

**Returns:** `torch.Tensor` of shape `(B, H)` containing mean-pooled embeddings

**Example:**

```python
from scmodelforge.models.components import mean_pool
import torch

hidden_states = torch.randn(8, 256, 512)
attention_mask = torch.ones(8, 256)
attention_mask[:, 200:] = 0  # Last 56 positions are padding

cell_embeddings = mean_pool(hidden_states, attention_mask)
print(cell_embeddings.shape)  # (8, 512)

# Without mask (all positions included)
cell_embeddings = mean_pool(hidden_states)
print(cell_embeddings.shape)  # (8, 512)
```

---

### `generate_causal_mask()`

```python
from scmodelforge.models.components import generate_causal_mask
```

Generate a causal (autoregressive) attention mask for left-to-right modeling. Returns a lower-triangular mask where allowed positions are `0.0` and blocked (future) positions are `-inf`, compatible with `nn.TransformerEncoder`'s `mask` parameter.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `seq_len` | `int` | Sequence length |
| `device` | `torch.device \| None` | Target device for the mask tensor |

**Returns:** `torch.Tensor` of shape `(seq_len, seq_len)` with causal attention mask

**Example:**

```python
from scmodelforge.models.components import generate_causal_mask
import torch

# Generate causal mask for sequence length 5
mask = generate_causal_mask(5)
print(mask)
# tensor([[ 0., -inf, -inf, -inf, -inf],
#         [ 0.,  0., -inf, -inf, -inf],
#         [ 0.,  0.,  0., -inf, -inf],
#         [ 0.,  0.,  0.,  0., -inf],
#         [ 0.,  0.,  0.,  0.,  0.]])

# Use in autoregressive model
seq_len = 200
input_ids = torch.randint(0, 5000, (8, seq_len))
causal_mask = generate_causal_mask(seq_len, device=input_ids.device)

# Pass to nn.TransformerEncoder
# encoder(embeddings, mask=causal_mask, src_key_padding_mask=padding_mask)
```

---

## Registry

### `register_model()`

```python
from scmodelforge.models import register_model
```

Class decorator that registers a model class under a string name in the global model registry. This enables instantiation by name via `get_model()` and integration with the configuration system.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Registry key (e.g., `"transformer_encoder"`) |

**Returns:** Decorator function that registers the class and returns it unmodified

**Raises:** `ValueError` if `name` is already registered

**Example:**

```python
from scmodelforge.models import register_model
import torch.nn as nn

@register_model("my_custom_model")
class MyCustomModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    @classmethod
    def from_config(cls, config):
        return cls(vocab_size=config.vocab_size, hidden_dim=config.hidden_dim)

    def forward(self, input_ids, attention_mask, **kwargs):
        # ... model logic ...
        pass

# Now available via registry
from scmodelforge.models import get_model, list_models

print(list_models())
# ['autoregressive_transformer', 'masked_autoencoder', 'my_custom_model', 'transformer_encoder']
```

---

### `get_model()`

```python
from scmodelforge.models import get_model
```

Instantiate a registered model by name using a `ModelConfig` object. Calls the model class's `from_config()` class method.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Registry key (e.g., `"transformer_encoder"`) |
| `config` | `ModelConfig` | Model configuration object with `vocab_size` set |

**Returns:** `nn.Module` instance of the requested model

**Raises:** `ValueError` if `name` is not in the registry

**Example:**

```python
from scmodelforge.models import get_model
from scmodelforge.config.schema import ModelConfig

config = ModelConfig(
    architecture="transformer_encoder",
    vocab_size=5000,
    hidden_dim=512,
    num_layers=12,
    num_heads=8,
)

model = get_model(config.architecture, config)
print(type(model).__name__)  # TransformerEncoder

# Works with any registered model
config.architecture = "autoregressive_transformer"
model = get_model(config.architecture, config)
print(type(model).__name__)  # AutoregressiveTransformer
```

---

### `list_models()`

```python
from scmodelforge.models import list_models
```

Return a sorted list of all registered model names.

**Returns:** `list[str]` of model registry keys

**Example:**

```python
from scmodelforge.models import list_models

available = list_models()
print(available)
# ['autoregressive_transformer', 'masked_autoencoder', 'transformer_encoder']
```

---

## Utilities

### `init_weights()`

```python
from scmodelforge.models import init_weights
```

Initialize model weights using best practices for transformer models. Applied recursively via `model.apply(init_weights)`. Uses Xavier uniform initialization for linear layers, normal distribution for embeddings, and standard initialization for LayerNorm.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `nn.Module` | A single module (called recursively by `apply()`) |

**Initialization Strategy:**
- `nn.Linear`: Xavier uniform for weights, zeros for biases
- `nn.Embedding`: Normal distribution (mean=0, std=0.02), zeros for padding index
- `nn.LayerNorm`: Ones for weights, zeros for biases

**Example:**

```python
from scmodelforge.models import init_weights
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5000, 512)
        self.linear = nn.Linear(512, 512)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        return self.layer_norm(self.linear(self.embedding(x)))

model = MyModel()
model.apply(init_weights)

# Check initialization
print(model.embedding.weight.mean())  # ~0.0
print(model.embedding.weight.std())   # ~0.02
print(model.linear.bias.abs().max())  # 0.0
```

---

### `count_parameters()`

```python
from scmodelforge.models import count_parameters
```

Count the total number of parameters in a model, optionally filtering to only trainable parameters.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | The model to count parameters for |
| `trainable_only` | `bool` | `True` | If `True`, count only parameters that require gradients |

**Returns:** `int` — Total number of (trainable) parameters

**Example:**

```python
from scmodelforge.models import TransformerEncoder, count_parameters

model = TransformerEncoder(
    vocab_size=5000,
    hidden_dim=512,
    num_layers=12,
    num_heads=8,
)

total = count_parameters(model, trainable_only=False)
trainable = count_parameters(model, trainable_only=True)

print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")

# Freeze some layers
for param in model.embedding.parameters():
    param.requires_grad = False

trainable = count_parameters(model, trainable_only=True)
print(f"Trainable after freezing embedding: {trainable:,}")
```

---

## See Also

- **Configuration**: `scmodelforge.config` — Model configuration schema (`ModelConfig`)
- **Tokenizers**: `scmodelforge.tokenizers` — Convert cells to token sequences for model input
- **Training**: `scmodelforge.training` — Lightning module and training pipeline
- **Evaluation**: `scmodelforge.eval` — Embedding extraction and benchmark evaluation
- **Fine-tuning**: `scmodelforge.finetuning` — Task-specific fine-tuning with frozen or LoRA-adapted backbones
