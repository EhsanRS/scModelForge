# Tokenizers Module (`scmodelforge.tokenizers`)

Tokenization strategies for single-cell gene expression data.

## Overview

The `scmodelforge.tokenizers` module converts preprocessed cell expression vectors into model-ready input sequences. It implements three distinct tokenization paradigms representing different approaches from the single-cell foundation model literature:

1. **Rank-Value** (Geneformer) — Rank genes by expression in descending order, using gene vocabulary indices as tokens
2. **Binned Expression** (scGPT) — Fixed gene order with discretized expression bins, using dual gene+bin representations
3. **Continuous Projection** (TranscriptFormer) — Fixed gene order with continuous expression values passed directly to the model

All tokenizers inherit from `BaseTokenizer`, which provides a unified interface for single-cell tokenization (`tokenize()`), batch collation (`tokenize_batch()`, `_collate()`), and masking integration. The `MaskingStrategy` class applies BERT-style masking (80/10/10) for pretraining, protecting special tokens like `[CLS]` and `[PAD]`.

The module also includes a registry system (`register_tokenizer()`, `get_tokenizer()`) for config-driven tokenizer instantiation and utility functions for ranking, binning, and tensor conversion.

Tokenization output is structured as `TokenizedCell` or `MaskedTokenizedCell` dataclasses, which contain `input_ids`, `attention_mask`, optional `values`, optional `bin_ids`, and metadata. Batching automatically pads sequences to the batch maximum length and stacks tensors for efficient GPU processing.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| `TokenizedCell` | Dataclass holding tokenized output for a single cell |
| `MaskedTokenizedCell` | Extends `TokenizedCell` with masking labels and positions |
| `BaseTokenizer` | Abstract base class for all tokenization strategies |
| `RankValueTokenizer` | Geneformer-style rank-value tokenization |
| `BinnedExpressionTokenizer` | scGPT-style discrete binning tokenization |
| `ContinuousProjectionTokenizer` | TranscriptFormer-style continuous values tokenization |
| `MaskingStrategy` | BERT-style masking for pretraining |
| `register_tokenizer()` | Decorator to register a tokenizer class by name |
| `get_tokenizer()` | Instantiate a registered tokenizer by name |
| `list_tokenizers()` | List all registered tokenizer names |
| `ensure_tensor()` | Convert numpy array to torch tensor |
| `rank_genes_by_expression()` | Rank genes by descending expression |
| `compute_bin_edges()` | Compute bin edges for expression discretization |
| `digitize_expression()` | Map continuous values to discrete bins |

---

## API Reference

### `TokenizedCell`

Dataclass holding the output of tokenization for a single cell. All tensor fields are 1-D with shape `(seq_len,)`. Batching (padding and stacking) is handled by `BaseTokenizer._collate()`.

```python
from scmodelforge.tokenizers import TokenizedCell
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `input_ids` | `torch.Tensor` | Token indices (1-D long tensor). For rank-value, these equal gene vocabulary indices; other strategies may differ. |
| `attention_mask` | `torch.Tensor` | Binary mask (1-D long tensor), 1 for real tokens, 0 for padding |
| `values` | `torch.Tensor` or `None` | Continuous expression values (1-D float tensor), optional |
| `bin_ids` | `torch.Tensor` or `None` | Discrete bin indices (1-D long tensor), only populated by `BinnedExpressionTokenizer` |
| `gene_indices` | `torch.Tensor` | Original gene vocabulary indices for each position (1-D long tensor) |
| `metadata` | `dict[str, Any]` | Pass-through metadata dictionary |

#### Example

```python
from scmodelforge.tokenizers import TokenizedCell
import torch

# Create a tokenized cell manually (normally created by tokenizers)
cell = TokenizedCell(
    input_ids=torch.tensor([3, 100, 250, 42], dtype=torch.long),  # [CLS] + 3 genes
    attention_mask=torch.ones(4, dtype=torch.long),
    values=torch.tensor([0.0, 5.2, 3.8, 1.1], dtype=torch.float32),
    gene_indices=torch.tensor([3, 100, 250, 42], dtype=torch.long),
    metadata={"cell_type": "T cell"}
)

print(cell.input_ids.shape)  # (4,)
print(cell.metadata)  # {'cell_type': 'T cell'}
```

---

### `MaskedTokenizedCell`

Extends `TokenizedCell` with label and position information needed for masked-gene-prediction loss during pretraining.

```python
from scmodelforge.tokenizers import MaskedTokenizedCell
```

#### Additional Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `labels` | `torch.Tensor` | Ground-truth `input_ids` at masked positions; `-100` elsewhere (PyTorch cross-entropy ignore index). 1-D long tensor. |
| `masked_positions` | `torch.Tensor` | Boolean tensor (1-D), `True` where masking was applied |

#### Example

```python
from scmodelforge.tokenizers import MaskedTokenizedCell
import torch

# Masked cell (normally created by MaskingStrategy.apply())
masked_cell = MaskedTokenizedCell(
    input_ids=torch.tensor([3, 2, 250, 42], dtype=torch.long),  # [CLS], [MASK], gene, gene
    attention_mask=torch.ones(4, dtype=torch.long),
    values=torch.tensor([0.0, 5.2, 3.8, 1.1], dtype=torch.float32),
    gene_indices=torch.tensor([3, 100, 250, 42], dtype=torch.long),
    labels=torch.tensor([-100, 100, -100, -100], dtype=torch.long),  # True label at position 1
    masked_positions=torch.tensor([False, True, False, False])
)

print(masked_cell.labels)  # tensor([-100,  100, -100, -100])
print(masked_cell.masked_positions.sum())  # 1 position masked
```

---

### `BaseTokenizer`

Abstract base class for all tokenization strategies. Defines the common interface for single-cell tokenization, batch processing, and collation.

```python
from scmodelforge.tokenizers import BaseTokenizer
```

#### Constructor

```python
BaseTokenizer(gene_vocab: GeneVocab, max_len: int = 2048)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene_vocab` | `GeneVocab` | *required* | Gene vocabulary mapping gene names to indices |
| `max_len` | `int` | `2048` | Maximum sequence length (including special tokens like CLS) |

#### Abstract Methods

Subclasses must implement:

##### `tokenize()`

```python
@abstractmethod
tokenize(
    expression: np.ndarray | torch.Tensor,
    gene_indices: np.ndarray | torch.Tensor,
    metadata: dict[str, Any] | None = None
) -> TokenizedCell
```

Convert a single cell's expression vector into model input.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expression` | `np.ndarray` or `torch.Tensor` | *required* | 1-D array of expression values |
| `gene_indices` | `np.ndarray` or `torch.Tensor` | *required* | 1-D array of gene vocabulary indices (same length as `expression`) |
| `metadata` | `dict` or `None` | `None` | Optional pass-through metadata |

**Returns:** `TokenizedCell`

##### `vocab_size` (property)

```python
@property
@abstractmethod
vocab_size() -> int
```

Total vocabulary size (for embedding layer construction).

##### `strategy_name` (property)

```python
@property
@abstractmethod
strategy_name() -> str
```

Human-readable name for this tokenization strategy.

#### Instance Methods

##### `tokenize_batch()`

```python
tokenize_batch(
    expressions: list[np.ndarray | torch.Tensor],
    gene_indices_list: list[np.ndarray | torch.Tensor],
    metadata_list: list[dict[str, Any]] | None = None
) -> dict[str, torch.Tensor]
```

Tokenize a batch of cells and collate into padded tensors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expressions` | `list` | *required* | List of 1-D expression vectors |
| `gene_indices_list` | `list` | *required* | List of 1-D gene index vectors (parallel to `expressions`) |
| `metadata_list` | `list` or `None` | `None` | Optional list of metadata dicts |

**Returns:** Dictionary with keys `input_ids`, `attention_mask`, `gene_indices`, and optionally `values`, `bin_ids`, `labels`, `masked_positions`.

##### `_collate()` (internal)

```python
_collate(cells: list[TokenizedCell]) -> dict[str, torch.Tensor]
```

Pad and stack a list of tokenized cells into a batch dict. Padding length is the batch maximum (not `max_len`) for memory efficiency. Sequences longer than `max_len` are truncated.

---

### `RankValueTokenizer`

Geneformer-style tokenization: rank genes by expression value in descending order, filter to non-zero genes, truncate to `max_len`, and optionally prepend a `[CLS]` token.

```python
from scmodelforge.tokenizers import RankValueTokenizer
```

#### Constructor

```python
RankValueTokenizer(
    gene_vocab: GeneVocab,
    max_len: int = 2048,
    prepend_cls: bool = True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene_vocab` | `GeneVocab` | *required* | Gene vocabulary |
| `max_len` | `int` | `2048` | Maximum sequence length (including CLS if enabled) |
| `prepend_cls` | `bool` | `True` | Whether to prepend a `[CLS]` token (ID=3) at the start |

#### Properties

- `vocab_size: int` — Returns `len(gene_vocab)` (includes special tokens)
- `strategy_name: str` — Returns `"rank_value"`

#### Methods

##### `tokenize()`

Implements the abstract method from `BaseTokenizer`.

**Steps:**
1. Filter to non-zero expressed genes
2. Rank genes by expression value (descending, stable sort)
3. Truncate to `max_len - 1` (reserving space for CLS if enabled)
4. Optionally prepend `[CLS]` token
5. Build `input_ids` (gene vocabulary indices), `attention_mask`, `values` (expression values), and `gene_indices`

**Returns:** `TokenizedCell` with `input_ids = gene_indices` and `values = expression`.

#### Example

```python
from scmodelforge.data import GeneVocab
from scmodelforge.tokenizers import RankValueTokenizer
import numpy as np

# Build vocabulary
vocab = GeneVocab.from_genes(["GeneA", "GeneB", "GeneC"])

# Create tokenizer
tokenizer = RankValueTokenizer(vocab, max_len=10, prepend_cls=True)

# Expression vector and gene indices
expression = np.array([0.0, 5.2, 3.1, 0.0, 8.5])
gene_indices = np.array([4, 5, 6, 7, 8])  # Vocab indices (0-3 are special tokens)

# Tokenize
cell = tokenizer.tokenize(expression, gene_indices)
print(cell.input_ids)  # tensor([3, 8, 5, 6]) — [CLS], highest, second, third
print(cell.values)  # tensor([0.0, 8.5, 5.2, 3.1])
print(cell.attention_mask)  # tensor([1, 1, 1, 1])
```

---

### `BinnedExpressionTokenizer`

scGPT-style tokenization: fixed gene order with discretized expression bins. Each gene position gets dual representations: gene vocabulary index and bin index.

```python
from scmodelforge.tokenizers import BinnedExpressionTokenizer
```

#### Constructor

```python
BinnedExpressionTokenizer(
    gene_vocab: GeneVocab,
    max_len: int = 2048,
    n_bins: int = 51,
    binning_method: str = "uniform",
    bin_edges: np.ndarray | None = None,
    value_max: float = 10.0,
    prepend_cls: bool = True,
    include_zero_genes: bool = True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene_vocab` | `GeneVocab` | *required* | Gene vocabulary |
| `max_len` | `int` | `2048` | Maximum sequence length (including CLS) |
| `n_bins` | `int` | `51` | Number of expression bins |
| `binning_method` | `str` | `"uniform"` | `"uniform"` for evenly spaced bins or `"quantile"` for data-driven quantile bins |
| `bin_edges` | `np.ndarray` or `None` | `None` | Pre-computed bin edges. If provided, `n_bins` and `binning_method` are ignored. |
| `value_max` | `float` | `10.0` | Upper bound for uniform binning (typically max log-expression) |
| `prepend_cls` | `bool` | `True` | Whether to prepend a `[CLS]` token |
| `include_zero_genes` | `bool` | `True` | Whether to include zero-expression genes in the sequence |

#### Methods

##### `fit()`

```python
fit(expression_values: np.ndarray) -> BinnedExpressionTokenizer
```

Compute bin edges from data (required for quantile binning before calling `tokenize()`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expression_values` | `np.ndarray` | *required* | 1-D array of expression values to compute quantiles from |

**Returns:** `self` (for method chaining)

##### `tokenize()`

Implements the abstract method from `BaseTokenizer`.

**Steps:**
1. Filter zero genes if `include_zero_genes=False`
2. Truncate to `max_len - 1` (if CLS enabled)
3. Compute bin IDs via `digitize_expression()`
4. Optionally prepend `[CLS]` token (bin 0)
5. Build `input_ids` (gene indices), `bin_ids`, `values`, `attention_mask`, `gene_indices`

**Raises:**
- `RuntimeError` — If `tokenize()` is called before `fit()` for quantile binning.

#### Properties

- `vocab_size: int` — Returns `len(gene_vocab)`
- `strategy_name: str` — Returns `"binned_expression"`
- `bin_edges: np.ndarray | None` — Currently stored bin edges (None if quantile fit not yet called)
- `n_bin_tokens: int` — Returns `n_bins` (for model embedding construction)

#### Example

```python
from scmodelforge.data import GeneVocab
from scmodelforge.tokenizers import BinnedExpressionTokenizer
import numpy as np

# Build vocabulary
vocab = GeneVocab.from_genes(["GeneA", "GeneB", "GeneC"])

# Uniform binning
tokenizer = BinnedExpressionTokenizer(
    vocab,
    n_bins=51,
    binning_method="uniform",
    value_max=10.0,
    prepend_cls=True,
    include_zero_genes=False
)

# Tokenize (uniform binning doesn't require fit())
expression = np.array([0.0, 5.2, 3.1, 0.0, 8.5])
gene_indices = np.array([4, 5, 6, 7, 8])

cell = tokenizer.tokenize(expression, gene_indices)
print(cell.input_ids)  # tensor([3, 5, 6, 8]) — [CLS] + non-zero genes
print(cell.bin_ids)  # tensor([0, 26, 15, 43]) — bin indices
print(cell.values)  # tensor([0.0, 5.2, 3.1, 8.5])

# Quantile binning (requires fit)
tokenizer_q = BinnedExpressionTokenizer(
    vocab,
    n_bins=51,
    binning_method="quantile"
)

# Fit on dataset
all_values = np.random.lognormal(0, 1, 10000)
tokenizer_q.fit(all_values)

# Now tokenize
cell_q = tokenizer_q.tokenize(expression, gene_indices)
```

---

### `ContinuousProjectionTokenizer`

TranscriptFormer-style tokenization: fixed gene order with continuous expression values passed directly to the model without discretization.

```python
from scmodelforge.tokenizers import ContinuousProjectionTokenizer
```

#### Constructor

```python
ContinuousProjectionTokenizer(
    gene_vocab: GeneVocab,
    max_len: int = 2048,
    prepend_cls: bool = True,
    include_zero_genes: bool = True,
    log_transform: bool = False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene_vocab` | `GeneVocab` | *required* | Gene vocabulary |
| `max_len` | `int` | `2048` | Maximum sequence length (including CLS) |
| `prepend_cls` | `bool` | `True` | Whether to prepend a `[CLS]` token |
| `include_zero_genes` | `bool` | `True` | Whether to include zero-expression genes |
| `log_transform` | `bool` | `False` | If `True`, apply `log1p` to expression values at tokenize time (normally done in preprocessing) |

#### Properties

- `vocab_size: int` — Returns `len(gene_vocab)`
- `strategy_name: str` — Returns `"continuous_projection"`

#### Methods

##### `tokenize()`

Implements the abstract method from `BaseTokenizer`.

**Steps:**
1. Filter zero genes if `include_zero_genes=False`
2. Optionally apply `log1p` transform
3. Truncate to `max_len - 1` (if CLS enabled)
4. Optionally prepend `[CLS]` token
5. Build `input_ids` (gene indices), `values` (expression), `attention_mask`, `gene_indices`

**Returns:** `TokenizedCell` with `input_ids = gene_indices` and `values = expression`.

#### Example

```python
from scmodelforge.data import GeneVocab
from scmodelforge.tokenizers import ContinuousProjectionTokenizer
import numpy as np

# Build vocabulary
vocab = GeneVocab.from_genes(["GeneA", "GeneB", "GeneC"])

# Create tokenizer
tokenizer = ContinuousProjectionTokenizer(
    vocab,
    max_len=10,
    prepend_cls=True,
    include_zero_genes=False,
    log_transform=False
)

# Expression and gene indices
expression = np.array([0.0, 5.2, 3.1, 0.0, 8.5])
gene_indices = np.array([4, 5, 6, 7, 8])

# Tokenize
cell = tokenizer.tokenize(expression, gene_indices)
print(cell.input_ids)  # tensor([3, 5, 6, 8]) — [CLS] + non-zero genes
print(cell.values)  # tensor([0.0, 5.2, 3.1, 8.5]) — continuous values
print(cell.attention_mask)  # tensor([1, 1, 1, 1])
```

---

### `MaskingStrategy`

Apply BERT-style masking to a `TokenizedCell` for pretraining. Of the selected positions:
- `mask_action_ratio` (default 80%) are replaced with `[MASK]` token
- `random_replace_ratio` (default 10%) are replaced with a random vocab token
- Remaining (default 10%) are kept unchanged

The strategy ensures at least one token is always masked (`max(1, round(n_maskable * mask_ratio))`), and protects `[CLS]` and `[PAD]` tokens from masking.

```python
from scmodelforge.tokenizers import MaskingStrategy
```

#### Constructor

```python
MaskingStrategy(
    mask_ratio: float = 0.15,
    mask_action_ratio: float = 0.8,
    random_replace_ratio: float = 0.1,
    vocab_size: int | None = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask_ratio` | `float` | `0.15` | Fraction of maskable tokens to select (must be in (0, 1]) |
| `mask_action_ratio` | `float` | `0.8` | Fraction of selected tokens replaced by `[MASK]` |
| `random_replace_ratio` | `float` | `0.1` | Fraction of selected tokens replaced by random token |
| `vocab_size` | `int` or `None` | `None` | Total vocabulary size (required if `random_replace_ratio > 0`) |

**Raises:**
- `ValueError` — If ratios are out of bounds, sum to >1, or `vocab_size` is missing when `random_replace_ratio > 0`.

#### Methods

##### `apply()`

```python
apply(cell: TokenizedCell, seed: int | None = None) -> MaskedTokenizedCell
```

Apply masking to a tokenized cell.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cell` | `TokenizedCell` | *required* | Input tokenized cell (not modified in-place) |
| `seed` | `int` or `None` | `None` | Optional per-call seed for reproducibility |

**Returns:** `MaskedTokenizedCell` with masking applied.

**Behavior:**
1. Identify maskable positions (not `[CLS]`, not `[PAD]`)
2. Randomly select `n_to_mask = max(1, round(n_maskable * mask_ratio))` positions
3. Store original `input_ids` as `labels` at selected positions (`-100` elsewhere)
4. Apply transformations:
   - First `mask_action_ratio` fraction → `[MASK]` token
   - Next `random_replace_ratio` fraction → random token from vocab
   - Remaining → unchanged
5. Return `MaskedTokenizedCell` with `labels` and `masked_positions`

#### Example

```python
from scmodelforge.data import GeneVocab
from scmodelforge.tokenizers import RankValueTokenizer, MaskingStrategy

# Build vocab and tokenizer
vocab = GeneVocab.from_genes(["GeneA", "GeneB", "GeneC"])
tokenizer = RankValueTokenizer(vocab, prepend_cls=True)

# Tokenize a cell
import numpy as np
expression = np.array([5.2, 3.1, 8.5])
gene_indices = np.array([4, 5, 6])
cell = tokenizer.tokenize(expression, gene_indices)

print(cell.input_ids)  # tensor([3, 6, 4, 5]) — [CLS], highest, second, third

# Create masking strategy
masker = MaskingStrategy(
    mask_ratio=0.15,
    mask_action_ratio=0.8,
    random_replace_ratio=0.1,
    vocab_size=len(vocab)
)

# Apply masking
masked_cell = masker.apply(cell, seed=42)
print(masked_cell.input_ids)  # e.g., tensor([3, 2, 4, 5]) — [CLS], [MASK], gene, gene
print(masked_cell.labels)  # e.g., tensor([-100, 6, -100, -100]) — true ID at masked pos
print(masked_cell.masked_positions)  # e.g., tensor([False, True, False, False])
```

---

### Registry Functions

The tokenizer registry enables config-driven instantiation of tokenizers by name.

#### `register_tokenizer()`

Class decorator that registers a tokenizer under a given name.

```python
from scmodelforge.tokenizers.registry import register_tokenizer

@register_tokenizer("my_custom_tokenizer")
class MyCustomTokenizer(BaseTokenizer):
    ...
```

**Raises:**
- `ValueError` — If the name is already registered.

#### `get_tokenizer()`

Instantiate a registered tokenizer by name.

```python
from scmodelforge.tokenizers import get_tokenizer
```

##### Function Signature

```python
get_tokenizer(name: str, **kwargs: Any) -> BaseTokenizer
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Registry key (e.g., `"rank_value"`, `"binned_expression"`, `"continuous_projection"`) |
| `**kwargs` | `Any` | - | Forwarded to the tokenizer constructor |

**Raises:**
- `ValueError` — If `name` is not in the registry.

#### `list_tokenizers()`

Return sorted list of registered tokenizer names.

```python
from scmodelforge.tokenizers import list_tokenizers
```

##### Function Signature

```python
list_tokenizers() -> list[str]
```

**Returns:** Sorted list of registered tokenizer names.

#### Example

```python
from scmodelforge.data import GeneVocab
from scmodelforge.tokenizers import get_tokenizer, list_tokenizers

# List available tokenizers
print(list_tokenizers())  # ['binned_expression', 'continuous_projection', 'rank_value']

# Build vocab
vocab = GeneVocab.from_genes(["GeneA", "GeneB", "GeneC"])

# Instantiate by name
tokenizer = get_tokenizer("rank_value", gene_vocab=vocab, max_len=512, prepend_cls=True)
print(tokenizer.strategy_name)  # "rank_value"

# Different strategy
tokenizer2 = get_tokenizer("binned_expression", gene_vocab=vocab, n_bins=51)
print(tokenizer2.strategy_name)  # "binned_expression"
```

---

### Utility Functions

#### `ensure_tensor()`

Convert a numpy array or tensor to a torch tensor with the given dtype.

```python
from scmodelforge.tokenizers._utils import ensure_tensor
```

##### Function Signature

```python
ensure_tensor(x: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `np.ndarray` or `torch.Tensor` | *required* | Input array or tensor |
| `dtype` | `torch.dtype` | *required* | Desired torch dtype (e.g., `torch.float32`, `torch.long`) |

**Returns:** `torch.Tensor` with the specified dtype.

---

#### `rank_genes_by_expression()`

Rank genes by expression value in descending order. Filters out zero-expression genes, then sorts remaining genes from highest to lowest expression using stable sort (tied values maintain original relative order).

```python
from scmodelforge.tokenizers._utils import rank_genes_by_expression
```

##### Function Signature

```python
rank_genes_by_expression(
    expression: torch.Tensor,
    gene_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expression` | `torch.Tensor` | *required* | 1-D float tensor of expression values |
| `gene_indices` | `torch.Tensor` | *required* | 1-D int tensor of gene vocabulary indices (same length as `expression`) |

**Returns:**
- `ranked_genes` — Gene indices sorted by descending expression
- `ranked_values` — Corresponding expression values in descending order

#### Example

```python
from scmodelforge.tokenizers._utils import rank_genes_by_expression
import torch

expression = torch.tensor([0.0, 5.2, 3.1, 0.0, 8.5])
gene_indices = torch.tensor([4, 5, 6, 7, 8])

ranked_genes, ranked_values = rank_genes_by_expression(expression, gene_indices)
print(ranked_genes)  # tensor([8, 5, 6]) — highest to lowest
print(ranked_values)  # tensor([8.5, 5.2, 3.1])
```

---

#### `compute_bin_edges()`

Compute bin edges for expression discretization.

```python
from scmodelforge.tokenizers._utils import compute_bin_edges
```

##### Function Signature

```python
compute_bin_edges(
    values: np.ndarray | None = None,
    n_bins: int = 51,
    method: str = "uniform",
    value_max: float = 10.0
) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | `np.ndarray` or `None` | `None` | 1-D array of expression values (required for `"quantile"` method) |
| `n_bins` | `int` | `51` | Number of bins |
| `method` | `str` | `"uniform"` | `"uniform"` for evenly spaced edges or `"quantile"` for data-driven quantile edges |
| `value_max` | `float` | `10.0` | Upper bound for uniform binning |

**Returns:** `np.ndarray` of bin edges with shape `(n_bins + 1,)`.

**Raises:**
- `ValueError` — If `method` is unknown or `values` is missing for quantile binning.

#### Example

```python
from scmodelforge.tokenizers._utils import compute_bin_edges
import numpy as np

# Uniform binning
edges_uniform = compute_bin_edges(n_bins=5, method="uniform", value_max=10.0)
print(edges_uniform)  # [0.  2.5 5.  7.5 10. ]

# Quantile binning
data = np.random.lognormal(0, 1, 1000)
edges_quantile = compute_bin_edges(values=data, n_bins=5, method="quantile")
print(edges_quantile)  # Data-driven quantiles
```

---

#### `digitize_expression()`

Map continuous expression values to discrete bin indices. Zero values are always mapped to bin 0. Non-zero values are assigned to bins `[1, n_bins - 1]` via `torch.bucketize()`.

```python
from scmodelforge.tokenizers._utils import digitize_expression
```

##### Function Signature

```python
digitize_expression(
    values: torch.Tensor,
    bin_edges: np.ndarray
) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | `torch.Tensor` | *required* | 1-D float tensor of expression values |
| `bin_edges` | `np.ndarray` | *required* | Bin edges from `compute_bin_edges()` |

**Returns:** `torch.Tensor` — 1-D long tensor of bin indices in `[0, n_bins - 1]`.

#### Example

```python
from scmodelforge.tokenizers._utils import compute_bin_edges, digitize_expression
import torch
import numpy as np

# Compute edges
edges = compute_bin_edges(n_bins=5, method="uniform", value_max=10.0)

# Digitize values
values = torch.tensor([0.0, 2.0, 5.5, 9.0, 0.0])
bin_ids = digitize_expression(values, edges)
print(bin_ids)  # tensor([0, 1, 2, 4, 0]) — bin assignments
```

---

## See Also

- **Data Module** — `scmodelforge.data` for preprocessing and dataset construction
- **Models** — `scmodelforge.models` for architectures that consume tokenized inputs
- **Training** — `scmodelforge.training` for integration with PyTorch Lightning
- **Configuration** — `scmodelforge.config.schema.TokenizerConfig` for YAML-based tokenizer configuration
