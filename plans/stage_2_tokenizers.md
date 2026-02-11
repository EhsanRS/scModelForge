# Stage 2: scModelForge.tokenizers

## Overview

The tokenizer module is the most domain-specific component of scModelForge with no direct LLM analog. It converts a cell's gene expression vector into model-ready inputs. The key insight is that different single-cell foundation models tokenize cells in fundamentally different ways, and this module provides a standard interface to swap between strategies without changing anything else.

**Core responsibility:** Expression vector + gene list → `input_ids`, `attention_mask`, auxiliary tensors.

**Dependencies:** Stage 0 (scaffolding), Stage 1 (data — `GeneVocab`, `CellDataset` output format)
**Blocks:** Stage 3 (models consume tokenizer output), Stage 4 (training orchestrates tokenizer)

---

## Phase 1: Foundation (Months 1–3)

### Goals
- Define the `BaseTokenizer` interface that all strategies implement
- Implement `RankValueTokenizer` (Geneformer-style)
- Validate tokenization output against official Geneformer on a reference dataset
- Ensure tokenizer is decoupled from model — any tokenizer feeds any model

### Architecture

```
CellDataset output (expression, gene_indices)
     │
     ▼
┌────────────────────┐
│ BaseTokenizer      │  ← abstract interface
│  .tokenize(cell)   │     returns TokenizedCell
│  .decode(tokens)   │     optional inverse
└────────┬───────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
RankValue   (Phase 2: BinnedExpression, ContinuousProjection)
```

### File Structure

```
src/scmodelforge/tokenizers/
├── __init__.py              # Public API: BaseTokenizer, RankValueTokenizer, get_tokenizer
├── base.py                  # BaseTokenizer abstract class
├── rank_value.py            # RankValueTokenizer (Geneformer-style)
├── registry.py              # Tokenizer registry (string name → class)
└── _utils.py                # Shared utilities (ranking, binning, masking)
```

### Key Classes and Interfaces

#### `BaseTokenizer` (Abstract)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TokenizedCell:
    """Output of tokenization for a single cell."""
    input_ids: torch.Tensor          # (seq_len,) — token indices or gene indices
    attention_mask: torch.Tensor     # (seq_len,) — 1 for real tokens, 0 for padding
    values: torch.Tensor | None      # (seq_len,) — continuous values (optional)
    gene_indices: torch.Tensor       # (seq_len,) — original gene vocab indices
    metadata: dict[str, Any]         # Pass-through metadata

class BaseTokenizer(ABC):
    """Abstract base class for all tokenization strategies."""

    def __init__(self, gene_vocab: GeneVocab, max_len: int = 2048):
        self.gene_vocab = gene_vocab
        self.max_len = max_len

    @abstractmethod
    def tokenize(
        self,
        expression: np.ndarray | torch.Tensor,
        gene_indices: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        """Convert a single cell's expression into model input."""
        ...

    def tokenize_batch(
        self,
        expressions: list[np.ndarray],
        gene_indices_list: list[np.ndarray],
        metadata_list: list[dict] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of cells. Returns padded tensors."""
        cells = [
            self.tokenize(expr, genes, meta)
            for expr, genes, meta
            in zip(expressions, gene_indices_list, metadata_list or [{}] * len(expressions))
        ]
        return self._collate(cells)

    def _collate(self, cells: list[TokenizedCell]) -> dict[str, torch.Tensor]:
        """Pad and stack tokenized cells into a batch."""
        max_len = min(max(c.input_ids.shape[0] for c in cells), self.max_len)
        # Pad each tensor to max_len, stack into batch
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size (for embedding layer)."""
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name for this strategy."""
        ...
```

#### `RankValueTokenizer` (Geneformer-style)

In the Geneformer approach:
1. Filter to non-zero expressed genes
2. Rank genes by expression value (highest = rank 1)
3. Use gene identity (vocab index) as token, rank as position
4. Sequence is ordered from highest to lowest expression

```python
class RankValueTokenizer(BaseTokenizer):
    """Geneformer-style tokenization: rank genes by expression."""

    strategy_name = "rank_value"

    def __init__(
        self,
        gene_vocab: GeneVocab,
        max_len: int = 2048,
        prepend_cls: bool = True,
    ):
        super().__init__(gene_vocab, max_len)
        self.prepend_cls = prepend_cls

    @property
    def vocab_size(self) -> int:
        return len(self.gene_vocab)

    def tokenize(
        self,
        expression: np.ndarray,
        gene_indices: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedCell:
        """
        1. Filter to non-zero genes
        2. Rank by expression (descending)
        3. Take top max_len genes
        4. Return gene indices as input_ids (order = rank)
        """
        # Filter non-zero
        nonzero_mask = expression > 0
        expr_nz = expression[nonzero_mask]
        genes_nz = gene_indices[nonzero_mask]

        # Rank by expression (descending)
        rank_order = np.argsort(-expr_nz)
        ranked_genes = genes_nz[rank_order]
        ranked_expr = expr_nz[rank_order]

        # Truncate to max_len (leave room for CLS if prepending)
        effective_max = self.max_len - (1 if self.prepend_cls else 0)
        ranked_genes = ranked_genes[:effective_max]
        ranked_expr = ranked_expr[:effective_max]

        # Build input_ids
        input_ids = torch.tensor(ranked_genes, dtype=torch.long)
        if self.prepend_cls:
            cls_token = torch.tensor([self.gene_vocab.cls_token_id], dtype=torch.long)
            input_ids = torch.cat([cls_token, input_ids])

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        return TokenizedCell(
            input_ids=input_ids,
            attention_mask=attention_mask,
            values=torch.tensor(ranked_expr, dtype=torch.float32) if not self.prepend_cls
                   else torch.cat([torch.zeros(1), torch.tensor(ranked_expr, dtype=torch.float32)]),
            gene_indices=input_ids,  # In rank_value, input_ids ARE gene indices
            metadata=metadata or {},
        )
```

#### Tokenizer Registry

```python
# registry.py
_REGISTRY: dict[str, type[BaseTokenizer]] = {}

def register_tokenizer(name: str):
    """Decorator to register a tokenizer class."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_tokenizer(name: str, **kwargs) -> BaseTokenizer:
    """Instantiate a tokenizer by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown tokenizer: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)
```

### Masking Strategies

For pretraining, the tokenizer also needs to support masking. This is separate from tokenization but tightly coupled:

```python
@dataclass
class MaskedTokenizedCell(TokenizedCell):
    """TokenizedCell with masking applied for pretraining."""
    labels: torch.Tensor              # (seq_len,) — original values at masked positions, -100 elsewhere
    masked_positions: torch.Tensor    # (seq_len,) — boolean mask of which positions are masked

class MaskingStrategy:
    """Applies masking to tokenized cells for pretraining."""

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_token_id: int | None = None,  # If None, zero out instead
        random_replace_ratio: float = 0.1,  # BERT-style: 10% random, 10% keep
        keep_ratio: float = 0.1,
    ): ...

    def apply(self, cell: TokenizedCell) -> MaskedTokenizedCell:
        """Apply masking. Returns masked cell with labels."""
        ...
```

### Validation Against Official Geneformer

Create a validation script/test that:
1. Takes a reference `.h5ad` file with known Geneformer tokenization output
2. Runs `RankValueTokenizer` on the same data
3. Compares token sequences — they should be identical
4. This is a critical correctness test

```python
# tests/test_tokenizers/test_geneformer_validation.py
@pytest.mark.slow
def test_matches_official_geneformer():
    """Validate our RankValue tokenizer produces identical output to official Geneformer."""
    # Load reference data with known Geneformer tokenization
    # Run our tokenizer
    # Compare sequences
    ...
```

### Config Integration

```yaml
tokenizer:
  strategy: rank_value
  max_genes: 2048
  gene_vocab: human_protein_coding
  prepend_cls: true

  # Masking (for pretraining)
  masking:
    mask_ratio: 0.15
    random_replace_ratio: 0.1
    keep_ratio: 0.1
```

### Tests (Phase 1)

- `test_base_tokenizer.py`: Interface contract tests (any tokenizer must pass these).
- `test_rank_value.py`: Correctness of ranking, truncation, CLS prepending, edge cases (all-zero cell, single gene).
- `test_masking.py`: Mask ratio correctness, label generation, reproducibility with seed.
- `test_registry.py`: Registration, lookup, unknown name error.
- `test_geneformer_validation.py` (`@pytest.mark.slow`): Comparison with official Geneformer.

---

## Phase 2: Breadth (Months 4–6)

### BinnedExpressionTokenizer (scGPT-style)

scGPT discretises expression values into N bins and treats each bin as a token ID.

```python
class BinnedExpressionTokenizer(BaseTokenizer):
    """scGPT-style: bin expression values into discrete tokens."""

    strategy_name = "binned_expression"

    def __init__(
        self,
        gene_vocab: GeneVocab,
        n_bins: int = 51,
        max_len: int = 2048,
        binning_method: str = "uniform",  # "uniform", "quantile", "learned"
    ): ...
```

Key differences from RankValue:
- `input_ids` contains **two** parallel sequences: gene identity tokens AND expression bin tokens.
- Typically uses a fixed gene order (not ranked by expression).
- Sequence includes both zero and non-zero genes (padded to `max_len`).

### ContinuousProjectionTokenizer (TranscriptFormer-style)

Uses continuous expression values directly, projected through a learned linear layer.

```python
class ContinuousProjectionTokenizer(BaseTokenizer):
    """TranscriptFormer-style: continuous expression as model input."""

    strategy_name = "continuous_projection"

    def __init__(
        self,
        gene_vocab: GeneVocab,
        max_len: int = 2048,
        projection_dim: int | None = None,  # If None, raw values passed to model
        use_gene_embeddings: bool = False,   # Use pretrained gene embeddings
    ): ...
```

Key differences:
- `values` tensor is the primary input (not `input_ids`).
- Gene identity used for positional encoding or as attention bias.
- No discretization — preserves continuous nature of expression data.

### Additional Phase 2 Files

```
src/scmodelforge/tokenizers/
├── ...existing...
├── binned_expression.py     # BinnedExpressionTokenizer
├── continuous_projection.py # ContinuousProjectionTokenizer
└── transforms.py            # Shared transforms: normalization, binning functions
```

---

## Phase 3: Community & Scale (Months 7–12)

### GeneEmbeddingTokenizer

Uses pretrained gene embeddings (ESM-2, Gene2Vec, LLM-derived) as input representations.

```python
class GeneEmbeddingTokenizer(BaseTokenizer):
    """Use pretrained gene embeddings as input representation."""

    strategy_name = "gene_embedding"

    def __init__(
        self,
        gene_vocab: GeneVocab,
        embedding_source: str = "esm2",  # "esm2", "gene2vec", "custom"
        embedding_path: str | None = None,
        max_len: int = 2048,
    ): ...
```

### Plugin System

- Allow third-party tokenizers to register via entry points.
- `pyproject.toml` entry point: `[project.entry-points."scmodelforge.tokenizers"]`
- Community can create `pip install scmodelforge-my-tokenizer` packages.

### Tokenizer Benchmarking

- Utility to compare tokenizer strategies on the same dataset.
- Measures: throughput, vocabulary utilization, information retention, downstream task impact.

---

## Checklist

### Phase 1
- [ ] Define `TokenizedCell` dataclass
- [ ] Implement `BaseTokenizer` abstract class with `tokenize`, `tokenize_batch`, `_collate`
- [ ] Implement `RankValueTokenizer` with ranking, truncation, CLS prepend
- [ ] Implement `MaskingStrategy` with configurable ratios
- [ ] Implement tokenizer registry (`register_tokenizer`, `get_tokenizer`)
- [ ] Validate against official Geneformer tokenization output
- [ ] Add config parsing for `tokenizer:` section
- [ ] Write comprehensive tests
- [ ] Write docstrings and API documentation

### Phase 2
- [ ] Implement `BinnedExpressionTokenizer` (scGPT-style)
- [ ] Implement `ContinuousProjectionTokenizer` (TranscriptFormer-style)
- [ ] Add binning utilities (uniform, quantile, learned)
- [ ] Ensure all three tokenizers are interchangeable with all models

### Phase 3
- [ ] Implement `GeneEmbeddingTokenizer` (ESM-2, Gene2Vec)
- [ ] Build plugin system via entry points
- [ ] Create tokenizer benchmarking utility
