# Tokenization Strategies for Single-Cell Foundation Models

This tutorial explains the four tokenization strategies available in scModelForge and how to choose the right one for your use case.

## What is Tokenization?

If you have worked with natural language processing (NLP), you know that transformers operate on sequences of discrete tokens (words, subwords, or characters). In single-cell genomics, we face a similar challenge: **how do we convert a cell's gene expression profile into a sequence that a transformer can process?**

Unlike text, where word order is meaningful, gene expression data presents different challenges:

- Each cell has expression values for thousands of genes
- Expression values are continuous (not discrete tokens)
- Some genes have zero expression (not detected)
- The relative importance of genes varies by expression level

Different tokenization strategies make different tradeoffs between:

- Information preservation (how much expression detail is retained)
- Computational efficiency (sequence length, memory usage)
- Biological interpretability (does the representation make biological sense)
- Model compatibility (which pretraining objectives work with this strategy)

scModelForge provides four tokenization strategies inspired by recent single-cell foundation models: Geneformer, scGPT, and TranscriptFormer.

## Overview of Tokenization Strategies

| Strategy | Inspired by | Input | Token representation | Expression info | Config value |
|---|---|---|---|---|---|
| Rank-value | Geneformer | Raw counts | Gene ranked by expression | Implicit (rank order) | `rank_value` |
| Binned expression | scGPT | Normalized counts | Gene + discretized bin | Explicit (bin ID) | `binned_expression` |
| Continuous projection | TranscriptFormer | Normalized counts | Gene + continuous value | Full precision | `continuous_projection` |
| Gene embedding | Various | Normalized counts | Pretrained gene vectors | Via embedding | `gene_embedding` |

## Rank-Value Tokenization (Geneformer-Style)

### How It Works

Rank-value tokenization is the simplest and most robust strategy:

1. Filter to non-zero expressed genes
2. Sort genes by expression value (highest to lowest)
3. Use the sorted gene IDs as tokens
4. Expression magnitude is encoded implicitly by position in the sequence

The key insight: **position in the sequence encodes expression level**. The first token is the most highly expressed gene, the second is the next highest, and so on.

### Strengths

- Simple and robust to normalization differences
- Proven effective in Geneformer (published in Nature 2023)
- No hyperparameters to tune (no binning decisions)
- Works well with raw counts (no normalization required)

### Limitations

- Loses absolute expression magnitude (only relative ordering)
- Cannot distinguish between genes with similar expression levels
- Cannot represent zero-expressed genes

### Usage Example

```python
from scmodelforge.tokenizers import get_tokenizer
from scmodelforge.data import GeneVocab
import anndata as ad

# Load your data
adata = ad.read_h5ad("my_data.h5ad")

# Build gene vocabulary
vocab = GeneVocab.from_adata(adata, min_cells=10)

# Create tokenizer
tokenizer = get_tokenizer(
    "rank_value",
    gene_vocab=vocab,
    max_len=2048,
    prepend_cls=True
)

# Tokenize a single cell
cell_idx = 0
expression = adata.X[cell_idx].toarray().flatten()  # Dense array
gene_indices = vocab.get_indices(adata.var_names.tolist())

tokenized = tokenizer.tokenize(expression, gene_indices)

print(f"Sequence length: {tokenized.input_ids.shape[0]}")
print(f"First 10 tokens: {tokenized.input_ids[:10]}")
print(f"Expression values: {tokenized.values[:10]}")
```

The resulting `TokenizedCell` has these fields:

- `input_ids`: Gene vocabulary indices (sorted by expression)
- `attention_mask`: All ones for real tokens
- `values`: Original expression values (in sorted order)
- `gene_indices`: Same as `input_ids` for this strategy

### Configuration

```yaml
tokenizer:
  strategy: rank_value
  max_len: 2048
  prepend_cls: true
```

## Binned Expression Tokenization (scGPT-Style)

### How It Works

Binned expression tokenization preserves expression magnitude by discretizing continuous values into bins:

1. Map continuous expression values to discrete bins (default: 51 bins)
2. Each token carries both gene identity and expression bin
3. Models can predict both which gene is expressed and how much

This is analogous to how images are discretized into pixel values (0-255) for processing.

### Strengths

- Preserves expression magnitude information
- Model can predict expression level (useful for generative tasks)
- Explicit representation of expression (not just rank)
- Works with autoregressive models (next-token prediction)

### Limitations

- Binning introduces quantization error
- Requires choosing number of bins (tradeoff between precision and vocabulary size)
- More complex than rank-value

### Usage Example

```python
from scmodelforge.tokenizers import get_tokenizer

# Create tokenizer with binning
tokenizer = get_tokenizer(
    "binned_expression",
    gene_vocab=vocab,
    max_len=2048,
    n_bins=51,
    binning_method="uniform",
    prepend_cls=True
)

# For uniform binning, no fit() needed
tokenized = tokenizer.tokenize(expression, gene_indices)

# Check the bin assignments
print(f"Bin IDs: {tokenized.bin_ids[:10]}")
print(f"Gene IDs: {tokenized.input_ids[:10]}")
print(f"Original values: {tokenized.values[:10]}")
```

The `TokenizedCell` now includes `bin_ids`, which stores the discretized expression level for each gene.

### Binning Methods

scModelForge supports two binning methods:

1. **Uniform binning** (default): Divides the expression range into equal-width bins
   - Fast, no data fitting required
   - Good for normalized data with known range

2. **Quantile binning**: Divides data into equal-frequency bins
   - Requires `fit()` on representative data
   - Better for skewed distributions

```python
# Quantile binning example
tokenizer = get_tokenizer(
    "binned_expression",
    gene_vocab=vocab,
    n_bins=51,
    binning_method="quantile"
)

# Fit on training data
all_expression_values = adata.X.toarray().flatten()
tokenizer.fit(all_expression_values)

# Now tokenize
tokenized = tokenizer.tokenize(expression, gene_indices)
```

### Configuration

```yaml
tokenizer:
  strategy: binned_expression
  max_len: 2048
  n_bins: 51
  binning_method: uniform
  prepend_cls: true
```

## Continuous Projection Tokenization (TranscriptFormer-Style)

### How It Works

Continuous projection is the most information-preserving strategy:

1. Keep gene identity and expression value as continuous scalars
2. No discretization or ranking
3. Models project the continuous values directly into the hidden space

This is the "lossless" tokenization strategy.

### Strengths

- No information loss from discretization or ranking
- Full precision expression values
- Simplest conceptually (just package the data)

### Limitations

- Harder to use with classification-based pretraining objectives
- Models must handle continuous inputs (not all architectures do)
- May require careful normalization

### Usage Example

```python
from scmodelforge.tokenizers import get_tokenizer

tokenizer = get_tokenizer(
    "continuous_projection",
    gene_vocab=vocab,
    max_len=2048,
    prepend_cls=True,
    log_transform=True  # Apply log1p at tokenization time
)

tokenized = tokenizer.tokenize(expression, gene_indices)

# Values are continuous (possibly log-transformed)
print(f"Continuous values: {tokenized.values[:10]}")
```

### Log Transformation

The `log_transform` parameter applies `log1p` (log(x + 1)) at tokenization time. This is useful if your data is not pre-normalized:

```python
# Without log transform (assume data is already log-normalized)
tokenizer = get_tokenizer(
    "continuous_projection",
    gene_vocab=vocab,
    log_transform=False
)

# With log transform (apply to raw counts)
tokenizer = get_tokenizer(
    "continuous_projection",
    gene_vocab=vocab,
    log_transform=True
)
```

### Configuration

```yaml
tokenizer:
  strategy: continuous_projection
  max_len: 2048
  prepend_cls: true
  log_transform: false
```

## Gene Embedding Tokenization

### How It Works

Gene embedding tokenization uses pretrained gene representations (e.g., from Gene2Vec, protein language models, or biological knowledge graphs):

1. Load a pretrained embedding matrix (one vector per gene)
2. Tokenize similar to continuous projection
3. Models can leverage biological prior knowledge

This is analogous to using pretrained word embeddings (Word2Vec, GloVe) in NLP.

### Strengths

- Incorporates biological prior knowledge
- Can improve performance with limited training data
- Leverages gene function, pathway, or sequence similarity

### Limitations

- Requires pretrained embeddings file
- Quality depends on the source embeddings
- Embedding dimensions must match model architecture

### Usage Example

```python
from scmodelforge.tokenizers import get_tokenizer

tokenizer = get_tokenizer(
    "gene_embedding",
    gene_vocab=vocab,
    max_len=2048,
    embedding_path="./gene2vec_embeddings.npy",
    embedding_dim=200
)

# Tokenize normally
tokenized = tokenizer.tokenize(expression, gene_indices)

# Access the pretrained embeddings
gene_emb = tokenizer.gene_embeddings  # Shape: (vocab_size, 200)
print(f"Gene embedding shape: {gene_emb.shape}")
```

### Supported Embedding Formats

scModelForge can load embeddings from:

- `.pt` or `.pth` (PyTorch tensors)
- `.npy` (NumPy arrays)

The embedding file should contain a matrix of shape `(n_genes, embedding_dim)` with rows aligned to the gene vocabulary.

### Creating Gene Embeddings

If you do not have pretrained embeddings, you can create them from:

1. **Gene2Vec**: Train on gene co-expression networks
2. **Protein language models**: Use embeddings from ESM, ProtBERT, etc.
3. **Knowledge graphs**: Encode Gene Ontology or pathway relationships
4. **Gene sequences**: Use k-mer or sequence-based embeddings

### Configuration

```yaml
tokenizer:
  strategy: gene_embedding
  max_len: 2048
  embedding_path: ./pretrained/gene2vec.npy
  embedding_dim: 200
  prepend_cls: true
```

## Masking for Pretraining

Regardless of which tokenization strategy you choose, you will likely want to apply masking for self-supervised pretraining. scModelForge implements BERT-style masking:

- Select 15% of tokens at random (configurable)
- Of the selected tokens:
  - 80% are replaced with `[MASK]`
  - 10% are replaced with a random token
  - 10% are kept unchanged

This forces the model to learn bidirectional representations and prevents overfitting to the `[MASK]` token.

### Usage Example

```python
from scmodelforge.tokenizers import get_tokenizer
from scmodelforge.tokenizers.masking import MaskingStrategy

# Create tokenizer
tokenizer = get_tokenizer("rank_value", gene_vocab=vocab)

# Create masking strategy
masker = MaskingStrategy(
    mask_ratio=0.15,
    mask_action_ratio=0.8,
    random_replace_ratio=0.1,
    vocab_size=len(vocab)
)

# Tokenize and mask
tokenized = tokenizer.tokenize(expression, gene_indices)
masked = masker.apply(tokenized)

# Inspect the results
print(f"Original tokens: {tokenized.input_ids[:20]}")
print(f"Masked tokens: {masked.input_ids[:20]}")
print(f"Labels (for loss): {masked.labels[:20]}")
print(f"Masked positions: {masked.masked_positions[:20]}")
```

The `MaskedTokenizedCell` adds two fields:

- `labels`: Ground-truth token IDs at masked positions, `-100` elsewhere (PyTorch cross-entropy ignore index)
- `masked_positions`: Boolean tensor indicating which positions were masked

### Configuration

```yaml
masking:
  mask_ratio: 0.15
  mask_action_ratio: 0.8
  random_replace_ratio: 0.1
```

### Special Token Protection

The masking strategy automatically protects special tokens:

- `[CLS]` tokens are never masked
- `[PAD]` tokens are never masked

This ensures the model can rely on these positional and structural tokens.

## Which Strategy Should I Use?

Here is a decision flowchart to help you choose:

### Starting Out?

Use **rank-value tokenization**:

- Proven track record (Geneformer)
- Simplest to implement and debug
- No hyperparameters to tune
- Robust to normalization differences

### Need Expression Magnitude?

Use **binned expression tokenization** if:

- You want the model to predict expression levels
- You are building a generative model (autoregressive)
- You need explicit representation of expression

### Want Maximum Information?

Use **continuous projection tokenization** if:

- You have a model architecture that handles continuous inputs
- You want zero information loss
- You are willing to handle continuous-valued loss functions

### Have Pretrained Gene Embeddings?

Use **gene embedding tokenization** if:

- You have high-quality pretrained gene representations
- You have limited training data (transfer learning)
- You want to incorporate biological prior knowledge

### Quick Reference Table

| Use case | Recommended strategy |
|---|---|
| General pretraining | `rank_value` |
| Cell type classification | `rank_value` or `binned_expression` |
| Perturbation prediction | `binned_expression` or `continuous_projection` |
| Gene expression imputation | `continuous_projection` |
| Low-data regime | `gene_embedding` |
| Autoregressive generation | `binned_expression` |

## Batch Tokenization

In practice, you will tokenize entire datasets, not individual cells. scModelForge handles this automatically in the training pipeline via `TokenizedCellDataset`:

```python
from scmodelforge.data import CellDataset, GeneVocab
from scmodelforge.training.data_module import TokenizedCellDataset
from scmodelforge.tokenizers import get_tokenizer
from scmodelforge.tokenizers.masking import MaskingStrategy

# Prepare dataset
vocab = GeneVocab.from_adata(adata)
cell_dataset = CellDataset.from_adata(adata, vocab)

# Create tokenizer and masking strategy
tokenizer = get_tokenizer("rank_value", gene_vocab=vocab)
masker = MaskingStrategy(mask_ratio=0.15, vocab_size=len(vocab))

# Wrap with tokenization
tokenized_dataset = TokenizedCellDataset(
    cell_dataset,
    tokenizer,
    masking_strategy=masker
)

# Now you can iterate
for item in tokenized_dataset:
    print(item["input_ids"].shape)
    print(item["labels"].shape)
    break
```

The dataset automatically tokenizes and masks each cell on-the-fly during training.

## Advanced: Custom Tokenizers

If the built-in strategies do not fit your needs, you can implement a custom tokenizer by subclassing `BaseTokenizer`:

```python
from scmodelforge.tokenizers.base import BaseTokenizer, TokenizedCell
from scmodelforge.tokenizers.registry import register_tokenizer
import torch

@register_tokenizer("my_custom_strategy")
class MyCustomTokenizer(BaseTokenizer):
    @property
    def vocab_size(self) -> int:
        return len(self.gene_vocab)

    @property
    def strategy_name(self) -> str:
        return "my_custom_strategy"

    def tokenize(self, expression, gene_indices, metadata=None):
        # Your custom tokenization logic here
        # Must return a TokenizedCell
        return TokenizedCell(
            input_ids=...,
            attention_mask=...,
            values=...,
            gene_indices=...,
            metadata=metadata or {}
        )
```

Then use it like any built-in strategy:

```python
tokenizer = get_tokenizer("my_custom_strategy", gene_vocab=vocab)
```

## What's Next?

Now that you understand tokenization strategies, you can:

- Explore the [Pretraining Tutorial](pretraining.md) to learn how to train foundation models
- Read the [Building Custom Models](custom_model.md) to understand how models consume tokenized inputs
- Check the [Fine-tuning Tutorial](finetuning_cell_type.md) for downstream task adaptation

## Summary

Tokenization is the bridge between biological data and transformer models. scModelForge provides four flexible strategies:

1. **Rank-value**: Simple, robust, proven (Geneformer)
2. **Binned expression**: Explicit expression levels (scGPT)
3. **Continuous projection**: Maximum information preservation (TranscriptFormer)
4. **Gene embedding**: Incorporate biological priors

Choose based on your use case, data characteristics, and model architecture. When in doubt, start with rank-value tokenization.
