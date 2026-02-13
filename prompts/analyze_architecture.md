# Analyze Paper Architecture

Use this prompt when you find a new single-cell foundation model paper and want to understand how its architecture maps to scModelForge before implementing it.

## Step 1: Extract Core Architecture Details

Read the paper (or its methods section) and answer these questions:

### Model Architecture
- [ ] What is the base architecture? (Transformer encoder, decoder, encoder-decoder, other)
- [ ] How many layers/heads/hidden dimensions are used?
- [ ] What attention pattern? (Full bidirectional, causal/autoregressive, sparse, cross-attention)
- [ ] What positional encoding? (Learned, sinusoidal, relative, rotary/RoPE, none)
- [ ] What normalization? (Pre-norm, post-norm, RMSNorm)
- [ ] What activation function? (GELU, ReLU, SwiGLU, etc.)
- [ ] Is there a decoder or is it encoder-only?

### Gene/Cell Representation (Tokenization)
- [ ] How are genes represented? (Integer IDs, continuous embeddings, rank-ordered)
- [ ] How are expression values encoded? (Binned discrete, continuous projection, rank values, raw)
- [ ] Is there a CLS token? Other special tokens?
- [ ] What is the gene ordering strategy? (By expression rank, fixed order, random)
- [ ] Is there a maximum sequence length?
- [ ] Are there any novel embedding strategies? (Expression-aware positional encoding, etc.)

### Pretraining Objective
- [ ] What is the pretraining task? (Masked gene prediction, autoregressive, denoising, contrastive, multi-task)
- [ ] What is masked/predicted? (Gene identity, expression value, both)
- [ ] What is the loss function? (Cross-entropy, MSE, cosine similarity, combination)
- [ ] What masking strategy? (Random, structured, cell-type aware)
- [ ] What masking ratio?

### Special Components
- [ ] Any novel attention mechanisms? (Flash attention, linear attention, performer, etc.)
- [ ] Any novel prediction heads?
- [ ] Any novel embedding layers?
- [ ] Any auxiliary losses or regularization?

## Step 2: Map to scModelForge Components

For each architectural element, determine where it fits in scModelForge:

### Does it need a new tokenizer?

A new tokenizer is needed if the paper introduces a novel way to convert raw expression → token sequences. Map to existing tokenizers:

| Paper approach | Existing scModelForge tokenizer | Need new? |
|---|---|---|
| Rank genes by expression, use gene IDs | `RankValueTokenizer` (Geneformer-style) | No |
| Bin expression into discrete levels | `BinnedExpressionTokenizer` (scGPT-style) | No |
| Project continuous values into embeddings | `ContinuousProjectionTokenizer` | No |
| Something else entirely | — | **Yes** → `prompts/implement_tokenizer.md` |

### Does it need a new model?

A new model is needed if the paper introduces a novel architecture that cannot be represented by existing models. Map to existing models:

| Paper approach | Existing scModelForge model | Need new? |
|---|---|---|
| BERT-style masked encoder | `TransformerEncoder` | No |
| Autoregressive with causal mask, dual gene+expression heads | `AutoregressiveTransformer` | No |
| Asymmetric encoder-decoder with masking | `MaskedAutoencoder` | No |
| Novel architecture (e.g., Mamba, Perceiver, hybrid) | — | **Yes** → `prompts/implement_model.md` |

### Does it need new sub-components?

New components go in `src/scmodelforge/models/components/`. Consider:

| Paper element | Existing component | Need new? |
|---|---|---|
| Gene + position + expression embedding | `GeneExpressionEmbedding` | Maybe (if novel combination) |
| Predict masked gene identity | `MaskedGenePredictionHead` | No |
| Predict expression bins | `BinPredictionHead` | No |
| Predict continuous expression | `ExpressionPredictionHead` | No |
| Standard causal mask | `generate_causal_mask()` | No |
| CLS/mean pooling | `cls_pool()` / `mean_pool()` | No |
| Novel attention (RoPE, flash, etc.) | — | **Yes** → `prompts/implement_component.md` |
| Novel prediction head | — | **Yes** → `prompts/implement_component.md` |
| Novel embedding strategy | — | **Yes** → `prompts/implement_component.md` |

### Does it need new config fields?

If the architecture has parameters not covered by existing `ModelConfig` fields:

**Existing ModelConfig fields:**
- `architecture`, `hidden_dim`, `num_layers`, `num_heads`, `ffn_dim`
- `dropout`, `max_seq_len`, `pooling`, `activation`
- `use_expression_values`, `vocab_size`
- `n_bins`, `gene_loss_weight`, `expression_loss_weight`
- `decoder_dim`, `decoder_layers`, `decoder_heads`

If new parameters are needed, add them to `ModelConfig` in `config/schema.py` with sensible defaults so existing configs remain backward-compatible.

## Step 3: Create Implementation Plan

Based on the mapping above, create a plan listing:

1. **New files to create** — Implementation + test files
2. **Files to modify** — `__init__.py` exports, `config/schema.py`, docs
3. **Dependencies** — Any new Python packages needed (add as optional deps)
4. **Implementation order** — Components first, then the model/tokenizer that uses them
5. **Test strategy** — What fixtures and what to test

Then follow the appropriate implementation guide:
- `prompts/implement_model.md`
- `prompts/implement_tokenizer.md`
- `prompts/implement_benchmark.md`
- `prompts/implement_component.md`

## Step 4: Validation Questions

Before implementing, verify:

- [ ] Can any existing model/tokenizer be configured to replicate this paper's approach?
- [ ] Are the novel elements truly novel, or just hyperparameter changes?
- [ ] What is the minimum set of new code needed?
- [ ] Are all required dependencies already installed or need to be added?
- [ ] Does this architecture change how the training pipeline works, or just the model?

## Example: Analyzing a Hypothetical Paper

> "CellMamba: A State-Space Model for Single-Cell Transcriptomics"
> Uses Mamba blocks instead of attention, rank-value tokenization, masked gene prediction.

**Mapping:**
- Tokenizer: `RankValueTokenizer` — no new tokenizer needed
- Model: Need new model — Mamba blocks are not transformers
- Components: Need new `MambaBlock` component in `components/`
- Config: Need `state_dim`, `conv_dim`, `expand_factor` fields in `ModelConfig`
- Pretraining: Standard masked gene prediction — existing loss works
- Pipeline: No changes — Lightning module already handles `ModelOutput`

**Plan:**
1. Create `components/mamba.py` with `MambaBlock`
2. Create `models/cell_mamba.py` with `@register_model("cell_mamba")`
3. Add config fields to `ModelConfig`
4. Add imports to `__init__.py` files
5. Write tests
6. Add `mamba-ssm` as optional dependency
