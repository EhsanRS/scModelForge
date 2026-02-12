"""Shared fixtures for evaluation module tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scmodelforge.config.schema import EvalConfig
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.models.transformer_encoder import TransformerEncoder
from scmodelforge.tokenizers.rank_value import RankValueTokenizer


@pytest.fixture()
def tiny_adata():
    """40 cells x 30 genes, sparse, 2 cell types, 2 batches."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(42)
    n_cells, n_genes = 40, 30

    # Create expression data — make cell types linearly separable
    X = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float32)
    # Give first 20 cells higher expression in first 15 genes
    X[:20, :15] += 5.0
    # Give last 20 cells higher expression in last 15 genes
    X[20:, 15:] += 5.0

    X_sparse = sp.csr_matrix(X)

    gene_names = [f"gene_{i}" for i in range(n_genes)]
    cell_types = ["type_A"] * 20 + ["type_B"] * 20
    batches = (["batch_0"] * 10 + ["batch_1"] * 10) * 2

    obs = pd.DataFrame(
        {"cell_type": cell_types, "batch": batches},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)

    return ad.AnnData(X=X_sparse, obs=obs, var=var)


@pytest.fixture()
def tiny_vocab(tiny_adata):
    """GeneVocab built from tiny_adata."""
    return GeneVocab.from_adata(tiny_adata)


@pytest.fixture()
def tiny_model(tiny_vocab):
    """Tiny TransformerEncoder for fast tests."""
    return TransformerEncoder(
        vocab_size=len(tiny_vocab),
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        dropout=0.0,
        max_seq_len=64,
        pooling="cls",
        activation="gelu",
        use_expression_values=True,
    )


@pytest.fixture()
def tiny_tokenizer(tiny_vocab):
    """RankValueTokenizer for testing."""
    return RankValueTokenizer(gene_vocab=tiny_vocab, max_len=64, prepend_cls=True)


@pytest.fixture()
def synthetic_embeddings():
    """40x32 numpy array, linearly separable by cell type."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((40, 32)).astype(np.float32)
    # Make first 20 cells separable from last 20
    emb[:20, 0] += 5.0
    emb[20:, 0] -= 5.0
    return emb


@pytest.fixture()
def tiny_benchmark_config():
    """EvalConfig with linear_probe benchmark."""
    return EvalConfig(every_n_epochs=1, batch_size=16, benchmarks=["linear_probe"])
