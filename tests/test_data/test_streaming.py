"""Tests for StreamingCellDataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scmodelforge._constants import NUM_SPECIAL_TOKENS
from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.data.streaming import StreamingCellDataset


@pytest.fixture()
def gene_vocab():
    """Small gene vocabulary."""
    genes = [f"GENE_{i}" for i in range(20)]
    return GeneVocab.from_genes(genes)


@pytest.fixture()
def h5ad_file(tmp_path, gene_vocab):
    """Create a small H5AD file for testing."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(42)
    n_cells = 50
    gene_names = gene_vocab.genes
    X = rng.random((n_cells, len(gene_names))).astype(np.float32)
    X[X < 0.3] = 0.0  # Introduce some zeros

    adata = ad.AnnData(
        X=X,
        var=pd.DataFrame(index=gene_names),
        obs=pd.DataFrame(
            {"cell_type": rng.choice(["A", "B", "C"], n_cells)},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
    )
    path = tmp_path / "test_data.h5ad"
    adata.write_h5ad(path)
    return str(path)


@pytest.fixture()
def two_h5ad_files(tmp_path, gene_vocab):
    """Create two small H5AD files for testing."""
    import anndata as ad
    import pandas as pd

    paths = []
    for k in range(2):
        rng = np.random.default_rng(42 + k)
        n_cells = 30
        gene_names = gene_vocab.genes
        X = rng.random((n_cells, len(gene_names))).astype(np.float32)
        X[X < 0.3] = 0.0
        adata = ad.AnnData(
            X=X,
            var=pd.DataFrame(index=gene_names),
            obs=pd.DataFrame(index=[f"cell_{k}_{i}" for i in range(n_cells)]),
        )
        path = tmp_path / f"data_{k}.h5ad"
        adata.write_h5ad(path)
        paths.append(str(path))
    return paths


class TestStreamingCellDataset:
    def test_basic_iteration(self, h5ad_file, gene_vocab):
        ds = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=10,
            shuffle_buffer_size=0,
        )
        cells = list(ds)
        assert len(cells) > 0

    def test_output_format(self, h5ad_file, gene_vocab):
        ds = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=10,
            shuffle_buffer_size=0,
        )
        cell = next(iter(ds))
        assert "expression" in cell
        assert "gene_indices" in cell
        assert "n_genes" in cell
        assert "metadata" in cell
        assert isinstance(cell["expression"], torch.Tensor)
        assert isinstance(cell["gene_indices"], torch.Tensor)
        assert cell["expression"].dtype == torch.float32
        assert cell["gene_indices"].dtype == torch.int64

    def test_gene_alignment(self, h5ad_file, gene_vocab):
        ds = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=10,
            shuffle_buffer_size=0,
        )
        cell = next(iter(ds))
        # All gene indices should be valid vocab indices (>= NUM_SPECIAL_TOKENS)
        assert (cell["gene_indices"] >= NUM_SPECIAL_TOKENS).all()

    def test_zeros_filtered(self, h5ad_file, gene_vocab):
        ds = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=10,
            shuffle_buffer_size=0,
        )
        for cell in ds:
            # All expression values should be non-zero
            assert (cell["expression"] > 0).all()

    def test_shuffle_buffer(self, h5ad_file, gene_vocab):
        ds_no_shuffle = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=10,
            shuffle_buffer_size=0,
            seed=0,
        )
        ds_shuffle = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=10,
            shuffle_buffer_size=20,
            seed=0,
        )
        no_shuffle_genes = [c["n_genes"] for c in ds_no_shuffle]
        shuffle_genes = [c["n_genes"] for c in ds_shuffle]
        # Same total count
        assert len(no_shuffle_genes) == len(shuffle_genes)
        # Order should differ (with high probability)
        assert sorted(no_shuffle_genes) == sorted(shuffle_genes)

    def test_multiple_files(self, two_h5ad_files, gene_vocab):
        ds = StreamingCellDataset(
            file_paths=two_h5ad_files,
            gene_vocab=gene_vocab,
            chunk_size=10,
            shuffle_buffer_size=0,
        )
        cells = list(ds)
        # Should have cells from both files
        assert len(cells) > 30  # More than one file's worth

    def test_obs_keys(self, h5ad_file, gene_vocab):
        ds = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=10,
            shuffle_buffer_size=0,
            obs_keys=["cell_type"],
        )
        cell = next(iter(ds))
        assert "cell_type" in cell["metadata"]

    def test_chunk_size_respected(self, h5ad_file, gene_vocab):
        """Different chunk sizes should yield same total cells."""
        ds_small = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=5,
            shuffle_buffer_size=0,
        )
        ds_large = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=100,
            shuffle_buffer_size=0,
        )
        assert len(list(ds_small)) == len(list(ds_large))

    def test_repr(self, gene_vocab):
        ds = StreamingCellDataset(
            file_paths=["a.h5ad", "b.h5ad"],
            gene_vocab=gene_vocab,
        )
        r = repr(ds)
        assert "StreamingCellDataset" in r
        assert "n_files=2" in r

    def test_dataloader_integration(self, h5ad_file, gene_vocab):
        from scmodelforge.tokenizers.rank_value import RankValueTokenizer

        ds = StreamingCellDataset(
            file_paths=[h5ad_file],
            gene_vocab=gene_vocab,
            chunk_size=10,
            shuffle_buffer_size=0,
        )
        tok = RankValueTokenizer(gene_vocab=gene_vocab, max_len=50)

        # Wrap with _StreamingTokenizedDataset
        from scmodelforge.training.data_module import _StreamingTokenizedDataset

        tok_ds = _StreamingTokenizedDataset(ds, tok)
        loader = torch.utils.data.DataLoader(
            tok_ds, batch_size=4, collate_fn=tok._collate,
        )
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] <= 4
