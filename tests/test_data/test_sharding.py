"""Tests for shard conversion."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from scmodelforge.data.gene_vocab import GeneVocab
from scmodelforge.data.memmap_store import MemoryMappedStore
from scmodelforge.data.sharding import convert_to_shards, validate_shard_dir


@pytest.fixture()
def small_adata():
    """A small AnnData with 30 cells x 20 genes."""
    rng = np.random.default_rng(42)
    n_obs, n_vars = 30, 20
    X = sp.random(n_obs, n_vars, density=0.4, format="csr", random_state=42)
    obs = {"cell_type": rng.choice(["A", "B", "C"], n_obs)}
    var_names = [f"GENE_{i}" for i in range(n_vars)]
    adata = ad.AnnData(X=X)
    adata.var_names = var_names
    adata.obs = __import__("pandas").DataFrame(obs, index=[f"cell_{i}" for i in range(n_obs)])
    return adata


class TestConvertToShards:
    def test_single_h5ad(self, small_adata, tmp_path):
        vocab = GeneVocab.from_adata(small_adata)
        output = convert_to_shards(
            sources=[small_adata],
            gene_vocab=vocab,
            output_dir=tmp_path / "shards",
            shard_size=20,
        )
        assert output.exists()
        assert (output / "manifest.json").exists()

    def test_multiple_files(self, small_adata, tmp_path):
        vocab = GeneVocab.from_adata(small_adata)
        output = convert_to_shards(
            sources=[small_adata, small_adata],
            gene_vocab=vocab,
            output_dir=tmp_path / "shards",
            shard_size=40,
        )
        with open(output / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["total_cells"] == 60

    def test_manifest_correct(self, small_adata, tmp_path):
        vocab = GeneVocab.from_adata(small_adata)
        output = convert_to_shards(
            sources=[small_adata],
            gene_vocab=vocab,
            output_dir=tmp_path / "shards",
            shard_size=15,
        )
        with open(output / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["n_shards"] == 2
        assert manifest["total_cells"] == 30
        assert sum(manifest["shard_sizes"]) == 30
        assert "vocab_hash" in manifest

    def test_shard_files_exist(self, small_adata, tmp_path):
        vocab = GeneVocab.from_adata(small_adata)
        output = convert_to_shards(
            sources=[small_adata],
            gene_vocab=vocab,
            output_dir=tmp_path / "shards",
            shard_size=100,
        )
        shard_path = output / "shard_000"
        assert shard_path.is_dir()
        assert (shard_path / "X.npy").exists()
        assert (shard_path / "gene_indices.npy").exists()
        assert (shard_path / "n_genes.npy").exists()
        assert (shard_path / "obs.parquet").exists()
        assert (shard_path / "metadata.json").exists()

    def test_correct_shapes(self, small_adata, tmp_path):
        vocab = GeneVocab.from_adata(small_adata)
        output = convert_to_shards(
            sources=[small_adata],
            gene_vocab=vocab,
            output_dir=tmp_path / "shards",
            shard_size=100,
        )
        X = np.load(output / "shard_000" / "X.npy")
        G = np.load(output / "shard_000" / "gene_indices.npy")
        N = np.load(output / "shard_000" / "n_genes.npy")
        assert X.shape[0] == 30
        assert G.shape[0] == 30
        assert len(N) == 30
        assert X.shape == G.shape

    def test_roundtrip(self, small_adata, tmp_path):
        """Convert then read back and verify data."""
        vocab = GeneVocab.from_adata(small_adata)
        output = convert_to_shards(
            sources=[small_adata],
            gene_vocab=vocab,
            output_dir=tmp_path / "shards",
            shard_size=100,
        )
        store = MemoryMappedStore(output)
        assert len(store) == 30
        expr, gene_idx, _ = store.get_cell(0)
        assert len(expr) > 0
        assert len(expr) == len(gene_idx)

    def test_shard_size_controls_count(self, small_adata, tmp_path):
        vocab = GeneVocab.from_adata(small_adata)
        output = convert_to_shards(
            sources=[small_adata],
            gene_vocab=vocab,
            output_dir=tmp_path / "shards",
            shard_size=10,
        )
        with open(output / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["n_shards"] == 3
        assert manifest["shard_sizes"] == [10, 10, 10]

    def test_obs_metadata_preserved(self, small_adata, tmp_path):
        import pandas as pd

        vocab = GeneVocab.from_adata(small_adata)
        output = convert_to_shards(
            sources=[small_adata],
            gene_vocab=vocab,
            output_dir=tmp_path / "shards",
            shard_size=100,
            obs_keys=["cell_type"],
        )
        obs_df = pd.read_parquet(output / "shard_000" / "obs.parquet")
        assert "cell_type" in obs_df.columns
        assert len(obs_df) == 30

    def test_validate_shard_dir(self, small_adata, tmp_path):
        vocab = GeneVocab.from_adata(small_adata)
        output = convert_to_shards(
            sources=[small_adata],
            gene_vocab=vocab,
            output_dir=tmp_path / "shards",
            shard_size=100,
        )
        assert validate_shard_dir(output) is True

    def test_validate_missing_manifest(self, tmp_path):
        assert validate_shard_dir(tmp_path) is False
