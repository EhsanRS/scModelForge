"""Tests for GeneVocab."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scmodelforge._constants import CLS_TOKEN_ID, MASK_TOKEN_ID, NUM_SPECIAL_TOKENS, PAD_TOKEN_ID, UNK_TOKEN_ID
from scmodelforge.data.gene_vocab import GeneVocab

# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestConstruction:
    def test_from_genes(self):
        vocab = GeneVocab.from_genes(["TP53", "BRCA1", "EGFR"])
        assert len(vocab) == 3 + NUM_SPECIAL_TOKENS
        assert "TP53" in vocab
        assert "BRCA1" in vocab
        assert "UNKNOWN_GENE" not in vocab

    def test_from_genes_indices_start_after_special(self):
        vocab = GeneVocab.from_genes(["A", "B", "C"])
        assert vocab["A"] == NUM_SPECIAL_TOKENS
        assert vocab["B"] == NUM_SPECIAL_TOKENS + 1
        assert vocab["C"] == NUM_SPECIAL_TOKENS + 2

    def test_from_genes_deduplicates(self):
        """Duplicate genes must be removed — indices stay contiguous."""
        vocab = GeneVocab.from_genes(["A", "B", "A"])
        # Only 2 unique genes
        assert len(vocab) == 2 + NUM_SPECIAL_TOKENS
        # First occurrence order preserved, contiguous indices
        assert vocab["A"] == NUM_SPECIAL_TOKENS
        assert vocab["B"] == NUM_SPECIAL_TOKENS + 1
        # Max index must be < total vocab size (no out-of-range)
        max_idx = max(vocab[g] for g in ["A", "B"])
        assert max_idx < len(vocab)

    def test_from_genes_all_duplicates(self):
        """All-duplicate input collapses to a single gene."""
        vocab = GeneVocab.from_genes(["X", "X", "X"])
        assert len(vocab) == 1 + NUM_SPECIAL_TOKENS
        assert vocab["X"] == NUM_SPECIAL_TOKENS

    def test_from_genes_preserves_first_occurrence_order(self):
        """With duplicates, first-occurrence order is preserved."""
        vocab = GeneVocab.from_genes(["C", "A", "B", "A", "C"])
        assert vocab.genes == ["C", "A", "B"]
        assert vocab["C"] == NUM_SPECIAL_TOKENS
        assert vocab["A"] == NUM_SPECIAL_TOKENS + 1
        assert vocab["B"] == NUM_SPECIAL_TOKENS + 2

    def test_from_adata(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata)
        assert len(vocab) == 200 + NUM_SPECIAL_TOKENS
        assert "GENE_0" in vocab
        assert "GENE_199" in vocab

    def test_from_adata_custom_key(self, mini_adata):
        vocab = GeneVocab.from_adata(mini_adata, key="ensembl_id")
        assert "ENSG00000000000" in vocab
        assert "GENE_0" not in vocab

    def test_from_file_list(self, tmp_path):
        genes = ["TP53", "BRCA1", "EGFR"]
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(genes))
        vocab = GeneVocab.from_file(path)
        assert len(vocab) == 3 + NUM_SPECIAL_TOKENS
        assert "TP53" in vocab

    def test_from_file_dict(self, tmp_path):
        mapping = {"TP53": 4, "BRCA1": 5, "EGFR": 6}
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(mapping))
        vocab = GeneVocab.from_file(path)
        assert vocab["TP53"] == 4
        assert vocab["BRCA1"] == 5

    def test_invalid_indices_raises(self):
        with pytest.raises(ValueError, match="must start at"):
            GeneVocab({"TP53": 0})  # Collides with PAD_TOKEN_ID


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        original = GeneVocab.from_genes(["TP53", "BRCA1", "EGFR"])
        path = tmp_path / "vocab.json"
        original.save(path)

        loaded = GeneVocab.from_file(path)
        assert len(loaded) == len(original)
        assert loaded["TP53"] == original["TP53"]
        assert loaded["BRCA1"] == original["BRCA1"]

    def test_save_creates_parent_dirs(self, tmp_path):
        vocab = GeneVocab.from_genes(["A"])
        path = tmp_path / "nested" / "dir" / "vocab.json"
        vocab.save(path)
        assert path.exists()


# ------------------------------------------------------------------
# Encoding / Decoding
# ------------------------------------------------------------------


class TestEncoding:
    def test_encode_known_genes(self):
        vocab = GeneVocab.from_genes(["TP53", "BRCA1", "EGFR"])
        indices = vocab.encode(["TP53", "EGFR"])
        assert indices[0] == vocab["TP53"]
        assert indices[1] == vocab["EGFR"]
        assert indices.dtype == np.int64

    def test_encode_unknown_genes(self):
        vocab = GeneVocab.from_genes(["TP53", "BRCA1"])
        indices = vocab.encode(["TP53", "UNKNOWN"])
        assert indices[0] == vocab["TP53"]
        assert indices[1] == UNK_TOKEN_ID

    def test_encode_empty(self):
        vocab = GeneVocab.from_genes(["TP53"])
        indices = vocab.encode([])
        assert len(indices) == 0

    def test_decode_roundtrip(self):
        vocab = GeneVocab.from_genes(["TP53", "BRCA1", "EGFR"])
        genes = ["TP53", "BRCA1"]
        indices = vocab.encode(genes)
        decoded = vocab.decode(indices)
        assert decoded == genes

    def test_decode_special_tokens(self):
        vocab = GeneVocab.from_genes(["TP53"])
        decoded = vocab.decode([PAD_TOKEN_ID, UNK_TOKEN_ID, MASK_TOKEN_ID, CLS_TOKEN_ID])
        assert decoded == ["<pad>", "<unk>", "<mask>", "<cls>"]


# ------------------------------------------------------------------
# Gene alignment
# ------------------------------------------------------------------


class TestAlignment:
    def test_alignment_full_overlap(self):
        vocab = GeneVocab.from_genes(["A", "B", "C"])
        source_idx, vocab_idx = vocab.get_alignment_indices(["A", "B", "C"])
        assert list(source_idx) == [0, 1, 2]
        assert list(vocab_idx) == [vocab["A"], vocab["B"], vocab["C"]]

    def test_alignment_partial_overlap(self):
        vocab = GeneVocab.from_genes(["A", "B", "C"])
        source_idx, vocab_idx = vocab.get_alignment_indices(["X", "B", "Y", "C"])
        assert list(source_idx) == [1, 3]
        assert list(vocab_idx) == [vocab["B"], vocab["C"]]

    def test_alignment_no_overlap(self):
        vocab = GeneVocab.from_genes(["A", "B"])
        source_idx, vocab_idx = vocab.get_alignment_indices(["X", "Y", "Z"])
        assert len(source_idx) == 0
        assert len(vocab_idx) == 0


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------


class TestProperties:
    def test_special_token_ids(self):
        vocab = GeneVocab.from_genes(["A"])
        assert vocab.pad_token_id == PAD_TOKEN_ID
        assert vocab.unk_token_id == UNK_TOKEN_ID
        assert vocab.mask_token_id == MASK_TOKEN_ID
        assert vocab.cls_token_id == CLS_TOKEN_ID

    def test_genes_property(self):
        vocab = GeneVocab.from_genes(["C", "A", "B"])
        # Genes should be in vocabulary index order
        assert vocab.genes == ["C", "A", "B"]

    def test_repr(self):
        vocab = GeneVocab.from_genes(["A", "B"])
        r = repr(vocab)
        assert "n_genes=2" in r
        assert f"total_size={2 + NUM_SPECIAL_TOKENS}" in r
