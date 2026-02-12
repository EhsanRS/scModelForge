"""Gene vocabulary management for scModelForge.

Provides a unified mapping between gene identifiers (names or Ensembl IDs)
and integer indices, with reserved special tokens at the start of the
vocabulary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scmodelforge._constants import (
    CLS_TOKEN_ID,
    MASK_TOKEN_ID,
    NUM_SPECIAL_TOKENS,
    PAD_TOKEN_ID,
    SPECIAL_TOKENS,
    UNK_TOKEN_ID,
)

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class GeneVocab:
    """Unified gene vocabulary across datasets.

    Maps gene names (or Ensembl IDs) to integer indices. The first
    ``NUM_SPECIAL_TOKENS`` indices are reserved for special tokens
    (<pad>, <unk>, <mask>, <cls>).

    Parameters
    ----------
    gene_to_idx
        Mapping from gene name to index. Indices must start at
        ``NUM_SPECIAL_TOKENS`` (special tokens are added automatically).
    """

    def __init__(self, gene_to_idx: dict[str, int]) -> None:
        # Validate that no gene index collides with special tokens
        min_idx = min(gene_to_idx.values()) if gene_to_idx else NUM_SPECIAL_TOKENS
        if min_idx < NUM_SPECIAL_TOKENS:
            raise ValueError(
                f"Gene indices must start at {NUM_SPECIAL_TOKENS} "
                f"(indices 0–{NUM_SPECIAL_TOKENS - 1} are reserved for special tokens). "
                f"Got minimum index {min_idx}."
            )

        self._gene_to_idx = dict(gene_to_idx)
        self._idx_to_gene: dict[int, str] = {v: k for k, v in self._gene_to_idx.items()}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_genes(cls, genes: list[str] | np.ndarray) -> GeneVocab:
        """Build a vocabulary from a list of gene names.

        Indices are assigned sequentially starting after special tokens.

        Parameters
        ----------
        genes
            Ordered list of gene names.
        """
        gene_to_idx = {g: i + NUM_SPECIAL_TOKENS for i, g in enumerate(genes)}
        return cls(gene_to_idx)

    @classmethod
    def from_adata(cls, adata: AnnData, key: str = "var_names") -> GeneVocab:
        """Build a vocabulary from an AnnData object.

        Parameters
        ----------
        adata
            AnnData object.
        key
            Which var attribute to use. ``"var_names"`` uses
            ``adata.var_names``; any other string uses
            ``adata.var[key]``.
        """
        genes = list(adata.var_names) if key == "var_names" else list(adata.var[key])
        return cls.from_genes(genes)

    @classmethod
    def from_file(cls, path: str | Path) -> GeneVocab:
        """Load a vocabulary from a JSON file.

        The JSON file should contain a mapping of gene name to index,
        or a list of gene names (indices will be assigned automatically).

        Parameters
        ----------
        path
            Path to a JSON file.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return cls.from_genes(data)
        if isinstance(data, dict):
            # Convert string keys to int values if needed
            gene_to_idx = {k: int(v) for k, v in data.items()}
            return cls(gene_to_idx)
        raise ValueError(f"Expected JSON list or dict, got {type(data)}")

    @classmethod
    def multi_species(
        cls,
        organisms: tuple[str, ...] | list[str] = ("human", "mouse"),
        include_one2many: bool = False,
        base_genes: list[str] | None = None,
    ) -> GeneVocab:
        """Build a vocabulary for multi-species analysis.

        Uses the bundled ortholog table to create a vocabulary in the
        canonical (human) namespace.  Mouse genes are mapped to their
        human orthologs automatically.

        Parameters
        ----------
        organisms
            Organisms to include.
        include_one2many
            Whether to include one-to-many orthologs.
        base_genes
            Optional seed list of human gene names. If ``None`` the
            full canonical gene set from the ortholog table is used.

        Returns
        -------
        GeneVocab
            Vocabulary containing canonical human gene names.
        """
        from scmodelforge.data.ortholog_mapper import OrthologMapper

        mapper = OrthologMapper(
            organisms=list(organisms),
            include_one2many=include_one2many,
        )
        genes = list(base_genes) if base_genes is not None else mapper.get_all_canonical_genes()
        return cls.from_genes(genes)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the vocabulary to a JSON file.

        Parameters
        ----------
        path
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._gene_to_idx, f, indent=2)
        logger.info("Saved gene vocabulary (%d genes) to %s", len(self), path)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, gene_names: list[str] | np.ndarray) -> np.ndarray:
        """Convert gene names to indices.

        Unknown genes are mapped to ``UNK_TOKEN_ID``.

        Parameters
        ----------
        gene_names
            List of gene names to encode.

        Returns
        -------
        np.ndarray
            Integer indices of shape ``(len(gene_names),)``.
        """
        return np.array(
            [self._gene_to_idx.get(g, UNK_TOKEN_ID) for g in gene_names],
            dtype=np.int64,
        )

    def decode(self, indices: list[int] | np.ndarray) -> list[str]:
        """Convert indices back to gene names.

        Special token indices are returned as their string names
        (e.g., ``"<pad>"``). Unknown indices return ``"<unk>"``.

        Parameters
        ----------
        indices
            Integer indices to decode.

        Returns
        -------
        list[str]
            Gene names.
        """
        special_idx_to_name = {v: k for k, v in SPECIAL_TOKENS.items()}
        result = []
        for idx in indices:
            idx = int(idx)
            if idx in special_idx_to_name:
                result.append(special_idx_to_name[idx])
            elif idx in self._idx_to_gene:
                result.append(self._idx_to_gene[idx])
            else:
                result.append("<unk>")
        return result

    def get_alignment_indices(self, gene_names: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get index mapping to align a dataset's genes to this vocabulary.

        Returns two arrays:
        - ``source_indices``: positions in the input gene list that exist
          in this vocabulary
        - ``vocab_indices``: corresponding indices in this vocabulary

        This is used to efficiently extract and reorder columns from an
        AnnData matrix to match the vocabulary.

        Parameters
        ----------
        gene_names
            Gene names from the source dataset.

        Returns
        -------
        source_indices
            Indices into ``gene_names`` for genes that are in the vocab.
        vocab_indices
            Corresponding vocabulary indices.
        """
        source_idx = []
        vocab_idx = []
        for i, gene in enumerate(gene_names):
            if gene in self._gene_to_idx:
                source_idx.append(i)
                vocab_idx.append(self._gene_to_idx[gene])
        return np.array(source_idx, dtype=np.int64), np.array(vocab_idx, dtype=np.int64)

    # ------------------------------------------------------------------
    # Properties and dunder methods
    # ------------------------------------------------------------------

    @property
    def pad_token_id(self) -> int:
        return PAD_TOKEN_ID

    @property
    def unk_token_id(self) -> int:
        return UNK_TOKEN_ID

    @property
    def mask_token_id(self) -> int:
        return MASK_TOKEN_ID

    @property
    def cls_token_id(self) -> int:
        return CLS_TOKEN_ID

    @property
    def genes(self) -> list[str]:
        """List of all gene names in vocabulary order."""
        return [self._idx_to_gene[i] for i in sorted(self._idx_to_gene.keys())]

    def __len__(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._gene_to_idx) + NUM_SPECIAL_TOKENS

    def __contains__(self, gene: str) -> bool:
        return gene in self._gene_to_idx

    def __getitem__(self, gene: str) -> int:
        return self._gene_to_idx[gene]

    def __repr__(self) -> str:
        return f"GeneVocab(n_genes={len(self._gene_to_idx)}, total_size={len(self)})"
