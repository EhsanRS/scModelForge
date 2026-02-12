"""Multi-species gene name mapping via ortholog tables.

Maps gene names from non-canonical organisms (e.g. mouse) to a canonical
namespace (human) using bundled Ensembl ortholog data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from scmodelforge.data.orthologs import ORTHOLOG_TABLE_PATH

if TYPE_CHECKING:
    from scmodelforge.config.schema import MultiSpeciesConfig

logger = logging.getLogger(__name__)


class OrthologMapper:
    """Maps gene names between species using ortholog tables.

    Human gene names are the canonical namespace. Mouse genes are
    mapped to their human orthologs. Only one-to-one orthologs are
    used by default (set ``include_one2many=True`` to include
    one-to-many mappings).

    Parameters
    ----------
    organisms
        List of organisms to include (e.g. ``["human", "mouse"]``).
    canonical_organism
        The canonical namespace. Genes from other organisms are mapped
        to this organism's gene names.
    include_one2many
        Whether to include one-to-many orthologs. When ``True``, if a
        mouse gene maps to multiple human genes, the first mapping is
        used.
    ortholog_table_path
        Path to the ortholog TSV file. ``None`` uses the bundled table.
    """

    def __init__(
        self,
        organisms: list[str] | tuple[str, ...] = ("human", "mouse"),
        canonical_organism: str = "human",
        include_one2many: bool = False,
        ortholog_table_path: str | Path | None = None,
    ) -> None:
        self._organisms = list(organisms)
        self._canonical_organism = canonical_organism
        self._include_one2many = include_one2many
        self._table_path = Path(ortholog_table_path) if ortholog_table_path else ORTHOLOG_TABLE_PATH

        # Lazy-load the table
        self._table: pd.DataFrame | None = None
        self._mouse_to_human: dict[str, str] | None = None

    def _ensure_loaded(self) -> None:
        """Load and index the ortholog table on first access."""
        if self._table is not None:
            return

        if not self._table_path.exists():
            msg = f"Ortholog table not found: {self._table_path}"
            raise FileNotFoundError(msg)

        self._table = pd.read_csv(self._table_path, sep="\t")
        logger.info(
            "Loaded ortholog table: %d entries from %s",
            len(self._table),
            self._table_path.name,
        )

        # Filter by orthology type
        if not self._include_one2many:
            self._table = self._table[self._table["orthology_type"] == "one2one"].copy()
            logger.info("Filtered to one2one orthologs: %d entries", len(self._table))

        # Build mouse→human mapping (keep first occurrence for one2many)
        self._mouse_to_human = {}
        for _, row in self._table.iterrows():
            mouse_gene = row["mouse_gene_symbol"]
            human_gene = row["human_gene_symbol"]
            if mouse_gene not in self._mouse_to_human:
                self._mouse_to_human[mouse_gene] = human_gene

    @classmethod
    def from_config(cls, config: MultiSpeciesConfig) -> OrthologMapper:
        """Create an OrthologMapper from a configuration object.

        Parameters
        ----------
        config
            Multi-species configuration.

        Returns
        -------
        OrthologMapper
        """
        return cls(
            organisms=config.organisms,
            canonical_organism=config.canonical_organism,
            include_one2many=config.include_one2many,
            ortholog_table_path=config.ortholog_table,
        )

    def translate_gene_names(
        self,
        gene_names: list[str],
        source_organism: str,
    ) -> list[str]:
        """Translate gene names from a source organism to canonical names.

        Parameters
        ----------
        gene_names
            Gene names to translate.
        source_organism
            Organism the gene names come from (e.g. ``"mouse"``).

        Returns
        -------
        list[str]
            Translated gene names. Genes without an ortholog mapping
            are passed through as-is (they will become ``<unk>`` in
            the vocabulary).
        """
        if source_organism == self._canonical_organism:
            return list(gene_names)

        if source_organism == "mouse":
            self._ensure_loaded()
            assert self._mouse_to_human is not None
            return [self._mouse_to_human.get(g, g) for g in gene_names]

        msg = f"Unsupported source organism: {source_organism!r}. Supported: {self._organisms}"
        raise ValueError(msg)

    def get_all_canonical_genes(self) -> list[str]:
        """Get all unique canonical (human) gene names from the ortholog table.

        Returns
        -------
        list[str]
            Sorted list of unique human gene symbols.
        """
        self._ensure_loaded()
        assert self._table is not None
        return sorted(self._table["human_gene_symbol"].unique().tolist())

    @property
    def n_mapped(self) -> int:
        """Number of mapped ortholog pairs."""
        self._ensure_loaded()
        assert self._mouse_to_human is not None
        return len(self._mouse_to_human)

    @property
    def organisms(self) -> list[str]:
        """List of supported organisms."""
        return list(self._organisms)

    def __repr__(self) -> str:
        loaded = self._table is not None
        n = len(self._mouse_to_human) if self._mouse_to_human else "?"
        return (
            f"OrthologMapper(organisms={self._organisms}, "
            f"canonical={self._canonical_organism!r}, "
            f"one2many={self._include_one2many}, "
            f"n_mapped={n}, loaded={loaded})"
        )
