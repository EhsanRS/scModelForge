"""CELLxGENE Census data loading utilities.

Provides helpers to query the CZ CELLxGENE Census database and return
standard AnnData objects compatible with the existing
:class:`~scmodelforge.data.dataset.CellDataset` pipeline.

The ``cellxgene-census`` package is an optional dependency.  Install it
with ``pip install scModelForge[census]``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anndata as ad

    from scmodelforge.config.schema import CensusConfig

logger = logging.getLogger(__name__)


def build_obs_value_filter(filters: dict[str, Any]) -> str | None:
    """Convert a structured filter dict to a SOMA ``obs_value_filter`` string.

    Parameters
    ----------
    filters
        Mapping of column names to values.  Supported value types:

        * **str** — equality: ``tissue == 'brain'``
        * **list** — membership: ``tissue in ['brain', 'lung']``
        * **bool** — boolean equality: ``is_primary_data == True``
        * **int / float** — numeric equality: ``n_genes == 500``

    Returns
    -------
    str or None
        SOMA-compatible filter string, or ``None`` if *filters* is empty.
    """
    if not filters:
        return None

    clauses: list[str] = []
    for key, value in filters.items():
        if isinstance(value, list):
            items = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
            clauses.append(f"{key} in [{items}]")
        elif isinstance(value, bool):
            clauses.append(f"{key} == {value}")
        elif isinstance(value, str):
            clauses.append(f"{key} == '{value}'")
        else:
            # int / float
            clauses.append(f"{key} == {value}")

    return " and ".join(clauses)


def load_census_adata(
    census_config: CensusConfig,
    obs_keys: list[str] | None = None,
) -> ad.AnnData:
    """Load an AnnData object from CELLxGENE Census.

    Opens a Census connection, builds the query filter, and calls
    ``cellxgene_census.get_anndata()``.  The *obs_keys* parameter is
    merged with ``census_config.obs_columns`` so downstream modules can
    request additional metadata columns.

    Parameters
    ----------
    census_config
        Census configuration specifying organism, version, and filters.
    obs_keys
        Extra ``obs`` column names to include (e.g. label keys needed
        for fine-tuning).  Merged with ``census_config.obs_columns``.

    Returns
    -------
    anndata.AnnData
        AnnData loaded from Census.

    Raises
    ------
    ImportError
        If ``cellxgene-census`` is not installed.
    """
    try:
        import cellxgene_census
    except ImportError:
        msg = (
            "cellxgene-census is required for Census data loading. "
            "Install it with: pip install 'scModelForge[census]'"
        )
        raise ImportError(msg) from None

    # Build obs_value_filter
    obs_filter = census_config.obs_value_filter
    if obs_filter is None and census_config.filters:
        obs_filter = build_obs_value_filter(census_config.filters)

    # Merge obs column requests
    column_names_set: set[str] = set()
    if census_config.obs_columns:
        column_names_set.update(census_config.obs_columns)
    if obs_keys:
        column_names_set.update(obs_keys)
    column_names = sorted(column_names_set) if column_names_set else None

    logger.info(
        "Loading Census data: organism=%s, version=%s, filter=%s",
        census_config.organism,
        census_config.census_version,
        obs_filter,
    )

    with cellxgene_census.open_soma(census_version=census_config.census_version) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism=census_config.organism,
            obs_value_filter=obs_filter,
            var_value_filter=census_config.var_value_filter,
            column_names={"obs": column_names} if column_names else None,
        )

    logger.info("Loaded %d cells from Census", adata.n_obs)
    return adata
