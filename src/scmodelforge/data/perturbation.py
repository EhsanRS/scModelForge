"""Perturbation-aware data handling for single-cell experiments.

Parses perturbation metadata from AnnData obs columns and provides
a :class:`PerturbationDataset` that extends :class:`CellDataset`
with structured perturbation information per cell.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from scmodelforge.data._utils import collate_cells
from scmodelforge.data.anndata_store import AnnDataStore

if TYPE_CHECKING:
    from pathlib import Path

    import anndata as ad

    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.data.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Perturbation metadata
# ------------------------------------------------------------------

# Common column name patterns for auto-detection
_PERTURBATION_COLUMN_CANDIDATES = [
    "perturbation",
    "condition",
    "guide_identity",
    "guide_id",
    "treatment",
    "compound",
    "drug",
    "gene_target",
    "sgRNA",
]

_DOSE_COLUMN_CANDIDATES = [
    "dose",
    "concentration",
    "dose_value",
    "dose_amount",
]


@dataclass
class PerturbationMetadata:
    """Structured perturbation information for a single cell.

    Attributes
    ----------
    perturbation_type
        Type of perturbation: ``"crispr"``, ``"chemical"``, or
        ``"unknown"``.
    perturbation_name
        Gene target (CRISPR) or compound name (chemical).
    dose
        Dose amount, or ``None`` if not available.
    dose_unit
        Unit of dose (e.g. ``"uM"``), or ``None``.
    is_control
        Whether this cell is a control.
    """

    perturbation_type: str = "unknown"
    perturbation_name: str = ""
    dose: float | None = None
    dose_unit: str | None = None
    is_control: bool = False


def detect_perturbation_columns(adata: ad.AnnData) -> dict[str, str]:
    """Auto-detect perturbation-related columns in AnnData obs.

    Checks for common column naming conventions used in perturbation
    screens.

    Parameters
    ----------
    adata
        AnnData object.

    Returns
    -------
    dict[str, str]
        Mapping from role (``"perturbation"``, ``"dose"``) to the
        detected column name. Empty dict if no columns detected.
    """
    result: dict[str, str] = {}
    obs_cols = set(adata.obs.columns)

    for candidate in _PERTURBATION_COLUMN_CANDIDATES:
        if candidate in obs_cols:
            result["perturbation"] = candidate
            break

    for candidate in _DOSE_COLUMN_CANDIDATES:
        if candidate in obs_cols:
            result["dose"] = candidate
            break

    return result


def _infer_perturbation_type(column_name: str) -> str:
    """Infer perturbation type from the column name."""
    crispr_indicators = {"guide", "sgrna", "gene_target", "crispr"}
    chemical_indicators = {"drug", "compound", "treatment", "dose"}

    lower = column_name.lower()
    if any(ind in lower for ind in crispr_indicators):
        return "crispr"
    if any(ind in lower for ind in chemical_indicators):
        return "chemical"
    return "unknown"


def parse_perturbation_metadata(
    adata: ad.AnnData,
    perturbation_key: str | None = None,
    control_label: str = "control",
    dose_key: str | None = None,
    dose_unit: str | None = None,
) -> list[PerturbationMetadata]:
    """Parse perturbation metadata for every cell in an AnnData.

    Parameters
    ----------
    adata
        AnnData object with perturbation information in obs.
    perturbation_key
        Column name for perturbation labels. If ``None``, auto-detects.
    control_label
        Label used for control cells (case-insensitive comparison).
    dose_key
        Column name for dose values. If ``None``, auto-detects.
    dose_unit
        Unit of dose values (e.g. ``"uM"``).

    Returns
    -------
    list[PerturbationMetadata]
        One metadata object per cell.
    """
    # Auto-detect columns if not specified
    if perturbation_key is None or dose_key is None:
        detected = detect_perturbation_columns(adata)
        if perturbation_key is None:
            perturbation_key = detected.get("perturbation")
        if dose_key is None:
            dose_key = detected.get("dose")

    if perturbation_key is None:
        logger.warning(
            "No perturbation column detected in adata.obs. "
            "Returning empty metadata for all cells."
        )
        return [PerturbationMetadata() for _ in range(adata.n_obs)]

    pert_type = _infer_perturbation_type(perturbation_key)
    control_lower = control_label.lower()

    result = []
    for i in range(adata.n_obs):
        pert_name = str(adata.obs[perturbation_key].iloc[i])
        is_control = pert_name.lower() == control_lower

        dose = None
        if dose_key is not None and dose_key in adata.obs.columns:
            raw = adata.obs[dose_key].iloc[i]
            try:
                dose = float(raw)
            except (ValueError, TypeError):
                dose = None

        result.append(
            PerturbationMetadata(
                perturbation_type=pert_type,
                perturbation_name=pert_name,
                dose=dose,
                dose_unit=dose_unit,
                is_control=is_control,
            )
        )

    return result


# ------------------------------------------------------------------
# PerturbationDataset
# ------------------------------------------------------------------


class PerturbationDataset(Dataset):
    """PyTorch Dataset for perturbation-annotated single-cell data.

    Extends the standard :class:`CellDataset` pattern with structured
    perturbation metadata per cell.

    Parameters
    ----------
    adata
        AnnData object(s) or path(s) to .h5ad file(s).
    gene_vocab
        Gene vocabulary for index alignment.
    perturbation_key
        Column name for perturbation labels. Auto-detects if ``None``.
    control_label
        Label used for control cells.
    dose_key
        Column name for dose values. Auto-detects if ``None``.
    dose_unit
        Unit of dose values.
    preprocessing
        Optional preprocessing pipeline.
    obs_keys
        Additional observation metadata keys to pass through.
    """

    def __init__(
        self,
        adata: ad.AnnData | str | Path | list[ad.AnnData | str | Path],
        gene_vocab: GeneVocab,
        *,
        perturbation_key: str | None = None,
        control_label: str = "control",
        dose_key: str | None = None,
        dose_unit: str | None = None,
        preprocessing: PreprocessingPipeline | None = None,
        obs_keys: list[str] | None = None,
    ) -> None:
        # Normalise to list
        if not isinstance(adata, list):
            adata = [adata]

        self.gene_vocab = gene_vocab
        self.preprocessing = preprocessing
        self.store = AnnDataStore(adata, gene_vocab, obs_keys=obs_keys)

        # Parse perturbation metadata from all loaded adatas
        self._perturbation_metadata: list[PerturbationMetadata] = []
        for ad_obj in self.store._adatas:
            self._perturbation_metadata.extend(
                parse_perturbation_metadata(
                    ad_obj,
                    perturbation_key=perturbation_key,
                    control_label=control_label,
                    dose_key=dose_key,
                    dose_unit=dose_unit,
                )
            )

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single cell with perturbation metadata.

        Returns
        -------
        dict
            Standard cell dict (``expression``, ``gene_indices``,
            ``n_genes``, ``metadata``) plus ``"perturbation"``
            (:class:`PerturbationMetadata`).
        """
        expression, gene_indices, metadata = self.store.get_cell(idx)

        if self.preprocessing is not None:
            expression = self.preprocessing(expression)

        return {
            "expression": torch.from_numpy(expression),
            "gene_indices": torch.from_numpy(gene_indices),
            "n_genes": len(expression),
            "metadata": metadata,
            "perturbation": self._perturbation_metadata[idx],
        }

    def __repr__(self) -> str:
        return (
            f"PerturbationDataset(n_cells={len(self)}, "
            f"n_datasets={self.store.n_datasets})"
        )


# ------------------------------------------------------------------
# Collation
# ------------------------------------------------------------------


def collate_perturbation_cells(
    batch: list[dict[str, Any]],
    pad_value: int = 0,
) -> dict[str, Any]:
    """Collate function for PerturbationDataset batches.

    Extends :func:`~scmodelforge.data._utils.collate_cells` with
    perturbation-specific tensors.

    Parameters
    ----------
    batch
        List of dicts from :meth:`PerturbationDataset.__getitem__`.
    pad_value
        Padding value for expression/gene arrays.

    Returns
    -------
    dict
        Standard collated batch plus:
        - ``perturbation_names``: ``list[str]`` of perturbation names
        - ``is_control``: ``BoolTensor`` of shape ``(batch_size,)``
        - ``doses``: ``FloatTensor`` of shape ``(batch_size,)``,
          ``NaN`` for missing doses
    """
    # Delegate standard collation
    collated = collate_cells(batch, pad_value=pad_value)

    # Extract perturbation-specific fields
    perturbation_names = []
    is_control = []
    doses = []

    for item in batch:
        pert: PerturbationMetadata = item["perturbation"]
        perturbation_names.append(pert.perturbation_name)
        is_control.append(pert.is_control)
        doses.append(pert.dose if pert.dose is not None else math.nan)

    collated["perturbation_names"] = perturbation_names
    collated["is_control"] = torch.tensor(is_control, dtype=torch.bool)
    collated["doses"] = torch.tensor(doses, dtype=torch.float32)

    return collated
