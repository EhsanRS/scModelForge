"""BERT-style masking strategy for pretraining."""

from __future__ import annotations

import torch

from scmodelforge._constants import CLS_TOKEN_ID, MASK_TOKEN_ID, PAD_TOKEN_ID
from scmodelforge.tokenizers.base import MaskedTokenizedCell, TokenizedCell


class MaskingStrategy:
    """Apply BERT-style masking to a :class:`TokenizedCell`.

    Of the selected positions:
    - ``mask_action_ratio`` are replaced with ``[MASK]``
    - ``random_replace_ratio`` are replaced with a random vocab token
    - the rest are kept unchanged

    ``n_to_mask = max(1, round(n_maskable * mask_ratio))`` ensures at
    least one token is always masked.

    Parameters
    ----------
    mask_ratio
        Fraction of maskable tokens to select.
    mask_action_ratio
        Fraction of selected tokens replaced by ``[MASK]``.
    random_replace_ratio
        Fraction of selected tokens replaced by a random token.
    vocab_size
        Total vocabulary size (needed for random replacement).

    Raises
    ------
    ValueError
        If ratios are out of bounds or ``vocab_size`` is missing when
        ``random_replace_ratio > 0``.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_action_ratio: float = 0.8,
        random_replace_ratio: float = 0.1,
        vocab_size: int | None = None,
    ) -> None:
        if not 0.0 < mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1], got {mask_ratio}")
        if mask_action_ratio < 0 or random_replace_ratio < 0:
            raise ValueError("mask_action_ratio and random_replace_ratio must be >= 0")
        if mask_action_ratio + random_replace_ratio > 1.0:
            raise ValueError(
                f"mask_action_ratio ({mask_action_ratio}) + random_replace_ratio "
                f"({random_replace_ratio}) must be <= 1.0"
            )
        if random_replace_ratio > 0 and vocab_size is None:
            raise ValueError("vocab_size is required when random_replace_ratio > 0")

        self.mask_ratio = mask_ratio
        self.mask_action_ratio = mask_action_ratio
        self.random_replace_ratio = random_replace_ratio
        self.vocab_size = vocab_size

    def apply(self, cell: TokenizedCell, seed: int | None = None) -> MaskedTokenizedCell:
        """Apply masking to a tokenized cell.

        Parameters
        ----------
        cell
            Input tokenized cell (not modified in-place).
        seed
            Optional per-call seed for reproducibility.

        Returns
        -------
        MaskedTokenizedCell
            A new cell with masking applied.
        """
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()

        # Clone tensors to prevent aliasing
        input_ids = cell.input_ids.clone()
        attention_mask = cell.attention_mask.clone()
        values = cell.values.clone() if cell.values is not None else None
        gene_indices = cell.gene_indices.clone()

        seq_len = input_ids.shape[0]

        # Build labels: -100 everywhere by default
        labels = torch.full((seq_len,), -100, dtype=torch.long)
        masked_positions = torch.zeros(seq_len, dtype=torch.bool)

        # Determine maskable positions (not CLS, not PAD)
        maskable = (input_ids != CLS_TOKEN_ID) & (input_ids != PAD_TOKEN_ID)
        maskable_indices = torch.where(maskable)[0]
        n_maskable = maskable_indices.shape[0]

        if n_maskable == 0:
            return MaskedTokenizedCell(
                input_ids=input_ids,
                attention_mask=attention_mask,
                values=values,
                gene_indices=gene_indices,
                metadata=dict(cell.metadata),
                labels=labels,
                masked_positions=masked_positions,
            )

        # Select positions to mask
        n_to_mask = max(1, round(n_maskable * self.mask_ratio))
        perm = torch.randperm(n_maskable, generator=gen)
        selected = maskable_indices[perm[:n_to_mask]]

        # Store original token IDs as labels at selected positions
        labels[selected] = input_ids[selected]
        masked_positions[selected] = True

        # Decide action for each selected position
        n_mask_action = round(n_to_mask * self.mask_action_ratio)
        n_random = round(n_to_mask * self.random_replace_ratio)
        # Remaining positions are kept unchanged

        mask_positions = selected[:n_mask_action]
        random_positions = selected[n_mask_action : n_mask_action + n_random]
        # keep_positions = selected[n_mask_action + n_random:]  — no change needed

        # Apply [MASK] token
        input_ids[mask_positions] = MASK_TOKEN_ID

        # Apply random replacement
        if random_positions.numel() > 0 and self.vocab_size is not None:
            random_tokens = torch.randint(0, self.vocab_size, (random_positions.shape[0],), generator=gen)
            input_ids[random_positions] = random_tokens

        return MaskedTokenizedCell(
            input_ids=input_ids,
            attention_mask=attention_mask,
            values=values,
            gene_indices=gene_indices,
            metadata=dict(cell.metadata),
            labels=labels,
            masked_positions=masked_positions,
        )
