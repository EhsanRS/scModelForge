"""Custom encoder stack that supports ``**kwargs`` propagation.

``nn.TransformerEncoder.forward()`` does NOT pass ``**kwargs`` to layers,
which is needed for gene_bias attention (``gene_indices`` kwarg).  This
class uses an explicit loop over layers to support ``**kwargs``.

When ``attention_type`` is ``"standard"`` or ``"flash"``, the model builder
uses ``nn.TransformerEncoder`` instead (no kwargs needed), so this class
is only instantiated for ``gene_bias`` and ``linear`` attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScModelForgeEncoder(nn.Module):
    """Encoder stack with ``**kwargs`` propagation to layers.

    Parameters
    ----------
    layers
        ModuleList of encoder layers.
    norm
        Optional final LayerNorm.
    """

    def __init__(self, layers: nn.ModuleList, norm: nn.LayerNorm | None = None) -> None:
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Parameters
        ----------
        src
            Input tensor ``(B, S, D)``.
        mask
            Attention mask.
        src_key_padding_mask
            Padding mask ``(B, S)``.
        **kwargs
            Passed to each layer (e.g. ``gene_indices``).

        Returns
        -------
        torch.Tensor
            Encoded tensor ``(B, S, D)``.
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, **kwargs)
        if self.norm is not None:
            output = self.norm(output)
        return output
