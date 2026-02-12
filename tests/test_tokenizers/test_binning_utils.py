"""Tests for binning utility functions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scmodelforge.tokenizers._utils import compute_bin_edges, digitize_expression


class TestComputeBinEdges:
    def test_uniform_default(self):
        edges = compute_bin_edges(n_bins=51, method="uniform", value_max=10.0)
        assert edges.shape == (52,)
        assert edges[0] == 0.0
        assert edges[-1] == 10.0

    def test_uniform_custom_max(self):
        edges = compute_bin_edges(n_bins=10, method="uniform", value_max=5.0)
        assert edges.shape == (11,)
        np.testing.assert_allclose(edges, np.linspace(0.0, 5.0, 11))

    def test_quantile_basic(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(0.0, 10.0, size=1000).astype(np.float32)
        edges = compute_bin_edges(values=values, n_bins=10, method="quantile")
        # Should be monotonically increasing and deduplicated
        assert all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))

    def test_quantile_requires_values(self):
        with pytest.raises(ValueError, match="values are required"):
            compute_bin_edges(n_bins=10, method="quantile")

    def test_quantile_all_zeros_falls_back(self):
        values = np.zeros(100, dtype=np.float32)
        edges = compute_bin_edges(values=values, n_bins=10, method="quantile", value_max=10.0)
        # Falls back to uniform
        assert edges[0] == 0.0
        assert edges[-1] == 10.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown binning method"):
            compute_bin_edges(n_bins=10, method="invalid")

    def test_quantile_deduplicates(self):
        # Many repeated values should still produce unique edges
        values = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0, 10.0], dtype=np.float32)
        edges = compute_bin_edges(values=values, n_bins=10, method="quantile")
        assert len(edges) == len(np.unique(edges))


class TestDigitizeExpression:
    def test_basic_uniform(self):
        edges = compute_bin_edges(n_bins=10, method="uniform", value_max=10.0)
        values = torch.tensor([0.0, 1.0, 5.0, 9.9, 10.0], dtype=torch.float32)
        bin_ids = digitize_expression(values, edges)
        assert bin_ids.dtype == torch.long
        assert bin_ids[0] == 0  # zero always maps to bin 0
        assert all(0 <= b < 10 for b in bin_ids)

    def test_zero_always_bin_zero(self):
        edges = compute_bin_edges(n_bins=5, method="uniform", value_max=10.0)
        values = torch.tensor([0.0, 0.0, 5.0, 0.0], dtype=torch.float32)
        bin_ids = digitize_expression(values, edges)
        assert bin_ids[0] == 0
        assert bin_ids[1] == 0
        assert bin_ids[3] == 0
        assert bin_ids[2] > 0
