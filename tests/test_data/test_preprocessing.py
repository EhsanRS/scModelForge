"""Tests for PreprocessingPipeline."""

from __future__ import annotations

import numpy as np
import pytest

from scmodelforge.data.preprocessing import PreprocessingPipeline, select_highly_variable_genes


class TestPreprocessingPipeline:
    def test_library_size_normalization(self):
        pipe = PreprocessingPipeline(normalize="library_size", target_sum=1e4, log1p=False)
        raw = np.array([10.0, 20.0, 30.0, 40.0])
        result = pipe(raw)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result.sum(), 1e4, rtol=1e-5)

    def test_library_size_preserves_ratios(self):
        pipe = PreprocessingPipeline(normalize="library_size", target_sum=100, log1p=False)
        raw = np.array([10.0, 30.0, 60.0])
        result = pipe(raw)
        np.testing.assert_allclose(result, [10.0, 30.0, 60.0], rtol=1e-5)

    def test_log1p(self):
        pipe = PreprocessingPipeline(normalize=None, log1p=True)
        raw = np.array([0.0, 1.0, 99.0])
        result = pipe(raw)
        np.testing.assert_allclose(result, np.log1p(raw), rtol=1e-5)

    def test_no_preprocessing(self):
        pipe = PreprocessingPipeline(normalize=None, log1p=False)
        raw = np.array([1.0, 2.0, 3.0])
        result = pipe(raw)
        np.testing.assert_allclose(result, raw.astype(np.float32))

    def test_full_pipeline(self):
        pipe = PreprocessingPipeline(normalize="library_size", target_sum=1e4, log1p=True)
        raw = np.array([100.0, 200.0, 300.0, 400.0])
        result = pipe(raw)
        # Manually compute expected
        total = raw.sum()
        normalized = raw * (1e4 / total)
        expected = np.log1p(normalized)
        np.testing.assert_allclose(result, expected.astype(np.float32), rtol=1e-5)

    def test_zero_expression_cell(self):
        pipe = PreprocessingPipeline(normalize="library_size", target_sum=1e4, log1p=True)
        raw = np.array([0.0, 0.0, 0.0])
        result = pipe(raw)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_unknown_normalization_raises(self):
        pipe = PreprocessingPipeline(normalize="unknown_method")
        with pytest.raises(ValueError, match="Unknown normalisation"):
            pipe(np.array([1.0, 2.0]))

    def test_does_not_modify_input(self):
        pipe = PreprocessingPipeline(normalize="library_size", target_sum=1e4, log1p=True)
        raw = np.array([100.0, 200.0])
        raw_copy = raw.copy()
        pipe(raw)
        np.testing.assert_array_equal(raw, raw_copy)

    def test_repr(self):
        pipe = PreprocessingPipeline(normalize="library_size", target_sum=1e4, log1p=True)
        r = repr(pipe)
        assert "library_size" in r
        assert "log1p=True" in r


class TestHVGSelection:
    def test_selects_correct_number(self):
        rng = np.random.default_rng(42)
        expressions = rng.poisson(5, size=(100, 50)).astype(float)
        mask = select_highly_variable_genes(expressions, n_top_genes=10)
        assert mask.sum() == 10
        assert mask.dtype == bool
        assert len(mask) == 50

    def test_n_top_genes_exceeds_total(self):
        expressions = np.ones((10, 5))
        mask = select_highly_variable_genes(expressions, n_top_genes=100)
        assert mask.sum() == 5  # Capped at total genes

    def test_selects_high_variance_genes(self):
        # Gene 0 has high variance, gene 1 is constant
        expressions = np.array(
            [
                [0.0, 5.0, 5.0],
                [100.0, 5.0, 5.0],
                [0.0, 5.0, 5.0],
                [100.0, 5.0, 5.0],
            ]
        )
        mask = select_highly_variable_genes(expressions, n_top_genes=1)
        assert mask[0]  # Gene 0 should be selected (highest dispersion)
