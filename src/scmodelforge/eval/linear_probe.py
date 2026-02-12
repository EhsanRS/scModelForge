"""Linear probe benchmark — logistic regression on cell embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scmodelforge.eval.base import BaseBenchmark, BenchmarkResult
from scmodelforge.eval.registry import register_benchmark

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


@register_benchmark("linear_probe")
class LinearProbeBenchmark(BaseBenchmark):
    """Evaluate embeddings via logistic regression on cell-type labels.

    Splits data into train/test, fits an ``sklearn.linear_model.LogisticRegression``,
    and reports accuracy, macro-F1, and weighted-F1.

    Parameters
    ----------
    cell_type_key
        Column in ``adata.obs`` with cell-type labels.
    test_size
        Fraction of data to hold out for testing.
    max_iter
        Maximum iterations for the solver.
    seed
        Random seed for the train/test split and solver.
    """

    def __init__(
        self,
        cell_type_key: str = "cell_type",
        test_size: float = 0.2,
        max_iter: int = 1000,
        seed: int = 42,
    ) -> None:
        self._cell_type_key = cell_type_key
        self._test_size = test_size
        self._max_iter = max_iter
        self._seed = seed

    @property
    def name(self) -> str:
        return "linear_probe"

    @property
    def required_obs_keys(self) -> list[str]:
        return [self._cell_type_key]

    def run(
        self,
        embeddings: np.ndarray,
        adata: AnnData,
        dataset_name: str,
    ) -> BenchmarkResult:
        """Run the linear probe benchmark.

        Parameters
        ----------
        embeddings
            Cell embeddings of shape ``(n_cells, hidden_dim)``.
        adata
            AnnData with cell-type annotations in ``.obs``.
        dataset_name
            Dataset identifier for the result.

        Returns
        -------
        BenchmarkResult
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
        except ImportError as e:
            msg = "scikit-learn is required for LinearProbeBenchmark. Install with: pip install scikit-learn"
            raise ImportError(msg) from e

        self.validate_adata(adata)

        labels = adata.obs[self._cell_type_key].values
        le = LabelEncoder()
        y = le.fit_transform(labels)

        x_train, x_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=self._test_size, random_state=self._seed, stratify=y,
        )

        clf = LogisticRegression(max_iter=self._max_iter, random_state=self._seed)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        }

        metadata = {
            "n_cells": len(embeddings),
            "n_classes": len(le.classes_),
            "test_size": self._test_size,
            "cell_type_key": self._cell_type_key,
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=dataset_name,
            metrics=metrics,
            metadata=metadata,
        )
