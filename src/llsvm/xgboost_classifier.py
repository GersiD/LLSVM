"""Gradient-boosted trees text classifier (XGBoost).

Named xgboost_classifier.py, not xgboost.py, so it doesn't shadow the xgboost package.
"""

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from llsvm.text_classifier import TextClassifier
from llsvm.text_embedder import TextEmbedder


class XGBoostClassifier(TextClassifier):
    """XGBoost on text embeddings (histogram tree method; softmax for > 2 classes).

    Args:
        embedder: Turns texts into feature vectors.
        n_estimators: Number of boosting rounds (trees).
        max_depth: Maximum depth of each tree.
        learning_rate: Shrinkage applied to each tree's contribution.
        subsample: Fraction of rows sampled per tree.
        colsample_bytree: Fraction of embedding dimensions sampled per tree.
        seed: Seeds row/column sampling, for reproducibility.
        n_jobs: Threads to train with. Doesn't change the model, so it isn't
            part of `config()`.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        seed: int = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(embedder)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.n_jobs = n_jobs
        self._model: XGBClassifier | None = None

    def config(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.seed,
        }

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            tree_method="hist",
        ).fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
