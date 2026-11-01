"""Logistic regression text classifier (scikit-learn)."""

from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from llsvm.text_classifier import TextClassifier
from llsvm.text_embedder import TextEmbedder


class LogisticRegressionClassifier(TextClassifier):
    """L2-regularized (multinomial) logistic regression on text embeddings.

    Args:
        embedder: Turns texts into feature vectors.
        C: Inverse regularization strength; larger fits the training data harder.
        class_weight: "balanced" reweights classes inversely to their frequency;
            None weights all equally.
        max_iter: Iteration cap for the lbfgs solver.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        C: float = 1.0,
        class_weight: Literal["balanced"] | None = None,
        max_iter: int = 1000,
    ):
        super().__init__(embedder)
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self._model: LogisticRegression | None = None

    def config(self) -> dict[str, Any]:
        return {"C": self.C, "class_weight": self.class_weight, "max_iter": self.max_iter}

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter).fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
