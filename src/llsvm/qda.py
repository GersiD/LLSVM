"""Quadratic discriminant analysis text classifier (scikit-learn)."""

from typing import Any

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from llsvm.text_classifier import TextClassifier
from llsvm.text_embedder import TextEmbedder


class QDAClassifier(TextClassifier):
    """QDA: one Gaussian per class, each with its own covariance; predicts the most probable class.

    Uses scikit-learn's eigen solver with covariance shrinkage, which replaces each
    class's sample covariance S by (1 - shrinkage) * S + shrinkage * (trace(S) / d) * I.
    Shrinkage is needed on text embeddings: they have hundreds of dimensions and a
    class often has fewer examples than that, which makes S singular. The default
    of 0.5 did best on a validation split of the training data for both datasets
    (MiniLM embeddings). Class priors are the class frequencies in the training data.

    Args:
        embedder: Turns texts into feature vectors.
        shrinkage: Weight in [0, 1] of the scaled identity in each class covariance.
        tol: Absolute threshold below which a covariance eigenvalue counts as zero;
            fit raises if any class covariance is rank deficient. Embedding variances
            are small (about 1/d per dimension for unit-length vectors), so a low
            shrinkage may need a lower tol.
    """

    def __init__(self, embedder: TextEmbedder, shrinkage: float = 0.5, tol: float = 1e-4):
        super().__init__(embedder)
        self.shrinkage = shrinkage
        self.tol = tol
        self._model: QuadraticDiscriminantAnalysis | None = None

    def config(self) -> dict[str, Any]:
        return {"shrinkage": self.shrinkage, "tol": self.tol}

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=self.shrinkage, tol=self.tol).fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
