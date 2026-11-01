"""SVM text classifiers backed by scikit-learn.

Hyperparameters are validated by scikit-learn when `fit` is called.
"""

from typing import Any, Literal

import numpy as np
from sklearn.svm import SVC, LinearSVC

from llsvm.text_classifier import TextClassifier
from llsvm.text_embedder import TextEmbedder


class LinearSVM(TextClassifier):
    """Linear SVM via liblinear (LinearSVC). Scales to large datasets; one-vs-rest multiclass.

    Args:
        embedder: Turns texts into feature vectors.
        C: Inverse regularization strength; larger fits the training data harder.
        class_weight: "balanced" reweights classes inversely to their frequency
            (useful for imbalanced data like SMS spam); None weights all equally.
        loss: "squared_hinge" (liblinear default) or "hinge" (textbook SVM).
        tol: Stopping tolerance.
        max_iter: Iteration cap for the solver.
        seed: Seeds liblinear's coordinate shuffling, for reproducibility.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        C: float = 1.0,
        class_weight: Literal["balanced"] | None = None,
        loss: Literal["hinge", "squared_hinge"] = "squared_hinge",
        tol: float = 1e-4,
        max_iter: int = 1000,
        seed: int = 0,
    ):
        super().__init__(embedder)
        self.C = C
        self.class_weight = class_weight
        self.loss = loss
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self._svc: LinearSVC | None = None

    def config(self) -> dict[str, Any]:
        return {
            "C": self.C,
            "class_weight": self.class_weight,
            "loss": self.loss,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "seed": self.seed,
        }

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._svc = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            loss=self.loss,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.seed,
        ).fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict(X)


class KernelSVM(TextClassifier):
    """Kernel SVM via libsvm (SVC). Training is roughly O(n^2) or worse, so best
    for datasets up to ~tens of thousands of texts; one-vs-one multiclass.

    Args:
        embedder: Turns texts into feature vectors.
        kernel: "rbf", "poly", "sigmoid" or "linear".
        C: Inverse regularization strength.
        gamma: Kernel coefficient for rbf/poly/sigmoid: "scale" (1 / (d * X.var())),
            "auto" (1 / d), or a positive float.
        degree: Polynomial degree (poly only).
        coef0: Independent term (poly and sigmoid only).
        class_weight: "balanced" or None, as in LinearSVM.
        tol: Stopping tolerance.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        kernel: Literal["rbf", "poly", "sigmoid", "linear"] = "rbf",
        C: float = 1.0,
        gamma: Literal["scale", "auto"] | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        class_weight: Literal["balanced"] | None = None,
        tol: float = 1e-3,
    ):
        super().__init__(embedder)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.class_weight = class_weight
        self.tol = tol
        self._svc: SVC | None = None

    def config(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "class_weight": self.class_weight,
            "tol": self.tol,
        }

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._svc = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            class_weight=self.class_weight,
            tol=self.tol,
        ).fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict(X)
