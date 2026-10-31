"""Abstract interface for text classifiers built on a TextEmbedder."""

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Any, Self

import numpy as np

from hashing import stable_hash
from text_embedder import TextEmbedder


class TextClassifier(ABC):
    """Classifies text by embedding it and classifying the embeddings.

    Callers use `fit` and `predict` with raw text and any hashable labels.
    Subclasses implement `config`, `_fit` and `_predict`, which only ever see
    embeddings and integer class indices 0..k-1 (indices into `classes`).
    """

    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder
        self._classes: list[Hashable] | None = None

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """The classifier's hyperparameters, as a JSON-serializable dict."""

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on X (n, d) float32 and y (n,) int64 with values in 0..k-1, k >= 2."""

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Return an int array of shape (n,) with values in 0..k-1. X is non-empty."""

    @property
    def config_hash(self) -> str:
        """Identifies the untrained setup (classifier config + embedder), e.g. for logging results."""
        return stable_hash(type(self), {"classifier": self.config(), "embedder": self.embedder.model_hash})

    @property
    def is_fitted(self) -> bool:
        return self._classes is not None

    @property
    def classes(self) -> list[Hashable]:
        """Labels seen during fit. Sorted if they're mutually comparable, else in first-seen order."""
        if self._classes is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        return list(self._classes)

    def fit(self, texts: list[str], labels: list[Hashable]) -> Self:
        if not isinstance(texts, list):
            raise TypeError(f"texts is {type(texts).__name__}, expected list[str]")
        if not isinstance(labels, list):
            raise TypeError(f"labels is {type(labels).__name__}, expected list")
        if len(texts) != len(labels):
            raise ValueError(f"got {len(texts)} texts but {len(labels)} labels")
        classes = _unique_labels(labels)
        if len(classes) < 2:
            raise ValueError(f"need at least 2 classes to fit, got {len(classes)}")

        index = {c: i for i, c in enumerate(classes)}
        y = np.array([index[label] for label in labels], dtype=np.int64)
        X = self.embedder.embed(texts)

        self._classes = None  # stay unfitted if _fit raises
        self._fit(X, y)
        self._classes = classes
        return self

    def predict(self, texts: list[str]) -> list[Hashable]:
        """Predict a label for each text, in order."""
        classes = self.classes  # raises if not fitted
        X = self.embedder.embed(texts)
        if len(X) == 0:
            return []

        idx = np.asarray(self._predict(X))
        name = type(self).__name__
        if idx.shape != (len(X),):
            raise ValueError(f"{name}._predict returned shape {idx.shape}, expected ({len(X)},)")
        if not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"{name}._predict returned dtype {idx.dtype}, expected an integer dtype")
        if idx.min() < 0 or idx.max() >= len(classes):
            raise ValueError(f"{name}._predict returned indices outside 0..{len(classes) - 1}")
        return [classes[i] for i in idx]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(embedder={self.embedder!r}, fitted={self.is_fitted})"


def _unique_labels(labels: list[Hashable]) -> list[Hashable]:
    """Unique labels, sorted when possible so class indices don't depend on data order."""
    unique = list(dict.fromkeys(labels))  # raises TypeError for unhashable labels
    try:
        return sorted(unique)
    except TypeError:
        return unique
