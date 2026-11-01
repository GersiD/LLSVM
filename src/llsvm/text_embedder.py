"""Abstract interface for text embedding models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from llsvm.hashing import stable_hash


class TextEmbedder(ABC):
    """Maps text to fixed-size float32 vectors.

    Subclasses implement `config`, `size` and `_embed`. Callers use `embed`,
    which validates inputs and outputs so every embedder behaves the same way
    regardless of the model behind it.
    """

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Everything that affects the embeddings, as a JSON-serializable dict.

        Two instances of the same class with equal configs must produce the same
        embeddings, since `model_hash` is derived from this. Pin anything that can
        drift (e.g. a Hugging Face commit SHA rather than "main").
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """Dimension of each embedding vector."""

    @abstractmethod
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a non-empty list of strings into an array of shape (len(texts), size)."""

    @property
    def model_hash(self) -> str:
        """SHA-256 hex digest identifying this embedder's outputs, for caching."""
        return stable_hash(type(self), self.config())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts. Returns a float32 array of shape (len(texts), size)."""
        if not isinstance(texts, list):
            raise TypeError(f"texts is {type(texts).__name__}, expected list[str]")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"texts[{i}] is {type(t).__name__}, expected str")
        if not texts:
            return np.empty((0, self.size), dtype=np.float32)

        out = np.asarray(self._embed(texts), dtype=np.float32)
        expected = (len(texts), self.size)
        if out.shape != expected:
            raise ValueError(f"{type(self).__name__}._embed returned shape {out.shape}, expected {expected}")
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size}, model_hash={self.model_hash[:12]})"
