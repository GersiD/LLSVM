"""Deterministic test double shared across test files."""

import hashlib
import re

import numpy as np

from llsvm.text_embedder import TextEmbedder

_CLUSTER = re.compile(r"^c(\d+)\b")


class FakeEmbedder(TextEmbedder):
    """Deterministic fake embedder for tests; no model involved.

    Every text gets a pseudo-random vector seeded by its SHA-256 digest, scaled
    by `noise`. Texts starting with "c<k>" (e.g. "c2 hello") also get +1 on
    axis k, so they form linearly separable clusters for classifier tests.
    """

    def __init__(self, size: int = 16, noise: float = 0.05):
        self._size = size
        self._noise = noise

    def config(self) -> dict:
        return {"size": self._size, "noise": self._noise}

    @property
    def size(self) -> int:
        return self._size

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self._size))
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "little")
            out[i] = self._noise * np.random.default_rng(seed).standard_normal(self._size)
            if m := _CLUSTER.match(t):
                k = int(m.group(1))
                if k >= self._size:
                    raise ValueError(f"cluster c{k} needs size > {k}, got size {self._size}")
                out[i, k] += 1.0
        return out
