"""Per-class k-means text classifier."""

from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from llsvm.text_classifier import TextClassifier
from llsvm.text_embedder import TextEmbedder


class KMeansClassifier(TextClassifier):
    """Fits k centroids per class with k-means; predicts the class of the nearest centroid.

    With k=1 this is a nearest-centroid (class-mean) classifier. Larger k lets a
    class cover several separate regions of embedding space. A class with fewer
    than k examples gets one centroid per example. Distances are Euclidean.

    Args:
        embedder: Turns texts into feature vectors.
        k: Centroids per class.
        n_init: k-means restarts per class; the best (lowest inertia) is kept.
        seed: Seeds k-means initialization, for reproducibility.
    """

    def __init__(self, embedder: TextEmbedder, k: int = 1, n_init: int = 10, seed: int = 0):
        super().__init__(embedder)
        self.k = k
        self.n_init = n_init
        self.seed = seed
        self._centroids: np.ndarray | None = None  # (total_centroids, d)
        self._owners: np.ndarray | None = None  # (total_centroids,) class index of each centroid

    def config(self) -> dict[str, Any]:
        return {"k": self.k, "n_init": self.n_init, "seed": self.seed}

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        centroids, owners = [], []
        for c in range(y.max() + 1):
            Xc = X[y == c]
            km = KMeans(n_clusters=min(self.k, len(Xc)), n_init=self.n_init, random_state=self.seed).fit(Xc)
            centroids.append(km.cluster_centers_)
            owners.append(np.full(len(km.cluster_centers_), c, dtype=np.int64))
        self._centroids = np.vstack(centroids)
        self._owners = np.concatenate(owners)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2; ||x||^2 is constant per row, so drop it.
        dists = (self._centroids**2).sum(axis=1)[None, :] - 2 * X @ self._centroids.T
        return self._owners[dists.argmin(axis=1)]
