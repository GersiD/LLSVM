"""Targeted KMeansClassifier tests. General classifier behavior is covered by the contract suite."""

import numpy as np

from fakes import FakeEmbedder
from llsvm.kmeans import KMeansClassifier


def test_k_reaches_config_hash():
    emb = FakeEmbedder()
    assert KMeansClassifier(emb, k=3).config_hash != KMeansClassifier(emb).config_hash


def test_k1_centroids_are_class_means():
    texts = ["c0 a", "c0 b", "c1 a", "c1 b", "c1 c"]
    clf = KMeansClassifier(FakeEmbedder(), k=1).fit(texts, [0, 0, 1, 1, 1])
    X = clf.embedder.embed(texts)
    np.testing.assert_allclose(clf._centroids, [X[:2].mean(axis=0), X[2:].mean(axis=0)], atol=1e-6)


def test_class_smaller_than_k_gets_one_centroid_per_example():
    texts = [f"c0 {i}" for i in range(10)] + ["c1 a", "c1 b"]
    clf = KMeansClassifier(FakeEmbedder(), k=5).fit(texts, [0] * 10 + [1] * 2)
    np.testing.assert_array_equal(np.bincount(clf._owners), [5, 2])


def test_k2_learns_xor_where_k1_cannot():
    # Each class sits at two opposite corners, so both class means land in the
    # middle. Calls _fit/_predict directly to feed hand-made 2-D features.
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32).repeat(5, axis=0)
    X += np.random.default_rng(0).normal(scale=0.05, size=X.shape).astype(np.float32)
    y = np.array([0, 0, 1, 1]).repeat(5)

    k2 = KMeansClassifier(FakeEmbedder(), k=2)
    k2._fit(X, y)
    assert (k2._predict(X) == y).all()

    k1 = KMeansClassifier(FakeEmbedder(), k=1)
    k1._fit(X, y)
    assert (k1._predict(X) == y).mean() < 1.0
