"""Targeted SVM tests. General classifier behavior is covered by the contract suite."""

import numpy as np
import pytest

from fakes import FakeEmbedder
from llsvm.svm import KernelSVM, LinearSVM


@pytest.mark.parametrize(
    "cls, kwargs",
    [(LinearSVM, {"C": 2.0}), (LinearSVM, {"loss": "hinge"}), (KernelSVM, {"gamma": 0.5}), (KernelSVM, {"kernel": "poly"})],
    ids=["linear-C", "linear-loss", "kernel-gamma", "kernel-kernel"],
)
def test_hyperparameters_reach_config_hash(cls, kwargs):
    emb = FakeEmbedder()
    clf = cls(emb, **kwargs)
    assert all(clf.config()[k] == v for k, v in kwargs.items())
    assert clf.config_hash != cls(emb).config_hash


def test_rbf_learns_xor_where_linear_cannot():
    # Calls _fit/_predict directly to feed hand-made 2-D features.
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32).repeat(5, axis=0)
    X += np.random.default_rng(0).normal(scale=0.05, size=X.shape).astype(np.float32)
    y = np.array([0, 0, 1, 1]).repeat(5)

    rbf = KernelSVM(FakeEmbedder(), kernel="rbf", C=10.0)
    rbf._fit(X, y)
    assert (rbf._predict(X) == y).all()

    linear = LinearSVM(FakeEmbedder())
    linear._fit(X, y)
    assert (linear._predict(X) == y).mean() < 1.0


def test_balanced_class_weight_recovers_minority_class():
    # 40 vs 2 examples with strong regularization: unweighted sides with the majority.
    # LinearSVM only: liblinear regularizes the intercept, so the class imbalance in
    # the loss decides the boundary. SVC's intercept is unregularized, and at this C
    # every point is a bounded SV, leaving the intercept underdetermined.
    texts = [f"c0 maj{i}" for i in range(40)] + ["c1 min0", "c1 min1"]
    labels = ["ham"] * 40 + ["spam"] * 2
    test = ["c1 new0", "c1 new1", "c1 new2"]

    plain = LinearSVM(FakeEmbedder(), C=0.01).fit(texts, labels)
    balanced = LinearSVM(FakeEmbedder(), C=0.01, class_weight="balanced").fit(texts, labels)
    assert plain.predict(test) == ["ham"] * 3
    assert balanced.predict(test) == ["spam"] * 3
