"""Targeted QDAClassifier tests. General classifier behavior is covered by the contract suite."""

import numpy as np

from fakes import FakeEmbedder
from llsvm.logistic_regression import LogisticRegressionClassifier
from llsvm.qda import QDAClassifier


def test_shrinkage_reaches_config_hash():
    emb = FakeEmbedder()
    assert QDAClassifier(emb, shrinkage=0.2).config_hash != QDAClassifier(emb).config_hash


def test_separates_classes_by_spread_where_linear_cannot():
    # Both classes are centered at the origin; only their spread differs, so no
    # hyperplane separates them but per-class covariances do. Calls _fit/_predict
    # directly to feed hand-made 2-D features.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(scale=0.1, size=(200, 2)), rng.normal(scale=1.0, size=(200, 2))])
    y = np.array([0] * 200 + [1] * 200)

    qda = QDAClassifier(FakeEmbedder(), shrinkage=0.0)
    qda._fit(X, y)
    assert (qda._predict(X) == y).mean() > 0.9

    linear = LogisticRegressionClassifier(FakeEmbedder())
    linear._fit(X, y)
    assert (linear._predict(X) == y).mean() < 0.7
