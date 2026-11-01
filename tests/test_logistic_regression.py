"""Targeted LogisticRegressionClassifier tests. General behavior is covered by the contract suite."""

import pytest

from fakes import FakeEmbedder
from llsvm.logistic_regression import LogisticRegressionClassifier


@pytest.mark.parametrize("kwargs", [{"C": 2.0}, {"class_weight": "balanced"}, {"max_iter": 50}], ids=lambda kw: next(iter(kw)))
def test_hyperparameters_reach_config_hash(kwargs):
    emb = FakeEmbedder()
    clf = LogisticRegressionClassifier(emb, **kwargs)
    assert all(clf.config()[k] == v for k, v in kwargs.items())
    assert clf.config_hash != LogisticRegressionClassifier(emb).config_hash
