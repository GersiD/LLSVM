"""Targeted XGBoostClassifier tests. General behavior is covered by the contract suite."""

import pytest

from fakes import FakeEmbedder
from llsvm.xgboost_classifier import XGBoostClassifier


@pytest.mark.parametrize("kwargs", [{"n_estimators": 10}, {"max_depth": 2}, {"seed": 1}], ids=lambda kw: next(iter(kw)))
def test_hyperparameters_reach_config_hash(kwargs):
    emb = FakeEmbedder()
    clf = XGBoostClassifier(emb, **kwargs)
    assert all(clf.config()[k] == v for k, v in kwargs.items())
    assert clf.config_hash != XGBoostClassifier(emb).config_hash


def test_n_jobs_does_not_change_config_hash():
    emb = FakeEmbedder()
    assert XGBoostClassifier(emb, n_jobs=1).config_hash == XGBoostClassifier(emb).config_hash
