"""Model-agnostic contract tests, run against every classifier in conftest.CLASSIFIER_FACTORIES.

Classifiers are paired with FakeEmbedder, whose "c<k> ..." texts are linearly
separable by cluster k, so any working classifier should be perfect on them.
"""

import json
import re

import pytest

from fakes import FakeEmbedder


def cluster_data(n_classes: int, n_per_class: int, tag: str):
    texts = [f"c{k} {tag}{i}" for k in range(n_classes) for i in range(n_per_class)]
    clusters = [k for k in range(n_classes) for _ in range(n_per_class)]
    return texts, clusters


@pytest.fixture
def make(classifier_factory):
    return lambda embedder=None: classifier_factory(embedder or FakeEmbedder())


# --- Unfitted state -----------------------------------------------------------


def test_starts_unfitted(make):
    assert not make().is_fitted


def test_predict_before_fit_raises(make):
    with pytest.raises(RuntimeError):
        make().predict(["c0 hi"])


def test_classes_before_fit_raises(make):
    with pytest.raises(RuntimeError):
        make().classes


# --- Learning -----------------------------------------------------------------


def test_fit_returns_self_and_marks_fitted(make):
    clf = make()
    texts, y = cluster_data(2, 5, "train")
    assert clf.fit(texts, y) is clf
    assert clf.is_fitted


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_learns_separable_data(make, n_classes):
    # >= 90%, not 100%: tree models (XGBoost's hist method) put thresholds at
    # observed training values rather than mid-margin, so a held-out point just
    # outside the training range can land on the wrong side.
    clf = make().fit(*cluster_data(n_classes, 10, "train"))
    test_texts, test_y = cluster_data(n_classes, 5, "test")
    accuracy = sum(p == t for p, t in zip(clf.predict(test_texts), test_y)) / len(test_y)
    assert accuracy >= 0.9


def test_string_labels_round_trip(make):
    # Checks label mapping, not generalization, so it predicts on the training texts.
    names = ["ham", "spam"]
    texts, y = cluster_data(2, 10, "train")
    clf = make().fit(texts, [names[k] for k in y])
    assert clf.predict(texts) == [names[k] for k in y]


def test_label_types_preserved(make):
    texts, y = cluster_data(2, 5, "train")
    preds = make().fit(texts, y).predict(texts)
    assert all(type(p) is int for p in preds)


def test_classes_sorted(make):
    texts, y = cluster_data(3, 5, "train")
    names = ["medium", "high", "low"]
    clf = make().fit(texts, [names[k] for k in y])
    assert clf.classes == ["high", "low", "medium"]


def test_predictions_drawn_from_classes(make):
    texts, y = cluster_data(3, 5, "train")
    clf = make().fit(texts, y)
    assert set(clf.predict(cluster_data(3, 5, "test")[0])) <= set(clf.classes)


def test_training_order_does_not_matter(make):
    texts, y = cluster_data(3, 5, "train")
    order = [i for k in range(5) for i in range(k, len(texts), 5)]  # interleave classes
    a = make().fit(texts, y)
    b = make().fit([texts[i] for i in order], [y[i] for i in order])
    test_texts = cluster_data(3, 5, "test")[0]
    assert a.classes == b.classes
    assert a.predict(test_texts) == b.predict(test_texts)


def test_refit_replaces_classes(make):
    clf = make().fit(*cluster_data(2, 5, "train"))
    texts, y = cluster_data(3, 5, "train")
    clf.fit(texts, ["abc"[k] for k in y])
    assert clf.classes == ["a", "b", "c"]


def test_deterministic(make):
    clf = make().fit(*cluster_data(3, 5, "train"))
    test_texts = cluster_data(3, 5, "test")[0]
    assert clf.predict(test_texts) == clf.predict(test_texts)


# --- Predict output -----------------------------------------------------------


def test_predict_length_matches_input(make):
    clf = make().fit(*cluster_data(2, 5, "train"))
    assert len(clf.predict(["c0 a", "c1 b", "c0 c"])) == 3


def test_predict_empty(make):
    clf = make().fit(*cluster_data(2, 5, "train"))
    assert clf.predict([]) == []


# --- Input validation ---------------------------------------------------------


def test_length_mismatch_raises(make):
    with pytest.raises(ValueError):
        make().fit(["c0 a", "c1 b"], [0])


def test_single_class_raises(make):
    with pytest.raises(ValueError):
        make().fit(["c0 a", "c0 b"], ["x", "x"])


def test_empty_fit_raises(make):
    with pytest.raises(ValueError):
        make().fit([], [])


@pytest.mark.parametrize(
    "texts, labels",
    [(("c0 a", "c1 b"), [0, 1]), (["c0 a", "c1 b"], (0, 1)), (["c0 a", "c1 b"], [[0], [1]])],
    ids=["texts-tuple", "labels-tuple", "unhashable-labels"],
)
def test_fit_rejects_bad_types(make, texts, labels):
    with pytest.raises(TypeError):
        make().fit(texts, labels)


def test_predict_rejects_non_list(make):
    clf = make().fit(*cluster_data(2, 5, "train"))
    with pytest.raises(TypeError):
        clf.predict("c0 a")


# --- Identity -----------------------------------------------------------------


def test_config_is_json_serializable(make):
    json.dumps(make().config())


def test_config_hash_is_sha256_hex(make):
    assert re.fullmatch(r"[0-9a-f]{64}", make().config_hash)


def test_config_hash_stable_across_instances(make):
    assert make().config_hash == make().config_hash


def test_config_hash_depends_on_embedder(make):
    assert make(FakeEmbedder(8)).config_hash != make(FakeEmbedder(16)).config_hash
