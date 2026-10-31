"""Tests for the TextClassifier base class itself (not tied to any model)."""

import numpy as np
import pytest

from fakes import FakeEmbedder
from text_classifier import TextClassifier


class Recorder(TextClassifier):
    """Records what _fit receives; _predict returns `output` if set, else all zeros."""

    output = None

    def config(self):
        return {}

    def _fit(self, X, y):
        self.X, self.y = X, y

    def _predict(self, X):
        return self.output if self.output is not None else np.zeros(len(X), dtype=np.int64)


class OtherRecorder(Recorder):
    pass


class FailingFit(Recorder):
    def _fit(self, X, y):
        raise RuntimeError("boom")


class CountingEmbedder(FakeEmbedder):
    calls = 0

    def _embed(self, texts):
        self.calls += 1
        return super()._embed(texts)


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        TextClassifier(FakeEmbedder())


def test_starts_unfitted():
    clf = Recorder(FakeEmbedder())
    assert not clf.is_fitted
    with pytest.raises(RuntimeError):
        clf.predict(["a"])


def test_fit_passes_float32_embeddings_and_class_indices():
    clf = Recorder(FakeEmbedder(4)).fit(["a", "b", "c"], ["spam", "ham", "spam"])
    assert clf.X.shape == (3, 4) and clf.X.dtype == np.float32
    np.testing.assert_array_equal(clf.y, [1, 0, 1])  # classes == ["ham", "spam"]
    assert clf.y.dtype == np.int64


def test_predict_maps_indices_back_to_labels():
    clf = Recorder(FakeEmbedder()).fit(["a", "b"], ["spam", "ham"])
    clf.output = np.array([1, 0, 1])
    assert clf.predict(["x", "y", "z"]) == ["spam", "ham", "spam"]


def test_unsortable_labels_keep_first_seen_order():
    clf = Recorder(FakeEmbedder()).fit(["a", "b", "c"], ["x", 1, "x"])
    assert clf.classes == ["x", 1]


def test_failed_fit_leaves_unfitted():
    clf = FailingFit(FakeEmbedder())
    with pytest.raises(RuntimeError, match="boom"):
        clf.fit(["a", "b"], [0, 1])
    assert not clf.is_fitted


def test_classes_returns_copy():
    clf = Recorder(FakeEmbedder()).fit(["a", "b"], [0, 1])
    clf.classes.append(99)
    assert clf.classes == [0, 1]


@pytest.mark.parametrize(
    "output, match",
    [
        (np.zeros(3, dtype=np.int64), "shape"),
        (np.zeros(2, dtype=np.float64), "dtype"),
        (np.array([0, 2]), "outside"),
        (np.array([-1, 0]), "outside"),
    ],
    ids=["wrong-shape", "float-dtype", "too-large", "negative"],
)
def test_bad_predict_output_raises(output, match):
    clf = Recorder(FakeEmbedder()).fit(["a", "b"], [0, 1])
    clf.output = output
    with pytest.raises(ValueError, match=match):
        clf.predict(["a", "b"])


def test_embeds_once_per_fit_and_predict():
    emb = CountingEmbedder()
    clf = Recorder(emb).fit(["a", "b", "c"], [0, 1, 0])
    clf.predict(["d", "e"])
    assert emb.calls == 2


def test_config_hash_differs_across_classes():
    emb = FakeEmbedder()
    assert Recorder(emb).config_hash != OtherRecorder(emb).config_hash


def test_config_hash_depends_on_embedder():
    assert Recorder(FakeEmbedder(8)).config_hash != Recorder(FakeEmbedder(16)).config_hash


def test_repr():
    clf = Recorder(FakeEmbedder())
    assert repr(clf) == f"Recorder(embedder={clf.embedder!r}, fitted=False)"
