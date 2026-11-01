"""Tests for the TextEmbedder base class itself (not tied to any model)."""

import numpy as np
import pytest

from llsvm.text_embedder import TextEmbedder


class Constant(TextEmbedder):
    """Returns ones; config is whatever is passed in."""

    def __init__(self, size: int = 3, **extra):
        self._size = size
        self._extra = extra
        self.calls = 0

    def config(self):
        return {"size": self._size, **self._extra}

    @property
    def size(self):
        return self._size

    def _embed(self, texts):
        self.calls += 1
        return np.ones((len(texts), self._size), dtype=np.float64)


class OtherConstant(Constant):
    pass


class WrongShape(Constant):
    def _embed(self, texts):
        return np.zeros((len(texts), self._size + 1))


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        TextEmbedder()


def test_subclass_missing_methods_cannot_instantiate():
    class Incomplete(TextEmbedder):
        @property
        def size(self):
            return 4

    with pytest.raises(TypeError):
        Incomplete()


def test_wrong_output_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        WrongShape().embed(["hi"])


def test_output_cast_to_float32():
    assert Constant().embed(["a"]).dtype == np.float32


def test_embed_calls_model_once_per_call():
    e = Constant()
    e.embed(["a", "b", "c"])
    assert e.calls == 1


def test_empty_input_skips_model():
    e = Constant()
    e.embed([])
    assert e.calls == 0


def test_hash_ignores_config_key_order():
    assert Constant(a=1, b=2).model_hash == Constant(b=2, a=1).model_hash


def test_hash_changes_with_config():
    assert Constant(revision="abc").model_hash != Constant(revision="def").model_hash


def test_hash_differs_across_classes_with_same_config():
    assert Constant().model_hash != OtherConstant().model_hash


def test_hash_rejects_non_serializable_config():
    with pytest.raises(TypeError):
        Constant(bad=object()).model_hash


def test_repr():
    e = Constant()
    assert repr(e) == f"Constant(size=3, model_hash={e.model_hash[:12]})"
