"""Model-agnostic contract tests, run against every embedder in conftest.EMBEDDER_FACTORIES."""

import json
import re

import numpy as np
import pytest

TEXTS = [
    "Your package is out for delivery.",
    "WINNER!! You have been selected to receive a £900 prize reward!",
    "ok see u at 7",
]


def test_config_is_json_serializable(embedder):
    config = embedder.config()
    assert isinstance(config, dict)
    json.dumps(config)


def test_model_hash_is_sha256_hex(embedder):
    assert re.fullmatch(r"[0-9a-f]{64}", embedder.model_hash)


def test_model_hash_stable_across_instances(embedder, embedder_factory):
    assert embedder_factory().model_hash == embedder.model_hash


def test_same_hash_means_same_embeddings(embedder, embedder_factory):
    # The point of model_hash: a fresh instance with the same hash must
    # reproduce cached embeddings.
    np.testing.assert_allclose(embedder_factory().embed(TEXTS), embedder.embed(TEXTS), atol=1e-4)


def test_size_is_positive_int(embedder):
    assert isinstance(embedder.size, int) and embedder.size > 0


def test_embed_shape_and_dtype(embedder):
    out = embedder.embed(TEXTS)
    assert out.shape == (len(TEXTS), embedder.size)
    assert out.dtype == np.float32


def test_empty_input(embedder):
    out = embedder.embed([])
    assert out.shape == (0, embedder.size)
    assert out.dtype == np.float32


def test_outputs_are_finite(embedder):
    assert np.isfinite(embedder.embed(TEXTS)).all()


def test_deterministic(embedder):
    np.testing.assert_array_equal(embedder.embed(TEXTS), embedder.embed(TEXTS))


def test_batch_matches_individual(embedder):
    # Padding in batched inference can cause tiny numerical differences.
    batched = embedder.embed(TEXTS)
    single = np.concatenate([embedder.embed([t]) for t in TEXTS])
    np.testing.assert_allclose(batched, single, atol=1e-4)


def test_preserves_order(embedder):
    forward = embedder.embed(TEXTS)
    backward = embedder.embed(TEXTS[::-1])
    np.testing.assert_allclose(forward, backward[::-1], atol=1e-4)


def test_distinct_texts_get_distinct_embeddings(embedder):
    out = embedder.embed(TEXTS)
    for i in range(len(TEXTS)):
        for j in range(i + 1, len(TEXTS)):
            assert not np.allclose(out[i], out[j])


@pytest.mark.parametrize("text", ["", " ", "🎉🔥 lol", "ünïcödé", "x" * 10_000])
def test_edge_case_texts(embedder, text):
    out = embedder.embed([text])
    assert out.shape == (1, embedder.size)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("bad", ["a single string", tuple(TEXTS), iter(TEXTS)], ids=["str", "tuple", "iterator"])
def test_rejects_non_list(embedder, bad):
    with pytest.raises(TypeError):
        embedder.embed(bad)


@pytest.mark.parametrize("bad", [[None], ["ok", 3], [b"bytes"]])
def test_rejects_non_str_items(embedder, bad):
    with pytest.raises(TypeError):
        embedder.embed(bad)
