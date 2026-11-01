"""HuggingFaceTextEmbedder-specific tests (the shared contract runs via conftest)."""

import re

import numpy as np
import pytest

from llsvm.huggingface_text_embedder import HuggingFaceTextEmbedder

pytestmark = pytest.mark.slow

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def minilm():
    return HuggingFaceTextEmbedder(MODEL_ID)


def test_revision_resolved_to_commit_sha(minilm):
    assert re.fullmatch(r"[0-9a-f]{40}", minilm.config()["revision"])


def test_pinned_sha_matches_branch(minilm):
    sha = minilm.config()["revision"]
    assert HuggingFaceTextEmbedder(MODEL_ID, revision=sha).model_hash == minilm.model_hash


def test_known_dimension(minilm):
    assert minilm.size == 384


def test_normalized_outputs_are_unit_length(minilm):
    norms = np.linalg.norm(minilm.embed(["hello", "a much longer sentence about nothing"]), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_normalize_changes_hash(minilm):
    assert HuggingFaceTextEmbedder(MODEL_ID, normalize=False).model_hash != minilm.model_hash


def test_max_seq_length_changes_hash(minilm):
    assert HuggingFaceTextEmbedder(MODEL_ID, max_seq_length=32).model_hash != minilm.model_hash


def test_batch_size_does_not_change_hash(minilm):
    assert HuggingFaceTextEmbedder(MODEL_ID, batch_size=4).model_hash == minilm.model_hash


def test_similar_texts_closer_than_unrelated(minilm):
    a, b, c = minilm.embed(["Your package has shipped", "Your order is on its way", "I love pizza"])
    assert a @ b > a @ c
