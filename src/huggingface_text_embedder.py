"""TextEmbedder backed by a sentence-transformers model on the Hugging Face Hub."""

from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from text_embedder import TextEmbedder

# Weights for other runtimes that sentence-transformers (PyTorch) never loads.
# Many repos ship these too, often several times the size of the model itself.
_IGNORE_PATTERNS = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.ot", "*.msgpack"]


class HuggingFaceTextEmbedder(TextEmbedder):
    """Downloads (or reuses the local HF cache of) a model and embeds with it.

    Args:
        model_id: Hub repo id, e.g. "sentence-transformers/all-MiniLM-L6-v2".
        revision: Branch, tag or commit SHA. Always resolved to a commit SHA,
            which goes into `config()` so `model_hash` pins exact weights.
        normalize: L2-normalize embeddings. Most embedding models are trained
            for cosine similarity, which a linear SVM sees only on unit vectors.
        max_seq_length: Token limit; longer texts are truncated. None keeps the
            model's default.
        batch_size: Texts per forward pass. Doesn't affect outputs beyond
            padding-level float noise, so it isn't part of `config()`.
        device: "cpu", "cuda", "mps", or None to auto-select. Not part of
            `config()` for the same reason.
        progress: Show a progress bar while embedding. Not part of `config()`.
    """

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        normalize: bool = True,
        max_seq_length: int | None = None,
        batch_size: int = 32,
        device: str | None = None,
        progress: bool = False,
    ):
        local_path = Path(snapshot_download(model_id, revision=revision, ignore_patterns=_IGNORE_PATTERNS))
        self._model_id = model_id
        self._revision = local_path.name  # HF cache dirs are named by commit SHA
        self._normalize = normalize
        self._batch_size = batch_size
        self._progress = progress
        self._model = SentenceTransformer(str(local_path), device=device)
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length
        self._size = int(self._model.get_embedding_dimension())

    def config(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "revision": self._revision,
            "normalize": self._normalize,
            "max_seq_length": self._model.max_seq_length,
        }

    @property
    def size(self) -> int:
        return self._size

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=self._progress,
        )
