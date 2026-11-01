"""Embedding backends with a deterministic offline fallback."""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Iterable

LOG = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384


def tokenize(text: str) -> list[str]:
    """Words plus adjacent bigrams, lower-cased."""
    words = _WORD.findall((text or "").lower())
    bigrams = [f"{left}_{right}" for left, right in zip(words, words[1:])]
    return words + bigrams


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class HashingEmbedder:
    """Deterministic hashed bag-of-words/bigrams embedding."""

    backend = "hashing"

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = int(dim)
        self.model = "hashing-v1"
        self.name = self.model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            words = _WORD.findall((text or "").lower())
            features = [(word, 1.0) for word in words]
            features += [
                (f"{left}_{right}", 0.6) for left, right in zip(words, words[1:])
            ]
            for feature, weight in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign * weight
            vectors.append(_unit(vector))
        return vectors


class FastEmbedEmbedder:
    """Dense embeddings using fastembed (ONNX, CPU) when installed."""

    backend = "fastembed"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model = model
        self.name = model
        self._model = TextEmbedding(model_name=model)
        self.dim = None

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = [list(map(float, vector)) for vector in self._model.embed(list(texts))]
        if vectors and self.dim is None:
            self.dim = len(vectors[0])
        return vectors


def get_embedder(
    backend: str = "auto",
    model: str = DEFAULT_MODEL,
    dim: int = DEFAULT_DIM,
) -> HashingEmbedder | FastEmbedEmbedder:
    if backend == "hashing":
        return HashingEmbedder(dim=dim)
    try:
        return FastEmbedEmbedder(model=model)
    except Exception as exc:
        if backend == "fastembed":
            raise
        LOG.warning(
            "fastembed unavailable (%s); falling back to deterministic hashing embeddings.",
            exc,
        )
        return HashingEmbedder(dim=dim)
