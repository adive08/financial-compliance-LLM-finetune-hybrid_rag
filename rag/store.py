"""Vector stores, BM25 and persisted chunk storage."""
from __future__ import annotations

import json
import logging
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from rag.config import embeddings_path, vector_ids_path
from rag.embeddings import tokenize

LOG = logging.getLogger(__name__)

POINT_NAMESPACE = uuid.UUID("6f8b0a90-1b64-4a8f-9c52-6f2f4a1d8c11")


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class BM25:
    """Small Okapi BM25 implementation, used when rank_bm25 is unavailable."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.total = len(documents)
        self.lengths = [len(document) for document in documents]
        self.avg_length = (sum(self.lengths) / self.total) if self.total else 0.0
        self.frequencies = [Counter(document) for document in documents]
        document_frequency: Counter = Counter()
        for document in documents:
            document_frequency.update(set(document))
        self.idf = {
            term: math.log(1 + (self.total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query_tokens: list[str], limit: int = 20) -> list[tuple[int, float]]:
        if not self.total or not query_tokens:
            return []
        scores = [0.0] * self.total
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.frequencies):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                length = self.lengths[index] or 1
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_length or 1)
                )
                scores[index] += idf * (frequency * (self.k1 + 1)) / denominator
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in ranked[:limit] if score > 0]


class NumpyVectorStore:
    backend = "numpy"

    def __init__(self, ids: list[str], vectors: Iterable[Iterable[float]]) -> None:
        self.ids = list(ids)
        matrix = np.asarray(list(vectors), dtype="float32")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        self.vectors = _normalise(matrix)

    @classmethod
    def load(cls, directory: str | Path) -> "NumpyVectorStore":
        matrix = np.load(embeddings_path(directory))
        with open(vector_ids_path(directory), encoding="utf-8") as handle:
            ids = json.load(handle)
        return cls(ids, matrix)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        np.save(embeddings_path(directory), self.vectors)
        with open(vector_ids_path(directory), "w", encoding="utf-8") as handle:
            json.dump(self.ids, handle)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        query = np.asarray(vector, dtype="float32")
        norm = np.linalg.norm(query)
        if norm:
            query = query / norm
        scores = self.vectors @ query
        order = np.argsort(-scores)[:limit]
        return [(self.ids[int(index)], float(scores[int(index)])) for index in order]


class QdrantVectorStore:
    """Qdrant in embedded/local mode; no server or Docker required."""

    backend = "qdrant"

    def __init__(
        self, path: str | Path, collection: str, require_existing: bool = True
    ) -> None:
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = QdrantClient(path=str(path))
        if require_existing:
            names = {item.name for item in self.client.get_collections().collections}
            if collection not in names:
                self.close()
                raise RuntimeError(f"Qdrant collection {collection!r} does not exist")

    def recreate(self, dim: int) -> None:
        from qdrant_client.http import models as qmodels

        names = {item.name for item in self.client.get_collections().collections}
        if self.collection in names:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=dim, distance=qmodels.Distance.COSINE),
        )

    def upsert(self, ids: list[str], vectors: Iterable[Iterable[float]], payloads: list[dict]) -> None:
        from qdrant_client.http import models as qmodels

        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid5(POINT_NAMESPACE, chunk_id)),
                vector=list(map(float, vector)),
                payload={"chunk_id": chunk_id, **payload},
            )
            for chunk_id, vector, payload in zip(ids, vectors, payloads)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        query = list(map(float, vector))
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query,
                limit=limit,
                with_payload=True,
            )
            points = response.points
        except AttributeError:
            points = self.client.search(
                collection_name=self.collection,
                query_vector=query,
                limit=limit,
                with_payload=True,
            )
        results = []
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                results.append((str(chunk_id), float(point.score)))
        return results


def resolve_vector_backend(preferred: str = "auto") -> str:
    if preferred == "numpy":
        return "numpy"
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        if preferred == "qdrant":
            LOG.warning("qdrant-client not installed; using the numpy vector store.")
        return "numpy"
    return "qdrant"


def write_vector_index(
    directory: str | Path,
    chunks: list[dict],
    vectors: list[list[float]],
    backend: str,
    collection: str,
) -> str:
    directory = Path(directory)
    ids = [chunk["chunk_id"] for chunk in chunks]
    NumpyVectorStore(ids, vectors).save(directory)

    if backend != "qdrant":
        return "numpy"

    payloads = []
    for chunk in chunks:
        payloads.append({
            key: chunk.get(key)
            for key in (
                "doc_id", "title", "authority", "section", "category",
                "effective_date", "last_updated", "superseded_by",
                "superseded_date", "status", "url", "source",
            )
        })
    store = None
    try:
        store = QdrantVectorStore(
            directory / "qdrant", collection, require_existing=False)
        store.recreate(len(vectors[0]))
        store.upsert(ids, vectors, payloads)
        return "qdrant"
    except Exception as exc:
        LOG.warning("Qdrant index failed (%s); using the numpy vector store.", exc)
        return "numpy"
    finally:
        if store is not None:
            store.close()


def load_vector_index(
    directory: str | Path,
    backend: str,
    collection: str,
) -> NumpyVectorStore | QdrantVectorStore:
    if backend == "qdrant":
        try:
            return QdrantVectorStore(Path(directory) / "qdrant", collection)
        except Exception as exc:
            LOG.warning("Could not open Qdrant index (%s); falling back to numpy.", exc)
    return NumpyVectorStore.load(directory)


def save_chunks(directory: str | Path, chunks: list[dict]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "chunks.json", "w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)


def load_chunks(directory: str | Path) -> list[dict]:
    with open(Path(directory) / "chunks.json", encoding="utf-8") as handle:
        return json.load(handle)


def build_bm25(chunks: list[dict]) -> BM25:
    return BM25([tokenize(chunk.get("text", "")) for chunk in chunks])
