"""Hybrid dense + BM25 retrieval with temporal filtering."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from rag.config import (
    chunks_path,
    embeddings_path,
    load_rag_config,
    manifest_path,
    storage_dir as resolve_storage_dir,
    vector_ids_path,
)
from rag.embeddings import get_embedder, tokenize
from rag.store import (
    BM25,
    NumpyVectorStore,
    QdrantVectorStore,
    build_bm25,
    load_chunks,
    load_vector_index,
)
from rag.temporal import is_active, temporal_status

LOG = logging.getLogger(__name__)


class IndexNotBuilt(RuntimeError):
    """Raised when retrieval is attempted before the index exists."""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "score": round(float(self.score), 6),
            "dense_score": None if self.dense_score is None else round(float(self.dense_score), 6),
            "bm25_score": None if self.bm25_score is None else round(float(self.bm25_score), 6),
        }
        payload.update(self.extra)
        return payload


def index_status(
    storage_dir: str | Path | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = config or load_rag_config()
    directory = resolve_storage_dir(cfg, storage_dir)
    manifest_file = manifest_path(directory)
    status: dict[str, Any] = {
        "ready": False,
        "storage_dir": str(directory),
        "chunks": 0,
        "documents": 0,
        "reason": None,
    }
    if not manifest_file.is_file() or not chunks_path(directory).is_file():
        status["reason"] = "index not built"
        return status
    if not embeddings_path(directory).is_file() or not vector_ids_path(directory).is_file():
        status["reason"] = "index files are incomplete; rebuild the index"
        return status
    try:
        with open(manifest_file, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("vector_backend") == "qdrant" and not (directory / "qdrant").exists():
            status["reason"] = "Qdrant collection is missing; rebuild the index"
            return status
        status.update({
            "ready": True,
            "chunks": manifest.get("chunks", 0),
            "documents": manifest.get("documents", 0),
            "built_at": manifest.get("built_at"),
            "vector_backend": manifest.get("vector_backend"),
            "embedder_backend": manifest.get("embedder_backend"),
            "embedding_model": manifest.get("embedding_model"),
            "corpus_hash": manifest.get("corpus_hash"),
        })
    except (OSError, json.JSONDecodeError) as exc:
        status["reason"] = f"unreadable manifest: {exc}"
    return status


class HybridTemporalRetriever:
    """Loads a persisted index and performs hybrid, as-of-aware retrieval."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        config: dict | None = None,
        embedder=None,
        vector_index: NumpyVectorStore | QdrantVectorStore | None = None,
    ) -> None:
        self.config = config or load_rag_config()
        self.directory = resolve_storage_dir(self.config, storage_dir)
        self.collection = self.config["storage"].get("collection", "regulations")
        self.manifest: dict[str, Any] = {}
        self.chunks: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.bm25: BM25 | None = None
        self.embedder = embedder
        self.vector_index = vector_index
        self.ready = False
        self.reason: str | None = None
        self._lock = threading.Lock()
        self._load()

    def close(self) -> None:
        close = getattr(self.vector_index, "close", None)
        if callable(close):
            close()
        self.ready = False

    def _load(self) -> None:
        manifest_file = manifest_path(self.directory)
        if not manifest_file.is_file() or not chunks_path(self.directory).is_file():
            self.reason = (
                f"No RAG index in {self.directory}. "
                "Run: python scripts/build_rag_index.py"
            )
            return
        try:
            with open(manifest_file, encoding="utf-8") as handle:
                self.manifest = json.load(handle)
            self.chunks = load_chunks(self.directory)
        except (OSError, json.JSONDecodeError) as exc:
            self.reason = f"Could not read the RAG index: {exc}"
            return

        self.by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.bm25 = build_bm25(self.chunks)

        if self.embedder is None:
            try:
                self.embedder = get_embedder(
                    backend=self.manifest.get("embedder_backend", "auto"),
                    model=self.manifest.get("embedding_model", "BAAI/bge-small-en-v1.5"),
                    dim=int(self.manifest.get("dim") or 384),
                )
            except Exception as exc:
                self.reason = f"Embedding backend unavailable: {exc}"
                return

        expected_backend = self.manifest.get("embedder_backend")
        if expected_backend and getattr(self.embedder, "backend", None) != expected_backend:
            self.reason = (
                f"Index was built with '{expected_backend}' embeddings but "
                f"'{getattr(self.embedder, 'backend', 'unknown')}' is active. Rebuild the index."
            )
            return

        if self.vector_index is None:
            try:
                self.vector_index = load_vector_index(
                    self.directory,
                    backend=self.manifest.get("vector_backend", "numpy"),
                    collection=self.collection,
                )
            except Exception as exc:
                self.reason = f"Could not load the vector index: {exc}"
                return

        self.ready = True

    def _as_of(self, as_of_date: str | date | None) -> str:
        if as_of_date is None:
            return date.today().isoformat()
        if isinstance(as_of_date, date):
            return as_of_date.isoformat()
        return str(as_of_date)

    def retrieve(
        self,
        query: str,
        as_of_date: str | date | None = None,
        categories: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not self.ready:
            raise IndexNotBuilt(self.reason or "RAG index not ready")
        query = (query or "").strip()
        if not query:
            return []

        retrieval = self.config["retrieval"]
        as_of = self._as_of(as_of_date)
        limit = int(top_k or retrieval.get("top_k", 5))
        dense_limit = max(limit, int(retrieval.get("dense_candidates", 20)))
        bm25_limit = max(limit, int(retrieval.get("bm25_candidates", 20)))
        rrf_k = float(retrieval.get("rrf_k", 60))
        boost = float(retrieval.get("category_boost", 0.05))
        categories = categories or []

        with self._lock:
            query_vector = self.embedder.embed([query])[0]
            dense_hits = self.vector_index.search(query_vector, dense_limit)
            bm25_hits = self.bm25.search(tokenize(query), bm25_limit) if self.bm25 else []

        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        fused: dict[str, float] = {}

        for rank, (chunk_id, score) in enumerate(dense_hits, start=1):
            chunk = self.by_id.get(chunk_id)
            if chunk is None or not is_active(chunk, as_of):
                continue
            dense_scores[chunk_id] = score
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

        for rank, (index, score) in enumerate(bm25_hits, start=1):
            if index >= len(self.chunks):
                continue
            chunk = self.chunks[index]
            chunk_id = chunk["chunk_id"]
            if not is_active(chunk, as_of):
                continue
            bm25_scores[chunk_id] = score
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

        if categories:
            for chunk_id, score in list(fused.items()):
                if self.by_id[chunk_id].get("category") in categories:
                    fused[chunk_id] = score * (1.0 + boost)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
        results: list[RetrievedChunk] = []
        for chunk_id, score in ranked:
            chunk = self.by_id[chunk_id]
            metadata = {
                key: chunk.get(key)
                for key in (
                    "doc_id", "title", "authority", "section", "category",
                    "effective_date", "last_updated", "superseded_by",
                    "superseded_date", "status", "url", "source",
                )
            }
            metadata["temporal_status"] = temporal_status(chunk, as_of)
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk.get("text", ""),
                metadata=metadata,
                score=score,
                dense_score=dense_scores.get(chunk_id),
                bm25_score=bm25_scores.get(chunk_id),
            ))
        return results

    def retrieve_many(
        self,
        queries: list[str],
        as_of_date: str | date | None = None,
        categories: list[str] | None = None,
        top_k: int = 5,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        for query in queries:
            for chunk in self.retrieve(
                query, as_of_date=as_of_date, categories=categories, top_k=top_k
            ):
                existing = merged.get(chunk.chunk_id)
                if existing is None or chunk.score > existing.score:
                    merged[chunk.chunk_id] = chunk
        ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ordered[:limit] if limit else ordered

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "storage_dir": str(self.directory),
            "chunks": len(self.chunks),
            "documents": self.manifest.get("documents", 0),
            "vector_backend": self.manifest.get("vector_backend"),
            "embedder_backend": self.manifest.get("embedder_backend"),
            "built_at": self.manifest.get("built_at"),
        }
