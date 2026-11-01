"""Build or update the persisted hybrid RAG index."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.chunking import chunk_corpus
from rag.config import (
    chunks_path,
    corpus_path,
    embeddings_path,
    load_rag_config,
    manifest_path,
    storage_dir as resolve_storage_dir,
    vector_ids_path,
)
from rag.corpus import corpus_hash, load_corpus
from rag.embeddings import get_embedder
from rag.store import resolve_vector_backend, save_chunks, write_vector_index

LOG = logging.getLogger(__name__)


def _existing_manifest(directory: Path) -> dict[str, Any]:
    path = manifest_path(directory)
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _index_files_exist(directory: Path) -> bool:
    return (
        chunks_path(directory).is_file()
        and embeddings_path(directory).is_file()
        and vector_ids_path(directory).is_file()
    )


def build_index(
    corpus_file: str | Path | None = None,
    storage_dir: str | Path | None = None,
    config: dict | None = None,
    force: bool = False,
    embedder_backend: str | None = None,
    vector_backend: str | None = None,
) -> dict[str, Any]:
    cfg = config or load_rag_config()
    directory = resolve_storage_dir(cfg, storage_dir)
    source = corpus_path(cfg, corpus_file)
    directory.mkdir(parents=True, exist_ok=True)

    records = load_corpus(source, cfg)
    digest = corpus_hash(source, cfg)
    chunk_cfg = cfg["chunking"]
    chunks = chunk_corpus(
        records,
        max_chars=int(chunk_cfg.get("max_chars", 700)),
        overlap=int(chunk_cfg.get("overlap_chars", 100)),
    )
    if not chunks:
        raise ValueError("Corpus produced no chunks.")

    backend_pref = embedder_backend or cfg["embedding"].get("backend", "auto")
    embedder = get_embedder(
        backend=backend_pref,
        model=cfg["embedding"].get("model", "BAAI/bge-small-en-v1.5"),
        dim=int(cfg["embedding"].get("dim", 384)),
    )
    embedder_name = getattr(embedder, "model", getattr(embedder, "name", backend_pref))
    embedder_kind = getattr(embedder, "backend", backend_pref)

    preferred_vector = vector_backend or cfg["storage"].get("vector_backend", "auto")
    resolved_vector = resolve_vector_backend(preferred_vector)
    current_chunking = {
        "max_chars": int(chunk_cfg.get("max_chars", 700)),
        "overlap_chars": int(chunk_cfg.get("overlap_chars", 100)),
    }

    previous = _existing_manifest(directory)
    unchanged = (
        not force
        and _index_files_exist(directory)
        and previous.get("corpus_hash") == digest
        and previous.get("embedder_backend") == embedder_kind
        and previous.get("embedding_model") == embedder_name
        and previous.get("vector_backend") == resolved_vector
        and previous.get("chunking") == current_chunking
        and previous.get("chunks") == len(chunks)
    )
    if unchanged:
        return {
            "status": "up_to_date",
            "storage_dir": str(directory),
            "documents": len(records),
            "chunks": len(chunks),
            "vector_backend": previous.get("vector_backend"),
            "embedder_backend": previous.get("embedder_backend"),
            "corpus_hash": digest,
        }

    LOG.info("Embedding %d chunks with %s", len(chunks), embedder_name)
    vectors = embedder.embed(chunk["text"] for chunk in chunks)
    if not vectors:
        raise ValueError("Embedding backend returned no vectors.")
    dim = len(vectors[0])

    save_chunks(directory, chunks)
    used_vector = write_vector_index(
        directory,
        chunks,
        vectors,
        backend=resolved_vector,
        collection=cfg["storage"].get("collection", "regulations"),
    )

    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_file": str(source),
        "corpus_hash": digest,
        "documents": len(records),
        "chunks": len(chunks),
        "dim": dim,
        "embedder_backend": embedder_kind,
        "embedding_model": embedder_name,
        "vector_backend": used_vector,
        "collection": cfg["storage"].get("collection", "regulations"),
        "chunking": current_chunking,
    }
    with open(manifest_path(directory), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "status": "built",
        "storage_dir": str(directory),
        "documents": len(records),
        "chunks": len(chunks),
        "vector_backend": used_vector,
        "embedder_backend": embedder_kind,
        "corpus_hash": digest,
    }
