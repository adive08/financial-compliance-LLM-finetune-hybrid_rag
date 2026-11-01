"""Configuration and path helpers for the RAG layer."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "corpus": {
        "path": "data/corpus/regulations.jsonl",
    },
    "embedding": {
        "backend": "auto",
        "model": "BAAI/bge-small-en-v1.5",
        "dim": 384,
    },
    "retrieval": {
        "top_k": 5,
        "dense_candidates": 20,
        "bm25_candidates": 20,
        "rrf_k": 60,
        "category_boost": 0.05,
    },
    "chunking": {
        "max_chars": 700,
        "overlap_chars": 100,
    },
    "storage": {
        "dir": "storage",
        "collection": "regulations",
        "vector_backend": "auto",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_rag_config(path: str | os.PathLike[str] | None = None) -> dict:
    config_path = Path(path) if path else REPO_ROOT / "config" / "rag.yaml"
    loaded: dict = {}
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    return _deep_merge(DEFAULTS, loaded)


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def storage_dir(
    config: dict | None = None,
    override: str | os.PathLike[str] | None = None,
) -> Path:
    if override is not None:
        return resolve_path(override)
    cfg = config or load_rag_config()
    return resolve_path(cfg["storage"]["dir"])


def corpus_path(
    config: dict | None = None,
    override: str | os.PathLike[str] | None = None,
) -> Path:
    if override is not None:
        return resolve_path(override)
    cfg = config or load_rag_config()
    return resolve_path(cfg["corpus"]["path"])


def manifest_path(directory: str | os.PathLike[str]) -> Path:
    return Path(directory) / "manifest.json"


def chunks_path(directory: str | os.PathLike[str]) -> Path:
    return Path(directory) / "chunks.json"


def embeddings_path(directory: str | os.PathLike[str]) -> Path:
    return Path(directory) / "embeddings.npy"


def vector_ids_path(directory: str | os.PathLike[str]) -> Path:
    return Path(directory) / "vector_ids.json"
