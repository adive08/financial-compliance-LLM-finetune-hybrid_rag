"""Regulatory corpus loading, validation and hashing."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from rag.config import corpus_path, load_rag_config
from shared.prompt_format import CATEGORIES

REQUIRED_FIELDS = (
    "doc_id",
    "title",
    "authority",
    "section",
    "category",
    "text",
    "effective_date",
)
VALID_STATUS = {"current", "superseded", "draft"}


class CorpusError(ValueError):
    """Raised when the regulatory corpus is malformed."""


def _parse_date(value: str, field: str, doc_id: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise CorpusError(f"{doc_id}: invalid {field} {value!r}") from exc


def validate_record(record: dict) -> dict:
    missing = [field for field in REQUIRED_FIELDS if not record.get(field)]
    if missing:
        raise CorpusError(f"record {record.get('doc_id', '<unknown>')} missing {missing}")

    doc_id = str(record["doc_id"])
    if record["category"] not in CATEGORIES:
        raise CorpusError(f"{doc_id}: unknown category {record['category']!r}")

    _parse_date(record["effective_date"], "effective_date", doc_id)
    if record.get("last_updated"):
        _parse_date(record["last_updated"], "last_updated", doc_id)
    if record.get("superseded_date"):
        _parse_date(record["superseded_date"], "superseded_date", doc_id)

    status = record.get("status")
    if status and status not in VALID_STATUS:
        raise CorpusError(f"{doc_id}: invalid status {status!r}")

    canonical = dict(record)
    canonical.setdefault("last_updated", record["effective_date"])
    canonical.setdefault("superseded_by", None)
    canonical.setdefault("superseded_date", None)
    canonical.setdefault("url", None)
    canonical.setdefault("source", "synthetic")
    canonical.setdefault(
        "status",
        "superseded" if canonical["superseded_by"] else "current",
    )
    canonical["doc_id"] = doc_id
    canonical["text"] = str(record["text"]).strip()
    return canonical


def load_corpus(path: str | Path | None = None, config: dict | None = None) -> list[dict]:
    cfg = config or load_rag_config()
    target = Path(path) if path else corpus_path(cfg)
    if not target.is_file():
        raise CorpusError(
            f"Corpus not found at {target}. Run: python data/corpus/make_corpus.py"
        )

    records: list[dict] = []
    seen: set[str] = set()
    with open(target, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{target}:{line_no} invalid JSON: {exc}") from exc
            record = validate_record(raw)
            if record["doc_id"] in seen:
                raise CorpusError(f"{target}:{line_no} duplicate doc_id {record['doc_id']}")
            seen.add(record["doc_id"])
            records.append(record)

    if not records:
        raise CorpusError(f"{target} contains no regulations.")
    return records


def corpus_hash(path: str | Path | None = None, config: dict | None = None) -> str:
    cfg = config or load_rag_config()
    target = Path(path) if path else corpus_path(cfg)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest
