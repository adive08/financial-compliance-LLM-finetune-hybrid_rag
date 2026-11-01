"""Temporal validity rules for regulatory chunks."""
from __future__ import annotations

from datetime import date


def parse_date(value: str | date | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def temporal_status(metadata: dict, as_of: str | date | None = None) -> str:
    """Return not_yet_effective, superseded or current for a chunk."""
    check_date = parse_date(as_of) or date.today()
    effective = parse_date(metadata.get("effective_date"))
    superseded_date = parse_date(metadata.get("superseded_date"))

    if effective and effective > check_date:
        return "not_yet_effective"
    if metadata.get("superseded_by"):
        if superseded_date is None:
            return "superseded"
        if check_date >= superseded_date:
            return "superseded"
    return "current"


def is_active(metadata: dict, as_of: str | date | None = None) -> bool:
    return temporal_status(metadata, as_of) == "current"


def filter_active(chunks: list[dict], as_of: str | date | None = None) -> list[dict]:
    return [chunk for chunk in chunks if is_active(chunk, as_of)]


def version_chain(records: list[dict], doc_id: str) -> list[dict]:
    """Follow superseded_by links from any version to the newest document."""
    by_id = {record["doc_id"]: record for record in records}
    if doc_id not in by_id:
        return []
    chain = [by_id[doc_id]]
    seen = {doc_id}
    current = by_id[doc_id]
    while current.get("superseded_by"):
        next_id = current["superseded_by"]
        if next_id in seen or next_id not in by_id:
            break
        seen.add(next_id)
        current = by_id[next_id]
        chain.append(current)
    return chain


def newest_version(records: list[dict], doc_id: str) -> dict | None:
    chain = version_chain(records, doc_id)
    return chain[-1] if chain else None
