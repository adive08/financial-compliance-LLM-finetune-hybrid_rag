"""Sentence-aware text chunking shared by ingestion and document analysis."""
from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    step = max(1, max_chars - overlap)
    start = 0
    while start < len(text):
        pieces.append(text[start:start + max_chars].strip())
        if start + max_chars >= len(text):
            break
        start += step
    return [piece for piece in pieces if piece]


def split_text(text: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    cleaned = _normalise(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    if overlap >= max_chars:
        overlap = max_chars // 4

    sentences = [part.strip() for part in _SENTENCE_BREAK.split(cleaned) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)
        tail: list[str] = []
        tail_len = 0
        for sentence in reversed(current):
            if tail_len + len(sentence) + 1 > overlap and tail:
                break
            tail.insert(0, sentence)
            tail_len += len(sentence) + 1
        current = tail
        current_len = tail_len

    for sentence in sentences:
        if len(sentence) > max_chars:
            flush()
            pieces = _hard_split(sentence, max_chars, overlap)
            chunks.extend(pieces[:-1])
            current = [pieces[-1]] if pieces else []
            current_len = len(current[0]) if current else 0
            continue

        if current and current_len + len(sentence) + 1 > max_chars:
            flush()

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunk = " ".join(current).strip()
        if chunk and (not chunks or chunks[-1] != chunk):
            chunks.append(chunk)
    return chunks


def chunk_document(record: dict, max_chars: int = 700, overlap: int = 100) -> list[dict]:
    """Turn one corpus record into one or more retrievable chunks."""
    pieces = split_text(record["text"], max_chars=max_chars, overlap=overlap)
    chunks: list[dict] = []
    for index, piece in enumerate(pieces):
        chunk = {
            "chunk_id": f"{record['doc_id']}::{index}",
            "chunk_index": index,
            "text": piece,
        }
        for key in (
            "doc_id",
            "title",
            "authority",
            "section",
            "category",
            "effective_date",
            "last_updated",
            "superseded_by",
            "superseded_date",
            "status",
            "url",
            "source",
        ):
            chunk[key] = record.get(key)
        chunks.append(chunk)
    return chunks


def chunk_corpus(records: list[dict], max_chars: int = 700, overlap: int = 100) -> list[dict]:
    chunks: list[dict] = []
    seen: set[str] = set()
    for record in records:
        for chunk in chunk_document(record, max_chars=max_chars, overlap=overlap):
            if chunk["chunk_id"] in seen:
                raise ValueError(f"duplicate chunk_id {chunk['chunk_id']}")
            seen.add(chunk["chunk_id"])
            chunks.append(chunk)
    return chunks
