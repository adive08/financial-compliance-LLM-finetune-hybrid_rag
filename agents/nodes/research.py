"""ResearchAgent: hybrid temporal retrieval for the document."""
from __future__ import annotations

from typing import Any


def _as_dict(hit: Any) -> dict:
    if hasattr(hit, "to_dict"):
        return hit.to_dict()
    if isinstance(hit, dict):
        return dict(hit)
    return {
        "chunk_id": getattr(hit, "chunk_id", ""),
        "text": getattr(hit, "text", ""),
        "metadata": getattr(hit, "metadata", {}),
        "score": getattr(hit, "score", 0.0),
    }


def _merge(existing: dict[str, dict], hits: list[dict]) -> None:
    for hit in hits:
        chunk_id = hit.get("chunk_id")
        if not chunk_id:
            continue
        current = existing.get(chunk_id)
        if current is None or hit.get("score", 0) > current.get("score", 0):
            existing[chunk_id] = hit


def research_node(state, retriever=None, config=None) -> dict:
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    segments = state.get("segments", [])
    as_of = state.get("as_of_date")
    cfg = (config or {}).get("research", {})
    max_segments = int((config or {}).get("pipeline", {}).get("max_segments", 6))
    per_segment_k = int(cfg.get("per_segment_k", 3))
    global_k = int(cfg.get("top_k", 5))
    max_regulations = int(cfg.get("max_regulations", 12))

    if retriever is None or not getattr(retriever, "ready", True):
        reason = getattr(retriever, "reason", None) or "RAG index not built"
        errors.append(f"research: {reason}")
        return {
            "retrieved": [],
            "segment_regulations": {},
            "trace": trace + ["research: skipped (no index)"],
            "errors": errors,
            "retry_requested": False,
        }

    selected = segments[:max_segments]
    segment_regulations: dict[str, list[dict]] = {}
    merged: dict[str, dict] = {}
    queries: list[str] = []

    for segment in selected:
        query = segment["text"][:1200]
        queries.append(query)
        try:
            hits = retriever.retrieve(
                query, as_of_date=as_of, top_k=per_segment_k)
        except Exception as exc:
            errors.append(f"research: retrieval failed for segment {segment['index']}: {exc}")
            hits = []
        serialised = [_as_dict(hit) for hit in hits]
        segment_regulations[str(segment["index"])] = serialised
        _merge(merged, serialised)

    retry_count = int(state.get("retry_count", 0))
    category_terms = sorted({
        str(violation.get("category"))
        for violation in state.get("violations", [])
        if violation.get("category")
    })
    if retry_count and category_terms:
        queries.extend(f"{term} regulatory disclosure requirement" for term in category_terms)

    if selected:
        global_query = " ".join(
            [str(state.get("document", {}).get("filename", ""))]
            + [segment["text"][:400] for segment in selected[:3]]
            + ([f"{term} compliance rule" for term in category_terms] if retry_count else [])
        )[:2000]
        try:
            global_hits = retriever.retrieve(
                global_query, as_of_date=as_of, top_k=global_k)
        except Exception as exc:
            errors.append(f"research: global retrieval failed: {exc}")
            global_hits = []
        _merge(merged, [_as_dict(hit) for hit in global_hits])

    retrieved = sorted(
        merged.values(), key=lambda item: item.get("score", 0), reverse=True
    )[:max_regulations]

    categories = sorted({
        item.get("metadata", {}).get("category")
        for item in retrieved
        if item.get("metadata", {}).get("category")
    })
    trace.append(
        f"research: {len(retrieved)} regulation(s) across "
        f"{len(selected)} segment(s); categories={','.join(categories) or 'none'}"
    )
    return {
        "retrieved": retrieved,
        "segment_regulations": segment_regulations,
        "trace": trace,
        "errors": errors,
        "retry_requested": False,
    }
