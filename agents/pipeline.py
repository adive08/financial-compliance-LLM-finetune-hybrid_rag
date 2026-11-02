"""Public entry point for document analysis."""
from __future__ import annotations

import copy
import os
import threading
from datetime import date

from agents.config import load_agents_config, resolve_mode
from agents.graph import build_graph
from agents.llm import LLMClient
from agents.nodes.advisory import template_fix
from agents.schemas import (
    build_result,
    category_label,
    severity,
)
from rag.chunking import split_segments
from rag.config import load_rag_config, manifest_path, storage_dir
from rag.retriever import HybridTemporalRetriever

_DEFAULT_LOCK = threading.Lock()
_DEFAULT_RETRIEVERS: dict[str, tuple[int | None, HybridTemporalRetriever]] = {}


def _default_retriever() -> HybridTemporalRetriever:
    """Cache the index client across requests (Qdrant local holds a file lock)."""
    config = load_rag_config()
    directory = storage_dir(config)
    manifest = manifest_path(directory)
    stamp = manifest.stat().st_mtime_ns if manifest.is_file() else None
    key = f"{directory}|{config['storage'].get('collection')}"

    with _DEFAULT_LOCK:
        cached = _DEFAULT_RETRIEVERS.get(key)
        if cached and cached[0] == stamp and cached[1].ready:
            return cached[1]
        if cached:
            try:
                cached[1].close()
            except Exception:
                pass
        retriever = HybridTemporalRetriever(config=config)
        _DEFAULT_RETRIEVERS[key] = (stamp, retriever)
        return retriever


def _segments(text: str, config: dict) -> list[dict]:
    pipeline_cfg = config["pipeline"]
    pieces = split_segments(
        text,
        max_chars=int(pipeline_cfg.get("segment_chars", 300)),
        overlap=int(pipeline_cfg.get("segment_overlap", 60)),
    )
    pieces = pieces[: int(pipeline_cfg.get("max_segments", 6))]
    return [{"index": index, "text": piece} for index, piece in enumerate(pieces)]


def _run_single(text: str, document: dict) -> dict:
    from app.inference import MODEL, make_snippet

    snippet = make_snippet(text)
    parsed = MODEL.analyze(snippet)
    violations: list[dict] = []
    errors: list[str] = []
    if parsed.get("issue_found") is None:
        errors.append("Model reply could not be parsed as JSON.")
    elif parsed.get("issue_found"):
        category = str(parsed.get("category") or "unknown")
        violations.append({
            "id": "V1",
            "category": category,
            "category_label": category_label(category),
            "severity": severity(parsed.get("severity")),
            "explanation": parsed.get("explanation") or "",
            "original_snippet": snippet,
            "suggested_fix": template_fix(category),
            "cited_regulation": None,
            "confidence": 0.55,
        })

    if violations:
        verdict, confidence = "non_compliant", 0.55
    elif errors:
        verdict, confidence = "needs_review", 0.35
    else:
        verdict, confidence = "compliant", 0.6

    document = dict(document)
    document["segments_analyzed"] = 1
    document["regulations_retrieved"] = 0
    return build_result(
        document=document,
        violations=violations,
        verdict=verdict,
        confidence=confidence,
        trace=["single: fine-tuned adapter (no retrieval)"],
        errors=errors,
    )


def run_analysis(
    document_text: str,
    filename: str | None = None,
    as_of_date: str | None = None,
    mode: str | None = None,
    llm=None,
    retriever=None,
    config: dict | None = None,
    apply_advisory: bool = True,
) -> dict:
    cfg = config or load_agents_config()
    chosen_mode = resolve_mode(mode, cfg)
    text = (document_text or "").strip()
    if not text:
        raise ValueError("The document contains no analysable text.")

    effective_date = as_of_date or date.today().isoformat()
    document = {
        "filename": filename or "uploaded-document",
        "as_of_date": effective_date,
        "characters": len(text),
        "excerpt": text[:700],
        "mode": chosen_mode,
    }

    if chosen_mode == "single":
        return _run_single(text, document)

    runtime_config = copy.deepcopy(cfg)
    if not apply_advisory:
        runtime_config["advisory"]["use_model"] = False

    segments = _segments(text, runtime_config)
    if not segments:
        raise ValueError("The document could not be split into analysable segments.")

    state = {
        "document": document,
        "segments": segments,
        "as_of_date": effective_date,
        "retrieved": [],
        "segment_regulations": {},
        "violations": [],
        "trace": [],
        "errors": [],
        "retry_count": 0,
        "retry_requested": False,
    }

    if retriever is None:
        retriever = _default_retriever()
    if llm is None:
        llm = LLMClient()

    graph = build_graph(llm=llm, retriever=retriever, config=runtime_config)
    final_state = graph.invoke(state)
    result = final_state.get("result")
    if not result:
        result = build_result(
            document=document,
            violations=[],
            verdict="needs_review",
            confidence=0.0,
            trace=final_state.get("trace", []),
            errors=final_state.get("errors", []) + ["pipeline: no result produced"],
        )
    return result


def pipeline_mode() -> str:
    return resolve_mode(os.environ.get("PIPELINE_MODE"))
