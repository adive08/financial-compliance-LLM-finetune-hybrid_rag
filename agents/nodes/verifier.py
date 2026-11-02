"""VerifierAgent: dedupe, ground citations, score and finalise."""
from __future__ import annotations

from agents.schemas import (
    SEVERITY_RANK,
    build_result,
    category_label,
    citation_from_chunk,
    normalize_snippet,
    severity,
)


def _choose_citation(regulations: list[dict], category: str) -> dict | None:
    if not regulations:
        return None
    matching = [
        item for item in regulations
        if item.get("metadata", {}).get("category") == category
    ]
    if not matching:
        return None

    def rank(item: dict) -> tuple[float, str]:
        metadata = item.get("metadata", {})
        return (
            round(float(item.get("score") or 0), 2),
            str(metadata.get("effective_date") or ""),
        )

    return citation_from_chunk(max(matching, key=rank))


def _dedupe(violations: list[dict]) -> list[dict]:
    kept: dict[tuple[str, str], dict] = {}
    for violation in violations:
        category = str(violation.get("category") or "unknown")
        key = (category, normalize_snippet(violation.get("original_snippet", ""))[:160])
        current = kept.get(key)
        if current is None:
            kept[key] = violation
            continue
        if SEVERITY_RANK.get(violation.get("severity"), 0) > SEVERITY_RANK.get(
            current.get("severity"), 0
        ):
            kept[key] = violation
        existing = current.get("_regulations") or []
        incoming = violation.get("_regulations") or []
        merged = {item.get("chunk_id"): item for item in existing + incoming}
        kept[key]["_regulations"] = list(merged.values())
    return list(kept.values())


def _violation_confidence(violation: dict) -> float:
    score = 0.55
    if violation.get("cited_regulation"):
        score += 0.2
    if violation.get("explanation"):
        score += 0.1
    if violation.get("severity") in {"low", "medium", "high"}:
        score += 0.05
    regulations = violation.get("_regulations") or []
    if regulations:
        score += min(0.05, 0.01 * len(regulations))
    return min(0.95, score)


def verifier_node(state, config=None) -> dict:
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    retrieved = state.get("retrieved", [])
    raw_violations = state.get("violations", [])
    verifier_cfg = (config or {}).get("verifier", {})
    threshold = float(verifier_cfg.get("confidence_threshold", 0.55))
    max_retries = int(verifier_cfg.get("max_retries", 1))
    max_violations = int((config or {}).get("pipeline", {}).get("max_violations", 12))
    retry_count = int(state.get("retry_count", 0))

    deduped = sorted(
        _dedupe(raw_violations),
        key=lambda item: SEVERITY_RANK.get(item.get("severity"), 0),
        reverse=True,
    )[:max_violations]

    violations: list[dict] = []
    for index, raw in enumerate(deduped, start=1):
        violation = dict(raw)
        category = str(violation.get("category") or "unknown")
        regulations = violation.pop("_regulations", [])
        violation["id"] = f"V{index}"
        violation["category"] = category
        violation["category_label"] = category_label(category)
        violation["severity"] = severity(violation.get("severity"))
        violation["cited_regulation"] = _choose_citation(regulations, category)
        violation["confidence"] = round(_violation_confidence(violation), 3)
        violations.append(violation)

    if violations:
        verdict = "non_compliant"
        confidence = sum(item["confidence"] for item in violations) / len(violations)
    elif errors:
        verdict = "needs_review"
        confidence = 0.4
    else:
        verdict = "compliant"
        confidence = 0.72 if not errors else 0.4

    if retrieved:
        confidence += 0.03
    if errors:
        confidence -= min(0.15, 0.05 * len(errors))
        if verdict == "non_compliant" and len(errors) >= 3:
            verdict = "needs_review"
    confidence = max(0.0, min(0.98, confidence))

    retry_requested = (
        retry_count < max_retries
        and confidence < threshold
        and bool(retrieved)
        and (not violations or not any(item.get("cited_regulation") for item in violations))
    )

    document = dict(state.get("document", {}))
    document["segments_analyzed"] = len(state.get("segments", []))
    document["regulations_retrieved"] = len(retrieved)

    trace.append(
        f"verifier: verdict={verdict} confidence={confidence:.3f} "
        f"violations={len(violations)} retry={retry_requested}"
    )
    result = build_result(
        document=document,
        violations=violations,
        verdict=verdict,
        confidence=confidence,
        trace=trace,
        errors=errors,
    )
    return {
        "violations": violations,
        "result": result,
        "trace": trace,
        "errors": errors,
        "retry_requested": retry_requested,
        "retry_count": retry_count + 1 if retry_requested else retry_count,
    }
