"""ComplianceCheckerAgent: fine-tuned model plus retrieved regulations."""
from __future__ import annotations

from app.inference import ModelUnavailable
from agents.schemas import severity


def checker_node(state, llm, config=None) -> dict:
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    violations: list[dict] = []

    cfg = (config or {}).get("pipeline", {})
    research_cfg = (config or {}).get("research", {})
    max_segments = int(cfg.get("max_segments", 6))
    max_context_chars = int(research_cfg.get("max_context_chars", 900))
    include_in_prompt = bool(research_cfg.get("include_in_prompt", False))
    segments = state.get("segments", [])[:max_segments]
    per_segment = state.get("segment_regulations", {})
    fallback = state.get("retrieved", [])

    if llm is None:
        errors.append("checker: no language model client configured")
        return {
            "violations": [],
            "trace": trace + ["checker: skipped (no model client)"],
            "errors": errors,
        }

    for segment in segments:
        regulations = per_segment.get(str(segment["index"])) or fallback[:3]
        citation_pool = {
            item.get("chunk_id"): item
            for item in list(regulations) + list(fallback)
            if item.get("chunk_id")
        }
        try:
            parsed = llm.analyze(
                segment["text"],
                regulations=(regulations or None) if include_in_prompt else None,
                max_regulation_chars=max_context_chars,
            )
        except ModelUnavailable:
            raise
        except Exception as exc:
            errors.append(f"checker: model failed on segment {segment['index']}: {exc}")
            continue

        if parsed.get("issue_found") is None:
            errors.append(
                f"checker: unstructured model reply on segment {segment['index']}"
            )
            continue
        if not parsed.get("issue_found"):
            continue

        category = str(parsed.get("category") or "unknown")
        explanation = str(parsed.get("explanation") or "").strip()
        violations.append({
            "category": category,
            "severity": severity(parsed.get("severity")),
            "explanation": explanation or "The model flagged this excerpt as non-compliant.",
            "original_snippet": segment["text"],
            "suggested_fix": "",
            "cited_regulation": None,
            "_regulations": list(citation_pool.values()),
        })

    trace.append(
        f"checker: {len(segments)} segment(s) analysed, {len(violations)} issue(s)"
    )
    return {"violations": violations, "trace": trace, "errors": errors}
