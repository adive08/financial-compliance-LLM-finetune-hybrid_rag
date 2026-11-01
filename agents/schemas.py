"""Structured output helpers for the multi-agent pipeline."""
from __future__ import annotations

import re

from shared.prompt_format import CATEGORIES

SCHEMA_VERSION = "2.0"
ALLOWED_VERDICTS = {"compliant", "non_compliant", "needs_review"}
ALLOWED_SEVERITIES = ("none", "low", "medium", "high")
SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": 0}

CITATION_FIELDS = (
    "chunk_id", "doc_id", "title", "authority", "section", "category",
    "effective_date", "last_updated", "superseded_by", "superseded_date",
    "status", "url", "temporal_status", "score",
)


def category_label(category: str | None) -> str:
    if not category:
        return "Unknown"
    return CATEGORIES.get(category, str(category).replace("_", " ").title())


def severity(value: str | None) -> str:
    value = str(value or "unknown").lower()
    return value if value in ALLOWED_SEVERITIES else "unknown"


def citation_from_chunk(chunk: dict | None) -> dict | None:
    if not chunk:
        return None
    metadata = chunk.get("metadata", chunk)
    citation = {}
    for field in CITATION_FIELDS:
        if field in ("chunk_id", "score", "temporal_status"):
            continue
        citation[field] = metadata.get(field)
    citation["chunk_id"] = chunk.get("chunk_id") or metadata.get("chunk_id")
    citation["score"] = chunk.get("score")
    citation["temporal_status"] = metadata.get("temporal_status")
    if not citation.get("doc_id"):
        return None
    return citation


def normalize_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def build_summary(verdict: str, violations: list[dict], errors: list[str]) -> str:
    if verdict == "non_compliant":
        labels = sorted({violation.get("category_label", "issue") for violation in violations})
        return (
            f"{len(violations)} potential compliance issue(s) identified "
            f"across {len(labels)} category/categories: {', '.join(labels)}."
        )
    if verdict == "needs_review":
        return (
            "The analysis could not reach a confident verdict. Review the "
            "document manually and check the reported agent errors."
        )
    return "No compliance issue was identified in the analysed excerpts."


def build_result(
    document: dict,
    violations: list[dict],
    verdict: str,
    confidence: float,
    trace: list[str],
    errors: list[str],
) -> dict:
    verdict = verdict if verdict in ALLOWED_VERDICTS else "needs_review"
    return {
        "schema_version": SCHEMA_VERSION,
        "document": document,
        "violations": violations,
        "verdict": verdict,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
        "summary": build_summary(verdict, violations, errors),
        "agent_trace": list(trace),
        "errors": list(errors),
    }


def validate_result(result: dict) -> list[str]:
    problems: list[str] = []
    for key in ("schema_version", "document", "violations", "verdict", "confidence"):
        if key not in result:
            problems.append(f"missing key: {key}")
    if result.get("verdict") not in ALLOWED_VERDICTS:
        problems.append(f"invalid verdict: {result.get('verdict')}")
    if not isinstance(result.get("violations"), list):
        problems.append("violations is not a list")
    for index, violation in enumerate(result.get("violations", [])):
        for key in (
            "category", "severity", "explanation", "original_snippet",
            "suggested_fix", "cited_regulation",
        ):
            if key not in violation:
                problems.append(f"violation {index} missing {key}")
    return problems
