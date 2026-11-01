"""Shared state passed between LangGraph nodes."""
from __future__ import annotations

from typing import Any, TypedDict


class AnalysisState(TypedDict, total=False):
    document: dict[str, Any]
    segments: list[dict[str, Any]]
    as_of_date: str
    retrieved: list[dict[str, Any]]
    segment_regulations: dict[str, list[dict[str, Any]]]
    violations: list[dict[str, Any]]
    result: dict[str, Any]
    trace: list[str]
    errors: list[str]
    retry_count: int
    retry_requested: bool
