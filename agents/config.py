"""Configuration for the multi-agent pipeline."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from rag.config import REPO_ROOT

DEFAULTS: dict[str, Any] = {
    "pipeline": {
        "mode": "agents",
        "max_segments": 6,
        "segment_chars": 1200,
        "segment_overlap": 120,
        "max_violations": 12,
    },
    "research": {
        "top_k": 5,
        "per_segment_k": 3,
        "max_regulations": 12,
        "max_context_chars": 900,
    },
    "advisory": {
        "use_model": True,
        "max_model_calls": 3,
        "max_new_tokens": 90,
    },
    "verifier": {
        "confidence_threshold": 0.55,
        "max_retries": 1,
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


def load_agents_config(path: str | os.PathLike[str] | None = None) -> dict:
    config_path = Path(path) if path else REPO_ROOT / "config" / "agents.yaml"
    loaded: dict = {}
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    return _deep_merge(DEFAULTS, loaded)


def resolve_mode(mode: str | None = None, config: dict | None = None) -> str:
    cfg = config or load_agents_config()
    chosen = mode or os.environ.get("PIPELINE_MODE") or cfg["pipeline"].get("mode", "agents")
    chosen = str(chosen).strip().lower()
    return chosen if chosen in {"agents", "single"} else "agents"
