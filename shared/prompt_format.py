"""The single source of truth for how a prompt is assembled.

Both the trainer and the inference server import from here. If the two ever
build prompts differently the model will appear to have "not learned
anything" at inference time, which is a miserable bug to chase. Keep it in
one file.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a financial compliance review assistant. Given a document "
    "snippet, decide whether it shows a compliance issue. Reply with a single "
    "JSON object containing the keys: issue_found (boolean), category "
    "(string), severity (one of none/low/medium/high), and explanation "
    "(one or two sentences). Output JSON only."
)

DEFAULT_INSTRUCTION = (
    "Review the following financial document snippet for compliance issues. "
    "Respond with JSON only."
)

ADVISORY_SYSTEM_PROMPT = (
    "You are a financial compliance editor. Rewrite the given snippet so that "
    "it becomes compliant. Keep the meaning and the product details, remove "
    "guarantees or omissions, and add the required disclosure. Reply with the "
    "corrected snippet only."
)

# Categories the model is trained on. The UI uses this for display labels.
CATEGORIES = {
    "none": "No issue found",
    "misleading_performance": "Misleading performance claim",
    "missing_risk_disclosure": "Missing risk disclosure",
    "fee_transparency": "Fee transparency",
    "kyc_aml": "KYC / AML",
    "suitability": "Suitability",
    "data_privacy": "Data protection",
    "record_keeping": "Record keeping",
}


def format_regulations(regulations, max_chars: int | None = None) -> str:
    """Render retrieved regulations as a compact, citable context block."""
    if not regulations:
        return ""
    lines: list[str] = []
    for item in regulations:
        if isinstance(item, str):
            text = item
            metadata = {}
        else:
            metadata = item.get("metadata", item) if isinstance(item, dict) else {}
            text = item.get("text", "") if isinstance(item, dict) else str(item)
        citation = " ".join(
            str(metadata.get(key) or "") for key in ("title", "section", "authority")
        ).strip()
        prefix = f"[{metadata.get('doc_id', 'regulation')}]"
        if citation:
            prefix += f" {citation}"
        lines.append(f"{prefix}: {str(text).strip()}")
    block = "\n".join(lines)
    if max_chars and len(block) > max_chars:
        block = block[:max_chars].rsplit(" ", 1)[0] + " ..."
    return block


def build_messages(
    input_text: str,
    instruction: str = DEFAULT_INSTRUCTION,
    regulations=None,
    max_regulation_chars: int | None = None,
) -> list[dict]:
    """Return chat messages for a snippet, without the assistant turn.

    With ``regulations=None`` this is byte-for-byte the prompt used during
    training, so the Practical 1 adapter continues to see its familiar format.
    """
    user = f"{instruction}\n\n---\n{input_text.strip()}\n---"
    block = format_regulations(regulations, max_chars=max_regulation_chars)
    if block:
        user += f"\n\nRelevant regulatory excerpts:\n{block}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_advisory_messages(
    input_text: str,
    category: str,
    regulations=None,
    max_regulation_chars: int = 700,
) -> list[dict]:
    """Chat messages asking for a compliant rewrite of a snippet."""
    block = format_regulations(regulations, max_chars=max_regulation_chars)
    user = (
        f"Compliance category: {category}\n\n"
        f"Snippet to rewrite:\n---\n{input_text.strip()}\n---"
    )
    if block:
        user += f"\n\nRelevant regulatory excerpts:\n{block}"
    user += "\n\nReturn only the corrected snippet."
    return [
        {"role": "system", "content": ADVISORY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
