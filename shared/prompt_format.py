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


def build_messages(input_text: str, instruction: str = DEFAULT_INSTRUCTION) -> list[dict]:
    """Return chat messages for a snippet, without the assistant turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n\n---\n{input_text.strip()}\n---"},
    ]
