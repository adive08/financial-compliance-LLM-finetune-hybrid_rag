"""AdvisoryAgent: propose compliant rewording for each violation."""
from __future__ import annotations

import re

from shared.prompt_format import build_advisory_messages

CATEGORY_FIXES = {
    "misleading_performance": (
        "Past performance is not a reliable indicator of future results. "
        "Capital is at risk and no minimum return is guaranteed."
    ),
    "missing_risk_disclosure": (
        "Capital is at risk. The value of investments may fall as well as rise "
        "and you may get back less than you invested. Read the risk disclosures "
        "before investing."
    ),
    "fee_transparency": (
        "All applicable charges, including entry, ongoing and exit fees, are "
        "itemised in the total cost table provided with this document."
    ),
    "kyc_aml": (
        "The account will be activated only after identity verification and "
        "source-of-funds checks are completed and recorded."
    ),
    "suitability": (
        "This product is recommended only after a documented suitability "
        "assessment confirms that it matches the client's risk tolerance, "
        "investment horizon and objectives."
    ),
    "data_privacy": (
        "Client data is collected and shared only with recorded consent, stored "
        "securely, and retained only for the documented statutory period."
    ),
    "record_keeping": (
        "The advice, its rationale and the client's instructions are recorded in "
        "the client file and retained for the required period."
    ),
    "unknown": (
        "The statement should be reviewed by a compliance officer and corrected "
        "to include the applicable regulatory disclosure."
    ),
}


def template_fix(category: str | None) -> str:
    return CATEGORY_FIXES.get(str(category), CATEGORY_FIXES["unknown"])


def _clean(text: str, original: str) -> str:
    candidate = (text or "").strip()
    candidate = re.sub(r"^```[a-z]*\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(
        r"^(corrected|rewritten)\s*(snippet|text|version)?\s*:\s*",
        "", candidate, flags=re.IGNORECASE,
    )
    candidate = candidate.strip().strip('"').strip("'")
    candidate = re.split(r"\n\s*\n", candidate)[0].strip()
    if not candidate or len(candidate) < 20 or len(candidate) > 600:
        return ""
    normalised = re.sub(r"\s+", " ", candidate).strip().lower()
    source = re.sub(r"\s+", " ", original or "").strip().lower()
    if normalised in source or source in normalised:
        return ""
    numbers = set(re.findall(r"\d+(?:[.,]\d+)*%?", candidate))
    source_numbers = set(re.findall(r"\d+(?:[.,]\d+)*%?", original or ""))
    if not numbers.issubset(source_numbers):
        return ""
    if candidate.lower().startswith(("i ", "as an ai", "sure", "here is")):
        return ""
    return candidate


def advisory_node(state, llm, config=None) -> dict:
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    violations = [dict(item) for item in state.get("violations", [])]

    advisory_cfg = (config or {}).get("advisory", {})
    use_model = bool(advisory_cfg.get("use_model", True))
    max_calls = int(advisory_cfg.get("max_model_calls", 3))
    max_new_tokens = int(advisory_cfg.get("max_new_tokens", 90))

    model_calls = 0
    for violation in violations:
        violation["suggested_fix"] = template_fix(violation.get("category"))
        if not use_model or llm is None or model_calls >= max_calls:
            continue
        messages = build_advisory_messages(
            violation.get("original_snippet", ""),
            violation.get("category", "unknown"),
            violation.get("_regulations"),
        )
        model_calls += 1
        try:
            raw = llm.generate(messages, max_new_tokens=max_new_tokens)
        except Exception as exc:
            errors.append(f"advisory: model rewrite failed: {exc}")
            continue
        cleaned = _clean(raw, violation.get("original_snippet", ""))
        if cleaned:
            violation["suggested_fix"] = cleaned

    trace.append(
        f"advisory: {len(violations)} fix(es), {model_calls} model rewrite(s)"
    )
    return {"violations": violations, "trace": trace, "errors": errors}
