"""Self-contained checks for the Practical 1 pipeline. No pytest needed.

    python tests/test_pipeline.py

Covers the parts that are easy to break and annoying to debug on the demo
machine: checkpoint discovery (which --resume depends on), tolerance of
imperfect model output, and the web routes. Does not require a trained model,
a GPU, or torch.
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: object = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_checkpoint_discovery() -> None:
    from train.train import find_latest_checkpoint

    print("\n-- checkpoint discovery (--resume depends on this) --")
    with tempfile.TemporaryDirectory() as d:
        check("missing dir -> None", find_latest_checkpoint(os.path.join(d, "nope")) is None)
        check("empty dir -> None", find_latest_checkpoint(d) is None)

        for name in ("checkpoint-90", "checkpoint-100", "checkpoint-1000", "checkpoint-200"):
            os.makedirs(os.path.join(d, name))
        got = os.path.basename(find_latest_checkpoint(d))
        # Lexicographic sorting would wrongly pick checkpoint-90 here.
        check("picks numerically highest", got == "checkpoint-1000", got)

        os.makedirs(os.path.join(d, "final"))
        open(os.path.join(d, "checkpoint-9999"), "w").close()   # a file, not a dir
        got = os.path.basename(find_latest_checkpoint(d))
        check("ignores 'final' and stray files", got == "checkpoint-1000", got)


def test_output_parsing() -> None:
    from app.inference import make_snippet, parse_model_output

    print("\n-- model output parsing (a 0.5B model is not always tidy) --")
    r = parse_model_output(
        '{"issue_found": true, "category": "kyc_aml", "severity": "high", "explanation": "x"}')
    check("clean JSON parses", r["parsed"] and r["issue_found"] and r["category"] == "kyc_aml")
    check("friendly category label", r["category_label"] == "KYC / AML", r["category_label"])

    r = parse_model_output('Sure! Here you go:\n{"issue_found": false, "category": "none",'
                           ' "severity": "none", "explanation": "ok"}\nHope that helps.')
    check("JSON embedded in prose parses", r["parsed"] and r["issue_found"] is False)

    check("truncated JSON falls back",
          parse_model_output('{"issue_found": true, "category":')["parsed"] is False)
    check("empty reply falls back",
          "returned nothing" in parse_model_output("")["explanation"])

    r = parse_model_output('{"issue_found": true, "category": "brand_new_thing",'
                           ' "severity": "low", "explanation": "y"}')
    check("unknown category is titlecased", r["category_label"] == "Brand New Thing",
          r["category_label"])

    print("\n-- snippet windowing --")
    check("whitespace normalised", make_snippet("  hello   world  ") == "hello world")
    long_text = "This is a sentence about compliance. " * 200
    snippet = make_snippet(long_text)
    check("truncated to the cap", len(snippet) <= 1200, len(snippet))
    check("cut on a sentence boundary", snippet.endswith("."), repr(snippet[-20:]))


def test_web_routes() -> None:
    print("\n-- web routes (no trained model present) --")
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("SKIP  fastapi not installed (pip install -r requirements-app.txt)")
        return

    from app.main import app
    client = TestClient(app)

    r = client.get("/")
    check("GET / renders", r.status_code == 200, r.status_code)
    check("disclaimer is shown", "not legal, financial or compliance advice" in r.text)

    r = client.get("/health")
    check("GET /health responds", r.status_code == 200 and "adapter_present" in r.json())

    doc = b"Northgate Capital guarantees a minimum return of 12% per year."
    r = client.post("/analyze", files={"document": ("policy.txt", doc, "text/plain")})
    check("text upload is extracted", "Northgate Capital guarantees" in r.text)
    check("missing model is handled, not crashed", r.status_code in (200, 503), r.status_code)

    for label, payload, expected in (
        ("unsupported type", ("a.docx", b"xx", "application/octet-stream"), 400),
        ("empty file", ("empty.txt", b"", "text/plain"), 400),
        ("oversized file", ("big.txt", b"a" * (11 * 1024 * 1024), "text/plain"), 400),
        ("corrupt pdf", ("fake.pdf", b"not a pdf", "application/pdf"), 400),
    ):
        r = client.post("/analyze", files={"document": payload})
        check(f"{label} -> {expected}", r.status_code == expected, r.status_code)


def main() -> int:
    test_checkpoint_discovery()
    test_output_parsing()
    test_web_routes()
    print("\n" + ("ALL PASSED" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
