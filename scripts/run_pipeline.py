"""Run the compliance pipeline from the command line.

    python scripts/run_pipeline.py --file docs/sample-document.txt
    python scripts/run_pipeline.py --text "We guarantee a 15% return." --as-of 2024-06-01
    python scripts/run_pipeline.py --file report.txt --json --report out.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.inference import ModelUnavailable  # noqa: E402
from agents.pipeline import run_analysis  # noqa: E402


def _read(path: str) -> str:
    data = Path(path).read_bytes()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"Could not decode {path}")


def _print_human(result: dict) -> None:
    document = result.get("document", {})
    print(f"Document     : {document.get('filename')}")
    print(f"As-of date   : {document.get('as_of_date')}")
    print(f"Verdict      : {result.get('verdict')} "
          f"(confidence {result.get('confidence', 0) * 100:.0f}%)")
    print(f"Summary      : {result.get('summary')}")
    print(f"Segments     : {document.get('segments_analyzed')} | "
          f"regulations retrieved: {document.get('regulations_retrieved')}")
    print()
    violations = result.get("violations", [])
    if not violations:
        print("No violations recorded.")
    for violation in violations:
        print(f"[{violation.get('id')}] {violation.get('category_label')} "
              f"({violation.get('severity')}, confidence "
              f"{violation.get('confidence', 0) * 100:.0f}%)")
        print(f"  Finding   : {violation.get('explanation')}")
        print(f"  Snippet   : {violation.get('original_snippet')}")
        print(f"  Suggested : {violation.get('suggested_fix')}")
        citation = violation.get("cited_regulation")
        if citation:
            print(f"  Citation  : {citation.get('title')} "
                  f"section {citation.get('section')} ({citation.get('authority')}) "
                  f"[{citation.get('doc_id')}]")
        else:
            print("  Citation  : none")
        print()
    if result.get("errors"):
        print("Analysis notes:")
        for error in result["errors"]:
            print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", "-f", help="document to analyse")
    source.add_argument("--text", "-t", help="inline text to analyse")
    parser.add_argument("--as-of", dest="as_of", help="date (YYYY-MM-DD) for regulation validity")
    parser.add_argument("--mode", choices=("agents", "single"), help="pipeline mode")
    parser.add_argument("--json", action="store_true", help="print the full JSON result")
    parser.add_argument("--no-advisory", action="store_true", help="skip model rewrites")
    parser.add_argument("--report", help="also write a PDF report to this path")
    args = parser.parse_args()

    text = args.text if args.text is not None else _read(args.file)
    filename = args.file if args.file else "inline-text.txt"

    try:
        result = run_analysis(
            text,
            filename=filename,
            as_of_date=args.as_of,
            mode=args.mode,
            apply_advisory=not args.no_advisory,
        )
    except ModelUnavailable as exc:
        print(f"Model unavailable: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Could not analyse the document: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_human(result)

    if args.report:
        from app.report import build_report

        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(build_report(result))
        print(f"Wrote PDF report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
