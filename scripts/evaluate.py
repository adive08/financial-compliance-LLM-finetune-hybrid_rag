"""Evaluate the full pipeline on data/eval.jsonl.

    python scripts/evaluate.py --limit 10
    python scripts/evaluate.py --limit 0 --mode agents --as-of 2024-06-01

The script needs the trained adapter and (for citations) a built RAG index.
Results are written to reports/ as JSON and Markdown.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.config import resolve_mode  # noqa: E402
from agents.pipeline import run_analysis  # noqa: E402
from agents.schemas import validate_result  # noqa: E402
from app.inference import ModelUnavailable  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _evaluate_sample(record: dict, index: int, args) -> dict:
    expected = json.loads(record["output"])
    expected_issue = bool(expected.get("issue_found"))
    expected_category = str(expected.get("category", "none"))
    expected_severity = str(expected.get("severity", "none"))

    started = time.perf_counter()
    result = run_analysis(
        record["input"],
        filename=f"eval-{index}.txt",
        as_of_date=args.as_of,
        mode=args.mode,
        apply_advisory=not args.no_advisory,
    )
    latency = time.perf_counter() - started

    violations = result.get("violations") or []
    predicted_issue = bool(violations)
    predicted_categories = {str(item.get("category")) for item in violations}
    matching = [item for item in violations if item.get("category") == expected_category]
    predicted_severities = {str(item.get("severity")) for item in matching}
    citation_categories = {
        str((item.get("cited_regulation") or {}).get("category"))
        for item in matching
        if item.get("cited_regulation")
    }

    if expected_issue:
        verdict_correct = predicted_issue
        category_correct = expected_category in predicted_categories
        severity_correct = expected_severity in predicted_severities
        citation_correct = expected_category in citation_categories
    else:
        verdict_correct = not predicted_issue
        category_correct = not predicted_issue
        severity_correct = not predicted_issue
        citation_correct = not predicted_issue

    problems = validate_result(result)
    return {
        "index": index,
        "expected_issue": expected_issue,
        "expected_category": expected_category,
        "expected_severity": expected_severity,
        "predicted_issue": predicted_issue,
        "predicted_categories": sorted(predicted_categories),
        "verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "violations": len(violations),
        "verdict_correct": verdict_correct,
        "category_correct": category_correct,
        "severity_correct": severity_correct,
        "citation_correct": citation_correct,
        "json_valid": not problems,
        "schema_problems": problems,
        "errors": result.get("errors", []),
        "latency_s": round(latency, 3),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _compute_metrics(samples: list[dict]) -> dict:
    total = len(samples)
    tp = sum(1 for sample in samples if sample["expected_issue"] and sample["predicted_issue"])
    tn = sum(1 for sample in samples if not sample["expected_issue"] and not sample["predicted_issue"])
    fp = sum(1 for sample in samples if not sample["expected_issue"] and sample["predicted_issue"])
    fn = sum(1 for sample in samples if sample["expected_issue"] and not sample["predicted_issue"])
    positives = sum(1 for sample in samples if sample["expected_issue"])
    negatives = total - positives

    per_category: dict[str, dict] = defaultdict(lambda: {"expected": 0, "detected": 0, "correct": 0})
    for sample in samples:
        if not sample["expected_issue"]:
            continue
        key = sample["expected_category"]
        per_category[key]["expected"] += 1
        if sample["predicted_issue"]:
            per_category[key]["detected"] += 1
        if sample["category_correct"]:
            per_category[key]["correct"] += 1

    latencies = [sample["latency_s"] for sample in samples]
    return {
        "samples": total,
        "positives": positives,
        "negatives": negatives,
        "verdict_accuracy": _ratio(sum(s["verdict_correct"] for s in samples), total),
        "category_accuracy": _ratio(
            sum(s["category_correct"] for s in samples if s["expected_issue"]), positives),
        "severity_accuracy": _ratio(
            sum(s["severity_correct"] for s in samples if s["expected_issue"]), positives),
        "citation_accuracy": _ratio(
            sum(s["citation_correct"] for s in samples if s["expected_issue"]), positives),
        "json_validity": _ratio(sum(s["json_valid"] for s in samples), total),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "average_latency_s": round(sum(latencies) / total, 3) if total else 0.0,
        "max_latency_s": round(max(latencies), 3) if latencies else 0.0,
        "per_category": {
            key: {
                **value,
                "recall": _ratio(value["correct"], value["expected"]),
            }
            for key, value in sorted(per_category.items())
        },
    }


def _markdown(report: dict) -> str:
    metrics = report["metrics"]
    lines = [
        "# Pipeline evaluation",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Mode: `{report['config']['mode']}`",
        f"- As-of date: {report['config']['as_of'] or 'today'}",
        f"- Samples: {metrics['samples']} "
        f"({metrics['positives']} with issues, {metrics['negatives']} clean)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Verdict accuracy | {metrics['verdict_accuracy']:.2%} |",
        f"| Category accuracy (positives) | {metrics['category_accuracy']:.2%} |",
        f"| Severity accuracy (positives) | {metrics['severity_accuracy']:.2%} |",
        f"| Citation accuracy (positives) | {metrics['citation_accuracy']:.2%} |",
        f"| JSON validity | {metrics['json_validity']:.2%} |",
        f"| Average latency | {metrics['average_latency_s']:.2f}s |",
        f"| Max latency | {metrics['max_latency_s']:.2f}s |",
        "",
        "## Confusion matrix",
        "",
        "| | Predicted issue | Predicted clean |",
        "|---|---|---|",
        f"| Actual issue | {metrics['confusion']['tp']} | {metrics['confusion']['fn']} |",
        f"| Actual clean | {metrics['confusion']['fp']} | {metrics['confusion']['tn']} |",
        "",
        "## Per category",
        "",
        "| Category | Expected | Detected | Correct | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, values in metrics["per_category"].items():
        lines.append(
            f"| {category} | {values['expected']} | {values['detected']} | "
            f"{values['correct']} | {values['recall']:.2%} |"
        )
    lines += ["", "## Samples", ""]
    for sample in report["samples"]:
        status = "OK" if sample["verdict_correct"] else "WRONG"
        lines.append(
            f"- `{sample['index']:02d}` {status}: expected "
            f"{sample['expected_category']} / {sample['expected_severity']}, "
            f"predicted {', '.join(sample['predicted_categories']) or 'none'} "
            f"({sample['verdict']}, {sample['confidence']}), "
            f"{sample['latency_s']:.2f}s, {sample['violations']} violation(s)"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", default=str(ROOT / "data" / "eval.jsonl"))
    parser.add_argument("--limit", type=int, default=10,
                        help="number of examples (0 = all)")
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="regulation as-of date (default: today)")
    parser.add_argument("--mode", choices=("agents", "single"), default=None)
    parser.add_argument("--no-advisory", action="store_true",
                        help="skip model-generated rewrites for a faster run")
    parser.add_argument("--output-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    records = _read_jsonl(Path(args.eval_file))
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    if not records:
        print("No evaluation records found.", file=sys.stderr)
        return 1

    effective_mode = resolve_mode(args.mode)
    args.mode = effective_mode
    print(f"Evaluating {len(records)} example(s) in {effective_mode} mode...")
    samples: list[dict] = []
    started = time.perf_counter()
    for index, record in enumerate(records):
        try:
            sample = _evaluate_sample(record, index, args)
        except ModelUnavailable as exc:
            print(f"\nModel unavailable: {exc}", file=sys.stderr)
            print("Train the adapter and copy final/ before evaluating.", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"[{index:02d}] failed: {exc}", file=sys.stderr)
            sample = {
                "index": index,
                "expected_issue": bool(json.loads(record["output"]).get("issue_found")),
                "expected_category": json.loads(record["output"]).get("category"),
                "expected_severity": json.loads(record["output"]).get("severity"),
                "predicted_issue": False,
                "predicted_categories": [],
                "verdict": "error",
                "confidence": 0.0,
                "violations": 0,
                "verdict_correct": False,
                "category_correct": False,
                "severity_correct": False,
                "citation_correct": False,
                "json_valid": False,
                "schema_problems": [str(exc)],
                "errors": [str(exc)],
                "latency_s": 0.0,
            }
        samples.append(sample)
        status = "OK" if sample["verdict_correct"] else "WRONG"
        print(
            f"[{index:02d}] {status}: expected {sample['expected_category']} -> "
            f"{', '.join(sample['predicted_categories']) or 'none'} "
            f"({sample['latency_s']:.2f}s)"
        )

    metrics = _compute_metrics(samples)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "mode": args.mode or "configured",
            "as_of": args.as_of,
            "limit": args.limit,
            "advisory": not args.no_advisory,
            "eval_file": args.eval_file,
        },
        "duration_s": round(time.perf_counter() - started, 2),
        "metrics": metrics,
        "samples": samples,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"eval-{stamp}.json"
    md_path = output_dir / f"eval-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")

    print("\nMetrics")
    print(f"  verdict accuracy  : {metrics['verdict_accuracy']:.2%}")
    print(f"  category accuracy : {metrics['category_accuracy']:.2%}")
    print(f"  severity accuracy : {metrics['severity_accuracy']:.2%}")
    print(f"  citation accuracy : {metrics['citation_accuracy']:.2%}")
    print(f"  JSON validity     : {metrics['json_validity']:.2%}")
    print(f"  average latency   : {metrics['average_latency_s']:.2f}s")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
