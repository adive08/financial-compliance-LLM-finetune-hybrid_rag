"""Metric tests for the evaluation script (no model required)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate import _compute_metrics, _markdown


def _sample(expected_issue, predicted_issue, expected_category, predicted_categories,
            correct_category=True):
    return {
        "expected_issue": expected_issue,
        "predicted_issue": predicted_issue,
        "expected_category": expected_category,
        "predicted_categories": predicted_categories,
        "verdict_correct": expected_issue == predicted_issue,
        "category_correct": correct_category,
        "severity_correct": correct_category,
        "citation_correct": correct_category,
        "json_valid": True,
        "latency_s": 1.0,
    }


class MetricTests(unittest.TestCase):
    def test_confusion_and_accuracy(self):
        samples = [
            _sample(True, True, "kyc_aml", ["kyc_aml"]),
            _sample(True, True, "fee_transparency", ["kyc_aml"], correct_category=False),
            _sample(False, False, "none", []),
            _sample(False, True, "none", ["kyc_aml"], correct_category=False),
        ]
        metrics = _compute_metrics(samples)
        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["confusion"], {"tp": 2, "tn": 1, "fp": 1, "fn": 0})
        self.assertAlmostEqual(metrics["verdict_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["category_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["json_validity"], 1.0)
        self.assertEqual(metrics["per_category"]["kyc_aml"]["correct"], 1)
        self.assertEqual(metrics["per_category"]["fee_transparency"]["correct"], 0)

    def test_markdown_contains_summary(self):
        metrics = _compute_metrics([_sample(True, True, "kyc_aml", ["kyc_aml"])])
        report = {
            "generated_at": "now",
            "config": {"mode": "agents", "as_of": None},
            "metrics": metrics,
            "samples": [{
                "index": 0, "verdict_correct": True,
                "expected_category": "kyc_aml", "expected_severity": "high",
                "predicted_categories": ["kyc_aml"], "verdict": "non_compliant",
                "confidence": 0.9, "latency_s": 1.0, "violations": 1,
            }],
        }
        text = _markdown(report)
        self.assertIn("Verdict accuracy", text)
        self.assertIn("kyc_aml", text)
        self.assertIn("Confusion matrix", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
