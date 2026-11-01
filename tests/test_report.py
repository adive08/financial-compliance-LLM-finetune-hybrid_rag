"""PDF report tests (skipped when reportlab is not installed)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.report import build_report

HAS_REPORTLAB = importlib.util.find_spec("reportlab") is not None


def _sample_result() -> dict:
    return {
        "schema_version": "2.0",
        "document": {
            "filename": "sample.txt",
            "as_of_date": "2024-06-01",
            "mode": "agents",
            "characters": 120,
            "segments_analyzed": 1,
            "regulations_retrieved": 1,
        },
        "violations": [{
            "id": "V1",
            "category": "kyc_aml",
            "category_label": "KYC / AML",
            "severity": "high",
            "explanation": "Identity verification deferred until after funding.",
            "original_snippet": "Identity verification can be completed after funding.",
            "suggested_fix": "Verify identity and source of funds before activating the account.",
            "cited_regulation": {
                "doc_id": "rbi-kyc-md-2023",
                "title": "Master Direction - KYC (Revised)",
                "authority": "RBI",
                "section": "4.1",
                "effective_date": "2023-11-07",
                "last_updated": "2024-06-20",
                "status": "current",
                "superseded_by": None,
                "superseded_date": None,
                "temporal_status": "current",
            },
            "confidence": 0.9,
        }],
        "verdict": "non_compliant",
        "confidence": 0.9,
        "summary": "1 potential compliance issue identified.",
        "agent_trace": ["research: 1 regulation", "checker: 1 issue"],
        "errors": [],
    }


@unittest.skipUnless(HAS_REPORTLAB, "reportlab is not installed")
class ReportTests(unittest.TestCase):
    def test_pdf_is_generated(self):
        pdf = build_report(_sample_result())
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 2000)

    def test_empty_findings_report(self):
        result = _sample_result()
        result["violations"] = []
        result["verdict"] = "compliant"
        result["summary"] = "No compliance issue identified."
        pdf = build_report(result)
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
