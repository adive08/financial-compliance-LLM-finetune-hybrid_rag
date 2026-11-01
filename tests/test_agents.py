"""Multi-agent tests using a stub LLM/retriever (no GPU or adapter needed)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.config import load_agents_config
from agents.graph import SequentialGraph, build_graph
from agents.nodes.advisory import advisory_node, template_fix
from agents.nodes.verifier import verifier_node
from agents.pipeline import run_analysis
from agents.schemas import validate_result
from rag.retriever import RetrievedChunk

METADATA = {
    "doc_id": "rbi-kyc-md-2023",
    "title": "Master Direction - KYC (Revised)",
    "authority": "RBI",
    "section": "4.1",
    "category": "kyc_aml",
    "effective_date": "2023-11-07",
    "last_updated": "2024-06-20",
    "superseded_by": None,
    "superseded_date": None,
    "status": "current",
    "url": None,
    "source": "synthetic",
    "temporal_status": "current",
}


class FakeRetriever:
    ready = True
    reason = None

    def retrieve(self, query, as_of_date=None, categories=None, top_k=5):
        return [RetrievedChunk(
            chunk_id="rbi-kyc-md-2023::0",
            text="Regulated entities must verify identity and record the source of funds "
                 "before activating an account.",
            metadata=dict(METADATA),
            score=0.5,
            dense_score=0.7,
            bm25_score=4.2,
        )]


class MissingIndexRetriever:
    ready = False
    reason = "No RAG index in tests/"

    def retrieve(self, *args, **kwargs):
        raise AssertionError("retrieve should not be called without an index")


class FakeLLM:
    def __init__(self, issue=True, category="kyc_aml", severity="high"):
        self.issue = issue
        self.category = category
        self.severity = severity

    def analyze(self, snippet, regulations=None, max_regulation_chars=None):
        return {
            "issue_found": self.issue,
            "category": self.category if self.issue else "none",
            "severity": self.severity if self.issue else "none",
            "explanation": "Identity verification is deferred until after funding."
                           if self.issue else "No issue apparent.",
            "parsed": True,
        }

    def generate(self, messages, max_new_tokens=90):
        return ("The account will be activated only after identity verification "
                "and source-of-funds checks are complete.")


DOCUMENT = (
    "Northgate Capital guarantees a minimum return of 12% per year. "
    "Identity verification can be completed after the account has been funded."
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_agents_config()

    def test_full_pipeline_produces_structured_result(self):
        result = run_analysis(
            DOCUMENT,
            filename="test.txt",
            as_of_date="2024-06-01",
            llm=FakeLLM(),
            retriever=FakeRetriever(),
            config=self.config,
        )
        self.assertEqual(validate_result(result), [])
        self.assertEqual(result["verdict"], "non_compliant")
        self.assertEqual(len(result["violations"]), 1)
        violation = result["violations"][0]
        self.assertEqual(violation["category"], "kyc_aml")
        self.assertEqual(violation["category_label"], "KYC / AML")
        self.assertEqual(violation["cited_regulation"]["doc_id"], "rbi-kyc-md-2023")
        self.assertIn("activated only after", violation["suggested_fix"])
        self.assertGreater(result["confidence"], 0.5)
        self.assertTrue(any("research" in step for step in result["agent_trace"]))

    def test_compliant_document(self):
        result = run_analysis(
            "The document includes a capital-at-risk warning and a fee table.",
            llm=FakeLLM(issue=False),
            retriever=FakeRetriever(),
            config=self.config,
        )
        self.assertEqual(result["verdict"], "compliant")
        self.assertEqual(result["violations"], [])
        self.assertEqual(validate_result(result), [])

    def test_missing_index_degrades_gracefully(self):
        result = run_analysis(
            DOCUMENT,
            llm=FakeLLM(issue=False),
            retriever=MissingIndexRetriever(),
            config=self.config,
        )
        self.assertEqual(result["verdict"], "needs_review")
        self.assertTrue(result["errors"])
        self.assertTrue(any("index" in error.lower() for error in result["errors"]))

    def test_config_can_disable_model_rewrites(self):
        result = run_analysis(
            DOCUMENT,
            llm=FakeLLM(),
            retriever=FakeRetriever(),
            config=self.config,
            apply_advisory=False,
        )
        self.assertIn("activated only after", result["violations"][0]["suggested_fix"])


class VerifierTests(unittest.TestCase):
    def test_duplicate_violations_are_merged(self):
        violation = {
            "category": "kyc_aml",
            "severity": "high",
            "explanation": "Verification deferred.",
            "original_snippet": DOCUMENT,
            "suggested_fix": "Verify first.",
            "cited_regulation": None,
            "_regulations": [{"chunk_id": "rbi-kyc-md-2023::0",
                              "metadata": METADATA, "score": 0.5}],
        }
        state = {
            "document": {"filename": "test.txt", "as_of_date": "2024-06-01"},
            "segments": [{"index": 0, "text": DOCUMENT}],
            "violations": [dict(violation), dict(violation)],
            "retrieved": [{"chunk_id": "rbi-kyc-md-2023::0", "metadata": METADATA, "score": 0.5}],
            "trace": [],
            "errors": [],
            "retry_count": 0,
        }
        update = verifier_node(state, load_agents_config())
        self.assertEqual(len(update["violations"]), 1)
        self.assertEqual(update["result"]["verdict"], "non_compliant")
        self.assertIsNotNone(update["violations"][0]["cited_regulation"])

    def test_advisory_template_fallback(self):
        state = {
            "violations": [{
                "category": "fee_transparency",
                "original_snippet": "Additional charges may apply.",
                "_regulations": [],
            }],
            "trace": [],
            "errors": [],
        }
        config = load_agents_config()
        config["advisory"]["use_model"] = False
        update = advisory_node(state, llm=None, config=config)
        self.assertIn("itemised", update["violations"][0]["suggested_fix"])
        self.assertIn("itemised", template_fix("fee_transparency"))

    def test_graph_always_exposes_invoke(self):
        graph = build_graph(
            llm=FakeLLM(issue=False),
            retriever=FakeRetriever(),
            config=load_agents_config(),
        )
        self.assertTrue(hasattr(graph, "invoke"))
        self.assertIsInstance(
            build_graph(FakeLLM(), FakeRetriever(), load_agents_config(), use_langgraph=False),
            SequentialGraph,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
