"""RAG tests: corpus, temporal filtering, chunking, hybrid retrieval."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.build_index import build_index
from rag.chunking import chunk_corpus, split_text
from rag.corpus import load_corpus
from rag.retriever import HybridTemporalRetriever, index_status
from rag.store import BM25
from rag.temporal import is_active, temporal_status, version_chain


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_corpus()

    def test_corpus_shape(self):
        self.assertGreaterEqual(len(self.records), 30)
        categories = {record["category"] for record in self.records}
        self.assertEqual(len(categories), 7)
        for record in self.records:
            for field in ("doc_id", "title", "authority", "section", "text", "effective_date"):
                self.assertTrue(record.get(field), f"{record['doc_id']} missing {field}")
            self.assertIn(record["status"], {"current", "superseded"})

    def test_temporal_filter(self):
        old = next(record for record in self.records if record["doc_id"] == "rbi-kyc-md-2016")
        new = next(record for record in self.records if record["doc_id"] == "rbi-kyc-md-2023")
        self.assertTrue(is_active(old, "2020-06-01"))
        self.assertFalse(is_active(old, "2024-06-01"))
        self.assertFalse(is_active(new, "2020-06-01"))
        self.assertTrue(is_active(new, "2024-06-01"))
        self.assertEqual(temporal_status(old, "2020-06-01"), "current")
        self.assertEqual(temporal_status(old, "2024-06-01"), "superseded")
        self.assertEqual(temporal_status(new, "2020-06-01"), "not_yet_effective")

    def test_version_chain(self):
        chain = version_chain(self.records, "rbi-kyc-md-2016")
        self.assertEqual([record["doc_id"] for record in chain],
                         ["rbi-kyc-md-2016", "rbi-kyc-md-2023"])


class ChunkingTests(unittest.TestCase):
    def test_split_respects_limit_and_overlap(self):
        text = " ".join(f"Sentence number {index} about compliance." for index in range(120))
        chunks = split_text(text, max_chars=300, overlap=60)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 300)
        self.assertIn("compliance", chunks[0])

    def test_short_text_single_chunk(self):
        self.assertEqual(split_text("One short sentence.", 700, 100),
                         ["One short sentence."])

    def test_chunk_ids_unique(self):
        records = load_corpus()
        chunks = chunk_corpus(records, max_chars=300, overlap=50)
        ids = [chunk["chunk_id"] for chunk in chunks]
        self.assertEqual(len(ids), len(set(ids)))


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.storage = Path(self.tempdir.name)
        self.build = build_index(
            storage_dir=self.storage,
            force=True,
            embedder_backend="hashing",
            vector_backend="numpy",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_index_build_and_status(self):
        self.assertEqual(self.build["status"], "built")
        self.assertGreater(self.build["chunks"], 0)
        status = index_status(self.storage)
        self.assertTrue(status["ready"])
        self.assertEqual(status["embedder_backend"], "hashing")
        self.assertEqual(status["vector_backend"], "numpy")

    def test_incomplete_index_is_detected(self):
        (self.storage / "embeddings.npy").unlink()
        status = index_status(self.storage)
        self.assertFalse(status["ready"])
        self.assertIn("incomplete", status["reason"])

    def test_hybrid_query_returns_metadata(self):
        retriever = HybridTemporalRetriever(storage_dir=self.storage)
        self.assertTrue(retriever.ready, retriever.reason)
        hits = retriever.retrieve(
            "identity verification and source of funds before opening an account",
            as_of_date="2020-06-01",
            top_k=5,
        )
        self.assertTrue(hits)
        self.assertTrue(hits[0].metadata.get("effective_date"))
        self.assertIn("doc_id", hits[0].metadata)
        doc_ids = [hit.metadata["doc_id"] for hit in hits]
        self.assertIn("rbi-kyc-md-2016", doc_ids)
        self.assertNotIn("rbi-kyc-md-2023", doc_ids)

    def test_temporal_switch_to_new_version(self):
        retriever = HybridTemporalRetriever(storage_dir=self.storage)
        hits = retriever.retrieve(
            "identity verification and source of funds before opening an account",
            as_of_date="2024-06-01",
            top_k=5,
        )
        doc_ids = [hit.metadata["doc_id"] for hit in hits]
        self.assertIn("rbi-kyc-md-2023", doc_ids)
        self.assertNotIn("rbi-kyc-md-2016", doc_ids)

    def test_category_boost(self):
        retriever = HybridTemporalRetriever(storage_dir=self.storage)
        hits = retriever.retrieve(
            "annual charges and fees", as_of_date="2024-06-01",
            categories=["fee_transparency"], top_k=5,
        )
        self.assertTrue(hits)
        self.assertEqual(hits[0].metadata["category"], "fee_transparency")

    def test_bm25_scores_keyword_match(self):
        chunks = load_corpus()
        docs = [chunk["text"].lower().split() for chunk in chunks]
        index = BM25(docs)
        ranked = index.search("kyc identity verification".split(), limit=3)
        self.assertTrue(ranked)
        self.assertGreater(ranked[0][1], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
