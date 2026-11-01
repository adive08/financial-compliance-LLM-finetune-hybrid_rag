"""Build or update the hybrid RAG index.

    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --force --embedder hashing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.build_index import build_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", help="regulations JSONL (default: data/corpus/regulations.jsonl)")
    parser.add_argument("--storage", help="index directory (default: storage/)")
    parser.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    parser.add_argument(
        "--embedder", choices=("auto", "fastembed", "hashing"), default=None,
        help="embedding backend (auto prefers fastembed)",
    )
    parser.add_argument(
        "--vector", choices=("auto", "qdrant", "numpy"), default=None,
        help="vector store backend (auto prefers qdrant local mode)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = build_index(
        corpus_file=args.corpus,
        storage_dir=args.storage,
        force=args.force,
        embedder_backend=args.embedder,
        vector_backend=args.vector,
    )
    print(json.dumps(result, indent=2))
    if result["status"] == "up_to_date":
        print("Index is already current. Use --force to rebuild.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
