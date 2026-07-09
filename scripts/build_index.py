"""Build the embedded index and print a chunk/heading inventory.

Run: PYTHONNOUSERSITE=1 conda run -n ft python scripts/build_index.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingest import build_index  # noqa: E402

chunks = build_index()
print(f"built index: {len(chunks)} chunks from {len(set(c['doc_id'] for c in chunks))} docs\n")
by_doc = Counter(c["doc_id"] for c in chunks)
for doc, n in by_doc.items():
    print(f"  {doc}: {n} chunks")
print("\nheading inventory (doc :: heading_key  [windows]):")
seen = Counter(c["heading_key"] for c in chunks)
for hk, n in seen.items():
    print(f"  {hk}  [{n}]")
