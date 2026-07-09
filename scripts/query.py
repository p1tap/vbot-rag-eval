"""Retrieval CLI — inspect what the retriever returns for a question.

Run: PYTHONNOUSERSITE=1 conda run -n ft python scripts/query.py "your question"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retrieve import retrieve  # noqa: E402

q = " ".join(sys.argv[1:]) or "how does failover work?"
print(f"Q: {q}\n")
for r in retrieve(q):
    print(f"[{r['score']:.3f}] {r['heading_key']}")
    print(f"    {r['text'][:140].replace(chr(10), ' ')}...\n")
