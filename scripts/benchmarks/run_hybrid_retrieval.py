"""Score BM25 + pinned E5 reciprocal-rank fusion on bounded public tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EMBED_MODEL, EMBED_MODEL_REVISION  # noqa: E402
from rag.retrieval_eval import (  # noqa: E402
    CaseRetrievalResult,
    aggregate_results,
    documents_from_case,
)
from rag.retrievers import (  # noqa: E402
    BM25Retriever,
    RetrievalHit,
    reciprocal_rank_fusion,
)
from scripts.benchmarks.audit_suite import DEFAULT_INPUTS  # noqa: E402
from scripts.benchmarks.run_dense_retrieval import SUPPORTED, build  # noqa: E402
from scripts.benchmarks.run_retrieval import SCOPES, percentile  # noqa: E402


def run(benchmark_id: str, ks: list[int], rank_constant: int) -> dict:
    root, manifest = build(benchmark_id)
    passages = np.load(root / "passages.npy", mmap_mode="r")
    queries = np.load(root / "queries.npy", mmap_mode="r")
    bm25 = BM25Retriever()
    results: list[CaseRetrievalResult] = []
    latencies_ms: list[float] = []
    passage_offset = 0
    answerable = 0
    with DEFAULT_INPUTS[benchmark_id].open("r", encoding="utf-8") as handle:
        for case_position, line in enumerate(handle):
            case = json.loads(line)
            documents = documents_from_case(case)
            count = len(documents)
            started = perf_counter()
            lexical = bm25.rank(case["query"], documents, count)
            scores = passages[passage_offset : passage_offset + count] @ queries[case_position]
            dense_order = sorted(
                range(count),
                key=lambda index: (-float(scores[index]), documents[index].id),
            )
            dense = [
                RetrievalHit(documents[index], float(scores[index]), rank)
                for rank, index in enumerate(dense_order, start=1)
            ]
            fused = reciprocal_rank_fusion(
                [lexical, dense], k=max(ks), rank_constant=rank_constant
            )
            latencies_ms.append((perf_counter() - started) * 1000)
            relevant = frozenset(
                evidence["document_id"] for evidence in case["supporting_evidence"]
            )
            answerable += bool(relevant)
            results.append(
                CaseRetrievalResult(
                    case["id"],
                    relevant,
                    tuple(hit.document.id for hit in fused),
                )
            )
            passage_offset += count
    if passage_offset != manifest["document_count"]:
        raise ValueError("hybrid scoring document offset drifted")
    return {
        "report_version": "1.0.0",
        "benchmark_id": benchmark_id,
        "retriever": {
            "name": "bm25_e5_rrf",
            "rank_constant": rank_constant,
            "dense_model": EMBED_MODEL,
            "dense_model_revision": EMBED_MODEL_REVISION,
            "lexical": {"name": "bm25", "k1": bm25.k1, "b": bm25.b},
        },
        "retrieval_scope": SCOPES[benchmark_id],
        "source": {
            "sha256": manifest["source_sha256"],
            "passages_sha256": manifest["passages_sha256"],
            "queries_sha256": manifest["queries_sha256"],
        },
        "cases": {
            "total": manifest["case_count"],
            "answerable_scored": answerable,
            "without_gold_evidence": manifest["case_count"] - answerable,
            "candidate_documents_ranked": manifest["document_count"],
        },
        "metrics": aggregate_results(results, ks),
        "ranking_latency_ms_per_query": {
            "median": round(float(np.median(latencies_ms)), 3),
            "p95": round(percentile(latencies_ms, 0.95), 3),
            "note": "Excludes offline E5 embedding build time.",
        },
        "limitations": [
            "This is bounded candidate ranking, not open-corpus retrieval.",
            "Cases without gold evidence are excluded from retrieval quality metrics.",
            "End-to-end generation and answer correctness are not measured here.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=SUPPORTED)
    parser.add_argument("--k", default="1,2,4,5,10")
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ks = sorted({int(value) for value in args.k.split(",")})
    report = run(args.benchmark, ks, args.rank_constant)
    output = args.output or (
        ROOT
        / "artifacts"
        / "benchmarks"
        / args.benchmark
        / "retrieval"
        / f"{args.benchmark}-bm25-e5-rrf.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"hybrid retrieval complete: benchmark={args.benchmark}")


if __name__ == "__main__":
    main()
