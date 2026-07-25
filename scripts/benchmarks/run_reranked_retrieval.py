"""Rerank pinned E5 candidates with a pinned local cross-encoder.

This runner emits component evidence only. It does not promote the reranker or
change the online path. Use disjoint ``--offset``/``--limit`` slices for
development and confirmation, then compare against the matching E5 slice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    EMBED_MODEL,
    EMBED_MODEL_REVISION,
    RERANK_MODEL,
    RERANK_MODEL_REVISION,
)
from rag.retrieval_eval import (  # noqa: E402
    CaseRetrievalResult,
    aggregate_results,
    documents_from_case,
)
from rag.retrievers import CrossEncoderScorer, dense_confidence_features  # noqa: E402
from scripts.benchmarks.audit_suite import DEFAULT_INPUTS  # noqa: E402
from scripts.benchmarks.run_dense_retrieval import SUPPORTED, build  # noqa: E402
from scripts.benchmarks.run_retrieval import SCOPES, percentile  # noqa: E402


def dense_order(
    case: dict,
    passage_vectors: np.ndarray,
    query_vector: np.ndarray,
) -> tuple[list[int], list[float]]:
    scores = passage_vectors @ query_vector
    order = sorted(
        range(len(case["documents"])),
        key=lambda index: (-float(scores[index]), case["documents"][index]["id"]),
    )
    return order, [float(scores[index]) for index in order]


def run(
    benchmark_id: str,
    ks: list[int],
    *,
    candidate_depth: int,
    offset: int = 0,
    limit: int = 0,
    device: str = "auto",
    batch_size: int = 32,
    precision: str = "float32",
    scorer=None,
) -> tuple[dict, list[dict]]:
    if not ks or min(ks) < 1:
        raise ValueError("retrieval cutoffs must be positive")
    if candidate_depth < max(ks):
        raise ValueError("candidate depth must be at least the largest requested k")
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")
    root, manifest = build(benchmark_id)
    passages = np.load(root / "passages.npy", mmap_mode="r")
    queries = np.load(root / "queries.npy", mmap_mode="r")
    scorer = scorer or CrossEncoderScorer(
        device=device,
        batch_size=batch_size,
        precision=precision,
    )
    results: list[CaseRetrievalResult] = []
    case_rows: list[dict] = []
    latencies_ms: list[float] = []
    passage_offset = 0
    answerable = 0
    selected = 0
    pairs_scored = 0
    started_run = perf_counter()
    with DEFAULT_INPUTS[benchmark_id].open("r", encoding="utf-8") as handle:
        for case_position, line in enumerate(handle):
            case = json.loads(line)
            count = len(case["documents"])
            current_passage_offset = passage_offset
            passage_offset += count
            if case_position < offset:
                continue
            if limit and selected >= limit:
                break
            selected += 1
            documents = documents_from_case(case)
            started = perf_counter()
            dense_started = perf_counter()
            first_stage_order, first_stage_scores = dense_order(
                case,
                passages[current_passage_offset : current_passage_offset + count],
                queries[case_position],
            )
            dense_elapsed_ms = (perf_counter() - dense_started) * 1000
            candidate_indexes = first_stage_order[: min(candidate_depth, count)]
            candidates = [documents[index] for index in candidate_indexes]
            rerank_started = perf_counter()
            rerank_scores = scorer.score(case["query"], candidates)
            rerank_elapsed_ms = (perf_counter() - rerank_started) * 1000
            if len(rerank_scores) != len(candidates):
                raise ValueError("reranker returned a mismatched score count")
            reranked_positions = sorted(
                range(len(candidates)),
                key=lambda index: (
                    -float(rerank_scores[index]),
                    index,
                    candidates[index].id,
                ),
            )
            reranked_ids = tuple(
                candidates[index].id for index in reranked_positions[: max(ks)]
            )
            latencies_ms.append((perf_counter() - started) * 1000)
            pairs_scored += len(candidates)
            relevant = frozenset(
                evidence["document_id"] for evidence in case["supporting_evidence"]
            )
            answerable += bool(relevant)
            results.append(CaseRetrievalResult(case["id"], relevant, reranked_ids))
            case_rows.append(
                {
                    "id": case["id"],
                    "relevant_document_ids": sorted(relevant),
                    "dense_ranked_document_ids": [
                        documents[index].id
                        for index in first_stage_order[: max(ks)]
                    ],
                    "reranked_document_ids": list(reranked_ids),
                    "rerank_candidate_count": len(candidates),
                    "latency_ms": {
                        "dense": dense_elapsed_ms,
                        "rerank": rerank_elapsed_ms,
                        "total": dense_elapsed_ms + rerank_elapsed_ms,
                    },
                    "features": dense_confidence_features(first_stage_scores),
                }
            )
    if not selected:
        raise ValueError("selected benchmark slice is empty")
    elapsed = perf_counter() - started_run
    report = {
        "report_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": benchmark_id,
        "retriever": {
            "name": "e5_small_v2_cross_encoder",
            "first_stage": {
                "name": "e5_small_v2",
                "model": EMBED_MODEL,
                "model_revision": EMBED_MODEL_REVISION,
            },
            "reranker": {
                "name": getattr(scorer, "name", scorer.__class__.__name__),
                "model": getattr(scorer, "model_name", RERANK_MODEL),
                "model_revision": getattr(
                    scorer, "model_revision", RERANK_MODEL_REVISION
                ),
                "candidate_depth": candidate_depth,
                "max_tokens": getattr(scorer, "max_length", None),
                "batch_size": getattr(scorer, "batch_size", None),
                "device": getattr(scorer, "device", device),
                "precision": getattr(scorer, "precision", precision),
            },
        },
        "retrieval_scope": SCOPES[benchmark_id],
        "source": {
            "sha256": manifest["source_sha256"],
            "passages_sha256": manifest["passages_sha256"],
            "queries_sha256": manifest["queries_sha256"],
            "selection": {"offset": offset, "limit": selected},
        },
        "cases": {
            "total": selected,
            "answerable_scored": answerable,
            "without_gold_evidence": selected - answerable,
            "pairs_scored": pairs_scored,
        },
        "metrics": aggregate_results(results, ks),
        "latency_ms_per_query": {
            "median": round(float(np.median(latencies_ms)), 3),
            "p95": round(percentile(latencies_ms, 0.95), 3),
            "elapsed_seconds": round(elapsed, 3),
            "note": (
                "Includes dense scoring and local cross-encoder inference; "
                "excludes offline E5 embedding build time."
            ),
        },
        "limitations": [
            "This is bounded candidate reranking, not open-corpus retrieval.",
            "The public suite influenced earlier development; a disjoint experiment slice is not a never-observed product holdout.",
            "End-to-end answer and citation correctness require a separate generation evaluation.",
        ],
    }
    return report, case_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=SUPPORTED)
    parser.add_argument("--k", default="1,2,4,5,10")
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--precision",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases-output", type=Path)
    args = parser.parse_args()
    ks = sorted({int(value) for value in args.k.split(",") if value.strip()})
    report, case_rows = run(
        args.benchmark,
        ks,
        candidate_depth=args.candidate_depth,
        offset=args.offset,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch_size,
        precision=args.precision,
    )
    output = args.output or (
        ROOT
        / "artifacts"
        / "benchmarks"
        / args.benchmark
        / "retrieval"
        / f"{args.benchmark}-e5-cross-encoder.json"
    )
    cases_output = args.cases_output or output.with_suffix(".cases.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cases_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in case_rows) + "\n",
        encoding="utf-8",
    )
    print(
        f"reranked retrieval complete: benchmark={args.benchmark} "
        f"cases={report['cases']['total']} "
        f"pairs={report['cases']['pairs_scored']}"
    )
    print(f"report -> {output}")
    print(f"case evidence -> {cases_output}")


if __name__ == "__main__":
    main()
