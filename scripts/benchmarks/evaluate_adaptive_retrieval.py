"""Evaluate a frozen adaptive retrieval policy on disjoint case evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.retrievers import AdaptiveRetrievalPolicy  # noqa: E402
from scripts.benchmarks.audit_suite import file_sha256  # noqa: E402
from scripts.benchmarks.fit_adaptive_retrieval import (  # noqa: E402
    load_rows,
    metrics_for,
    metrics_for_policy,
)
from scripts.benchmarks.run_retrieval import percentile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quality-tolerance", type=float, default=0.005)
    parser.add_argument("--max-rerank-rate", type=float, default=0.75)
    parser.add_argument("--max-p95-ms", type=float, default=0.0)
    args = parser.parse_args()
    policy_payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = AdaptiveRetrievalPolicy.from_dict(policy_payload)
    rows = load_rows(args.cases)
    dense = metrics_for(rows, "dense_ranked_document_ids", policy.k)
    full = metrics_for(rows, "reranked_document_ids", policy.k)
    adaptive, reranked, pairs = metrics_for_policy(rows, policy, policy.k)
    if args.quality_tolerance < 0:
        raise SystemExit("--quality-tolerance must be non-negative")
    if not 0 <= args.max_rerank_rate <= 1:
        raise SystemExit("--max-rerank-rate must be between zero and one")
    routed_latencies = []
    full_pairs = 0
    for row in rows:
        route = policy.route(row.get("features", {}))
        latency = row.get("latency_ms", {})
        if "dense" in latency and "rerank" in latency:
            routed_latencies.append(
                float(latency["dense"])
                + (float(latency["rerank"]) if route == "rerank" else 0.0)
            )
        full_pairs += int(
            row.get(
                "rerank_candidate_count",
                len(row.get("reranked_document_ids", [])),
            )
        )
    rerank_rate = reranked / len(rows)
    quality_passed = all(
        adaptive[metric] >= dense[metric] - args.quality_tolerance
        for metric in dense
    )
    cost_passed = rerank_rate <= args.max_rerank_rate
    latency_summary = None
    latency_passed = True
    if routed_latencies:
        latency_summary = {
            "median": round(float(np.median(routed_latencies)), 3),
            "p95": round(percentile(routed_latencies, 0.95), 3),
            "measurement": "dense plus reranker only on routed cases",
        }
        latency_passed = (
            args.max_p95_ms <= 0
            or latency_summary["p95"] <= args.max_p95_ms
        )
    elif args.max_p95_ms > 0:
        latency_passed = False
    gate_passed = quality_passed and cost_passed and latency_passed
    report = {
        "report_version": "1.0.0",
        "status": (
            "confirmation_gate_passed_not_promoted"
            if gate_passed
            else "confirmation_gate_failed"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": policy.benchmark_id,
        "k": policy.k,
        "source": {
            "case_count": len(rows),
            "cases_sha256": file_sha256(args.cases),
            "policy_sha256": file_sha256(args.policy),
        },
        "routes": {
            "dense_cases": len(rows) - reranked,
            "reranked_cases": reranked,
            "rerank_rate": rerank_rate,
            "rerank_pairs": pairs,
            "full_rerank_pairs": full_pairs,
            "rerank_pair_reduction": (
                1.0 - pairs / full_pairs if full_pairs else None
            ),
        },
        "latency_ms_per_query": latency_summary,
        "metrics": {
            "dense": dense,
            "full_rerank": full,
            "adaptive": adaptive,
            "adaptive_delta_vs_dense": {
                metric: adaptive[metric] - dense[metric] for metric in dense
            },
        },
        "gate": {
            "passed": gate_passed,
            "quality": {
                "passed": quality_passed,
                "tolerance": args.quality_tolerance,
                "rule": "no tracked metric may regress beyond tolerance",
            },
            "compute": {
                "passed": cost_passed,
                "max_rerank_rate": args.max_rerank_rate,
            },
            "latency": {
                "passed": latency_passed,
                "max_p95_ms": args.max_p95_ms or None,
            },
            "boundary": "component confirmation only; end-to-end promotion remains separate",
        },
        "limitations": [
            "This is retrieval-component evidence, not end-to-end answer and citation evaluation.",
            "Promotion remains blocked until the end-to-end guardrails pass.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"adaptive confirmation: rerank_rate={report['routes']['rerank_rate']:.3f} "
        f"cases={len(rows)}"
    )
    print(f"report -> {args.output}")
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
