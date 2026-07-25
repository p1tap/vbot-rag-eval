"""Fit a transparent dense-vs-rerank routing rule on development evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.retrieval_eval import CaseRetrievalResult, aggregate_results  # noqa: E402
from rag.retrievers import AdaptiveRetrievalPolicy  # noqa: E402
from scripts.benchmarks.audit_suite import file_sha256  # noqa: E402


METRICS = ("case_recall_any", "case_recall_all", "evidence_recall", "mrr")


def load_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row.get("id") for row in rows]
    if not rows or any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise ValueError("adaptive retrieval evidence has missing case IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("adaptive retrieval evidence has duplicate case IDs")
    return rows


def metrics_for(rows: list[dict], field: str, k: int) -> dict[str, float]:
    results = []
    for row in rows:
        relevant = row.get("relevant_document_ids")
        ranking = row.get(field)
        if not isinstance(relevant, list) or not isinstance(ranking, list):
            raise ValueError(f"adaptive evidence row is missing {field}")
        results.append(
            CaseRetrievalResult(
                row["id"],
                frozenset(relevant),
                tuple(ranking),
            )
        )
    return aggregate_results(results, [k])[str(k)]


def metrics_for_policy(
    rows: list[dict],
    policy: AdaptiveRetrievalPolicy,
    k: int,
) -> tuple[dict[str, float], int, int]:
    results = []
    reranked = 0
    pairs = 0
    for row in rows:
        route, ranking = policy.choose(row)
        if route == "rerank":
            reranked += 1
            pairs += int(row.get("rerank_candidate_count", len(ranking)))
        results.append(
            CaseRetrievalResult(
                row["id"],
                frozenset(row["relevant_document_ids"]),
                ranking,
            )
        )
    return aggregate_results(results, [k])[str(k)], reranked, pairs


def quantile_thresholds(values: list[float], steps: int) -> list[float]:
    if steps < 2:
        raise ValueError("adaptive threshold grid requires at least two steps")
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("adaptive threshold values must be finite")
    # Use finite sentinels so the frozen policy remains strict RFC 8259 JSON.
    scale = max(1.0, abs(ordered[0]), abs(ordered[-1]))
    thresholds = [ordered[0] - scale * 1e-12]
    for step in range(steps + 1):
        index = round((len(ordered) - 1) * step / steps)
        thresholds.append(ordered[index])
    thresholds.append(ordered[-1])
    return sorted(set(thresholds))


def fit_policy(
    rows: list[dict],
    *,
    benchmark_id: str,
    k: int,
    gain_fraction: float = 0.8,
    tolerance: float = 0.0025,
    grid_steps: int = 20,
) -> tuple[AdaptiveRetrievalPolicy, dict]:
    if not 0 <= gain_fraction <= 1:
        raise ValueError("gain fraction must be between zero and one")
    if tolerance < 0:
        raise ValueError("adaptive tolerance must be non-negative")
    dense = metrics_for(rows, "dense_ranked_document_ids", k)
    full = metrics_for(rows, "reranked_document_ids", k)
    targets = {
        metric: (
            dense[metric] + gain_fraction * (full[metric] - dense[metric])
            if full[metric] >= dense[metric]
            else dense[metric] - tolerance
        )
        for metric in METRICS
    }
    top1_thresholds = quantile_thresholds(
        [row["features"]["dense_top1"] for row in rows],
        grid_steps,
    )
    margin_thresholds = quantile_thresholds(
        [row["features"]["dense_top1_top2_margin"] for row in rows],
        grid_steps,
    )
    best = None
    evaluated = 0
    for top1 in top1_thresholds:
        for margin in margin_thresholds:
            policy = AdaptiveRetrievalPolicy(
                benchmark_id=benchmark_id,
                k=k,
                top1_threshold=top1,
                margin_threshold=margin,
            )
            metrics, reranked, pairs = metrics_for_policy(rows, policy, k)
            evaluated += 1
            if any(metrics[metric] + 1e-12 < targets[metric] for metric in METRICS):
                continue
            rerank_rate = reranked / len(rows)
            aggregate_gain = sum(metrics[metric] - dense[metric] for metric in METRICS)
            objective = (rerank_rate, -aggregate_gain, top1, margin)
            if best is None or objective < best[0]:
                best = (objective, policy, metrics, reranked, pairs)
    if best is None:
        raise ValueError("no adaptive routing policy met the declared targets")
    _, policy, adaptive, reranked, pairs = best
    evidence = {
        "dense": dense,
        "full_rerank": full,
        "targets": targets,
        "adaptive": adaptive,
        "reranked_cases": reranked,
        "rerank_rate": reranked / len(rows),
        "rerank_pairs": pairs,
        "grid_candidates_evaluated": evaluated,
    }
    return policy, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--rerank-report", required=True, type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--gain-fraction", type=float, default=0.8)
    parser.add_argument("--tolerance", type=float, default=0.0025)
    parser.add_argument("--grid-steps", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = load_rows(args.cases)
    rerank_report = json.loads(args.rerank_report.read_text(encoding="utf-8"))
    if rerank_report.get("benchmark_id") != args.benchmark:
        raise SystemExit("rerank report benchmark does not match the policy")
    if rerank_report.get("cases", {}).get("total") != len(rows):
        raise SystemExit("rerank report case count does not match case evidence")
    if (
        rerank_report.get("source", {}).get("selection", {}).get("limit")
        != len(rows)
    ):
        raise SystemExit("rerank report selection does not match case evidence")
    policy, evidence = fit_policy(
        rows,
        benchmark_id=args.benchmark,
        k=args.k,
        gain_fraction=args.gain_fraction,
        tolerance=args.tolerance,
        grid_steps=args.grid_steps,
    )
    payload = {
        "policy_version": policy.policy_version,
        "status": "development_candidate_not_promoted",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": policy.benchmark_id,
        "k": policy.k,
        "routing": {
            "condition": (
                "rerank if dense_top1 <= threshold OR "
                "dense_top1_top2_margin <= threshold; otherwise use dense"
            ),
            "dense_top1_lte": policy.top1_threshold,
            "dense_top1_top2_margin_lte": policy.margin_threshold,
            "invalid_features": "rerank_fail_closed",
        },
        "retrieval_candidate": {
            "report_sha256": file_sha256(args.rerank_report),
            "retriever": rerank_report["retriever"],
            "source": rerank_report["source"],
        },
        "fit": {
            "gain_fraction": args.gain_fraction,
            "tolerance": args.tolerance,
            "grid_steps": args.grid_steps,
            "case_count": len(rows),
            "cases_sha256": file_sha256(args.cases),
            **evidence,
        },
        "limitations": [
            "Thresholds were selected on development evidence and require a disjoint confirmation run.",
            "The public benchmark has influenced prior system work and is not a never-observed product holdout.",
            "The policy is valid only for the named benchmark, retriever configuration, and cutoff.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"adaptive candidate: rerank_rate={evidence['rerank_rate']:.3f} "
        f"k={args.k} cases={len(rows)}"
    )
    print(f"policy -> {args.output}")


if __name__ == "__main__":
    main()
