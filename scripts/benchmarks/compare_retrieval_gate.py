"""Compare deterministic public-retrieval evidence at one cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


GUARDRAILS = {"case_recall_all", "evidence_recall"}
METRICS = ("case_recall_any", "case_recall_all", "evidence_recall", "mrr")
DECISIONS = ("PROMOTE", "NEEDS_REVIEW", "REJECT")


def source_sha(report: dict[str, Any]) -> str | None:
    source = report.get("source", {})
    return source.get("sha256") or source.get("cases_sha256")


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    k: int,
    tolerance: float,
) -> tuple[int, list[tuple[str, float, float, float, str]]]:
    identity = (
        "benchmark_id",
        "retrieval_scope",
    )
    for field in identity:
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"retrieval comparison identity mismatch: {field}")
    if source_sha(baseline) != source_sha(candidate):
        raise ValueError("retrieval comparison source checksum mismatch")
    if baseline["cases"]["answerable_scored"] != candidate["cases"]["answerable_scored"]:
        raise ValueError("retrieval comparison answerable case count mismatch")

    baseline_metrics = baseline["metrics"].get(str(k))
    candidate_metrics = candidate["metrics"].get(str(k))
    if not baseline_metrics or not candidate_metrics:
        raise ValueError(f"retrieval comparison is missing k={k}")
    worst = 0
    rows = []
    for metric in METRICS:
        if metric not in baseline_metrics or metric not in candidate_metrics:
            raise ValueError(f"retrieval comparison is missing metric {metric}")
        before = float(baseline_metrics[metric])
        after = float(candidate_metrics[metric])
        delta = after - before
        if delta >= -tolerance:
            verdict = "ok"
        elif metric in GUARDRAILS:
            verdict = "guardrail regressed"
            worst = 2
        elif delta >= -(2 * tolerance):
            verdict = "soft regression"
            worst = max(worst, 1)
        else:
            verdict = "large regression"
            worst = 2
        rows.append((metric, before, after, delta, verdict))
    return worst, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    try:
        decision, rows = compare(
            baseline, candidate, k=args.k, tolerance=args.tolerance
        )
    except (KeyError, TypeError, ValueError) as error:
        print(f"REJECT: {error}")
        raise SystemExit(2) from error
    print(f"{'metric':<20}{'baseline':>11}{'candidate':>12}{'delta':>10}  verdict")
    for metric, before, after, delta, verdict in rows:
        print(f"{metric:<20}{before:>11.4f}{after:>12.4f}{delta:>+10.4f}  {verdict}")
    print(f"\nDECISION: {DECISIONS[decision]}")
    raise SystemExit(decision)


if __name__ == "__main__":
    main()
