"""Summarize matched Vbot index experiments without publishing case records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.audit_suite import file_sha256


METRICS = (
    "case_recall_any",
    "case_recall_all_gold",
    "gold_heading_recall",
    "required_claim_coverage",
    "complete_required_claim_recall",
    "mrr",
)


def load(path: Path, k: int) -> tuple[dict, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report["variants"]["unique_heading_rank"][str(k)]
    return report, {name: metrics[name] for name in METRICS}


def deltas(candidate: dict, baseline: dict) -> dict:
    return {name: candidate[name] - baseline[name] for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--contextual", required=True, type=Path)
    parser.add_argument("--pre-chunk-control", required=True, type=Path)
    parser.add_argument("--late-chunking", required=True, type=Path)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = {}
    metrics = {}
    for name, path in {
        "e5_heading": args.baseline,
        "e5_provenance_context": args.contextual,
        "jina_pre_chunk_control": args.pre_chunk_control,
        "jina_late_chunking": args.late_chunking,
    }.items():
        reports[name], metrics[name] = load(path, args.k)
    dataset_ids = {
        (
            report["dataset"]["sha256"],
            report["dataset"]["case_count"],
        )
        for report in reports.values()
    }
    if len(dataset_ids) != 1:
        raise SystemExit("index experiments do not use the same dataset")
    contextual_delta = deltas(
        metrics["e5_provenance_context"],
        metrics["e5_heading"],
    )
    late_delta = deltas(
        metrics["jina_late_chunking"],
        metrics["jina_pre_chunk_control"],
    )
    contextual_guardrails = (
        "case_recall_all_gold",
        "gold_heading_recall",
        "required_claim_coverage",
        "complete_required_claim_recall",
    )
    payload = {
        "report_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "reviewed_vbot_development_retrieval_only",
        "k": args.k,
        "dataset": {
            "sha256": next(iter(dataset_ids))[0],
            "case_count": next(iter(dataset_ids))[1],
        },
        "evidence": {
            name: {
                "report_sha256": file_sha256(path),
                "index": reports[name]["index"],
            }
            for name, path in {
                "e5_heading": args.baseline,
                "e5_provenance_context": args.contextual,
                "jina_pre_chunk_control": args.pre_chunk_control,
                "jina_late_chunking": args.late_chunking,
            }.items()
        },
        "metrics": metrics,
        "comparisons": {
            "provenance_context_vs_e5_heading": {
                "delta": contextual_delta,
                "decision": (
                    "DEVELOPMENT_CANDIDATE_NEEDS_CONFIRMATION"
                    if all(
                        contextual_delta[name] >= 0
                        for name in contextual_guardrails
                    )
                    else "REJECT"
                ),
            },
            "late_chunking_vs_matched_jina_pre_chunk_control": {
                "delta": late_delta,
                "decision": (
                    "REJECT"
                    if any(late_delta[name] < 0 for name in contextual_guardrails)
                    else "DEVELOPMENT_CANDIDATE_NEEDS_CONFIRMATION"
                ),
            },
        },
        "promotion_boundary": (
            "Development evidence cannot promote an index. A disjoint, "
            "reviewed confirmation partition and end-to-end gate are required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"index experiment summary -> {args.output}")


if __name__ == "__main__":
    main()
