"""Compare provider-pinned runs of the same judge profile.

The report verifies that both runs saw the same frozen task groups before it
compares validity, decisions, safety errors, latency, and provider provenance.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.run_judge_bakeoff import load_jsonl  # noqa: E402


def profile_bundle(
    summary_path: Path, profile_id: str
) -> tuple[dict, dict, list[dict]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        profile = summary["profiles"][profile_id]
    except KeyError as exc:
        raise SystemExit(f"profile {profile_id!r} is absent from {summary_path}") from exc
    records_path = ROOT / profile["records_path"]
    if sha256_file(records_path) != profile["records_sha256"]:
        raise SystemExit(f"record hash mismatch for {profile_id}: {records_path}")
    return summary, profile, load_jsonl(records_path)


def task_decisions(records: list[dict]) -> tuple[dict[str, str], set[str]]:
    decisions: dict[str, str] = {}
    invalid_tasks: set[str] = set()
    for record in records:
        task_ids = {task["task_id"] for task in record["tasks"]}
        if not record.get("valid"):
            invalid_tasks.update(task_ids)
            continue
        verdicts = {row["task_id"]: row["decision"] for row in record["verdicts"]}
        if set(verdicts) != task_ids:
            raise SystemExit(f"valid group has incomplete verdicts: {record['group_id']}")
        decisions.update(verdicts)
    return decisions, invalid_tasks


def provider_counts(records: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for call in record.get("calls", []):
            counts[call.get("response_provider") or "unreported"] += 1
    return dict(sorted(counts.items()))


def frozen_group_map(records: list[dict]) -> dict[str, tuple[str, str]]:
    return {
        row["group_id"]: (
            row["answer_sha256"],
            json.dumps(row["tasks"], sort_keys=True, separators=(",", ":")),
        )
        for row in records
    }


def compare(
    left_summary: Path,
    left_profile_id: str,
    right_summary: Path,
    right_profile_id: str,
) -> dict:
    left_run, left_profile, left_records = profile_bundle(left_summary, left_profile_id)
    right_run, right_profile, right_records = profile_bundle(right_summary, right_profile_id)
    if left_run["frozen_inputs"] != right_run["frozen_inputs"]:
        raise SystemExit("provider runs do not declare identical frozen inputs")
    left_groups = frozen_group_map(left_records)
    right_groups = frozen_group_map(right_records)
    if left_groups != right_groups:
        raise SystemExit("provider runs do not share an identical frozen group universe")

    left_decisions, left_invalid = task_decisions(left_records)
    right_decisions, right_invalid = task_decisions(right_records)
    comparable = sorted(set(left_decisions) & set(right_decisions))
    disagreements = [
        {
            "task_id": task_id,
            "left": left_decisions[task_id],
            "right": right_decisions[task_id],
        }
        for task_id in comparable
        if left_decisions[task_id] != right_decisions[task_id]
    ]
    agreement = (
        (len(comparable) - len(disagreements)) / len(comparable) if comparable else 0.0
    )

    left_metrics = left_profile["metrics"]["overall"]
    right_metrics = right_profile["metrics"]["overall"]
    left_latency = left_metrics["latency_ms"]
    right_latency = right_metrics["latency_ms"]
    qualified = (
        right_metrics["false_accept_count"] == 0
        and right_metrics["high_critical_false_accept_count"] == 0
        and right_metrics["validity_rate"] >= left_metrics["validity_rate"]
        and agreement >= 0.95
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_input_check": {
            "identical": True,
            "group_count": len(left_groups),
        },
        "left": {
            "summary_path": str(left_summary.resolve().relative_to(ROOT)),
            "profile_id": left_profile_id,
            "profile_sha256": left_profile["profile_sha256"],
            "records_sha256": left_profile["records_sha256"],
            "valid_task_count": left_metrics["valid_task_count"],
            "validity_rate": left_metrics["validity_rate"],
            "human_exact_agreement": left_metrics["exact_agreement"],
            "cohen_kappa": left_metrics["cohen_kappa"],
            "false_accept_count": left_metrics["false_accept_count"],
            "high_critical_false_accept_count": left_metrics[
                "high_critical_false_accept_count"
            ],
            "mean_latency_ms": left_latency["mean"],
            "p95_latency_ms": left_latency["p95"],
            "provider_counts": provider_counts(left_records),
        },
        "right": {
            "summary_path": str(right_summary.resolve().relative_to(ROOT)),
            "profile_id": right_profile_id,
            "profile_sha256": right_profile["profile_sha256"],
            "records_sha256": right_profile["records_sha256"],
            "valid_task_count": right_metrics["valid_task_count"],
            "validity_rate": right_metrics["validity_rate"],
            "human_exact_agreement": right_metrics["exact_agreement"],
            "cohen_kappa": right_metrics["cohen_kappa"],
            "false_accept_count": right_metrics["false_accept_count"],
            "high_critical_false_accept_count": right_metrics[
                "high_critical_false_accept_count"
            ],
            "mean_latency_ms": right_latency["mean"],
            "p95_latency_ms": right_latency["p95"],
            "provider_counts": provider_counts(right_records),
        },
        "paired_comparison": {
            "comparable_task_count": len(comparable),
            "decision_agreement": agreement,
            "decision_disagreement_count": len(disagreements),
            "disagreements": disagreements,
            "left_only_valid_task_count": len(set(left_decisions) - set(right_decisions)),
            "right_only_valid_task_count": len(set(right_decisions) - set(left_decisions)),
            "invalid_in_both_task_count": len(left_invalid & right_invalid),
        },
        "qualification": {
            "qualified_as_ordered_fallback": qualified,
            "criteria": {
                "zero_false_accepts": right_metrics["false_accept_count"] == 0,
                "zero_high_critical_false_accepts": right_metrics[
                    "high_critical_false_accept_count"
                ]
                == 0,
                "validity_not_below_primary": right_metrics["validity_rate"]
                >= left_metrics["validity_rate"],
                "paired_decision_agreement_at_least_95_percent": agreement >= 0.95,
            },
            "routing_order": [left_profile_id, right_profile_id],
            "reason": (
                "The fallback preserves safety and validity with high paired semantic "
                "agreement, but remains second because its latency is higher."
                if qualified
                else "The fallback did not satisfy every predeclared qualification criterion."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-summary", type=Path, required=True)
    parser.add_argument("--left-profile", required=True)
    parser.add_argument("--right-summary", type=Path, required=True)
    parser.add_argument("--right-profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.left_summary.resolve(),
        args.left_profile,
        args.right_summary.resolve(),
        args.right_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["paired_comparison"], indent=2))
    print(json.dumps(report["qualification"], indent=2))


if __name__ == "__main__":
    main()
