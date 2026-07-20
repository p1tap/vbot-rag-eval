"""Compose promoted benchmark specialists into one audited 10,000-case report."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    aggregate,
    canonical_sha256,
    load_jsonl,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.0.0"
EXPECTED_COUNTS = {"hotpotqa": 3000, "natural_questions": 5000, "fever": 2000}


def verified_case_rows(report_path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("promotion_ready") is not True:
        raise ValueError(f"{report_path.name} is not promotion-ready")
    if canonical_sha256(report["identity"]) != report["identity_sha256"]:
        raise ValueError(f"{report_path.name} identity hash mismatch")
    case_path = ROOT / report["artifacts"]["case_records_path"]
    if sha256_file(case_path) != report["artifacts"]["case_records_sha256"]:
        raise ValueError(f"{report_path.name} case artifact hash mismatch")
    rows = load_jsonl(case_path)
    if canonical_sha256(rows) != report["artifacts"]["case_results_canonical_sha256"]:
        raise ValueError(f"{report_path.name} canonical case hash mismatch")
    return report, rows


def compose_rows(
    baseline_rows: list[dict], specialist_rows: dict[str, list[dict]]
) -> list[dict]:
    baseline_by_benchmark = {
        benchmark: {
            row["case_id"] for row in baseline_rows if row["benchmark_id"] == benchmark
        }
        for benchmark in EXPECTED_COUNTS
    }
    selected = []
    for benchmark, expected_count in EXPECTED_COUNTS.items():
        rows = specialist_rows[benchmark]
        if len(rows) != expected_count:
            raise ValueError(
                f"{benchmark} has {len(rows)} cases; expected {expected_count}"
            )
        if any(row["benchmark_id"] != benchmark for row in rows):
            raise ValueError(f"{benchmark} specialist contains another benchmark")
        case_ids = [row["case_id"] for row in rows]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"{benchmark} specialist has duplicate case IDs")
        if set(case_ids) != baseline_by_benchmark[benchmark]:
            raise ValueError(f"{benchmark} specialist universe differs from baseline")
        selected.extend(rows)
    expected_total = sum(EXPECTED_COUNTS.values())
    if len(selected) != expected_total or len(
        {row["case_id"] for row in selected}
    ) != expected_total:
        raise ValueError("composed report does not contain one unique expected universe")
    if any(row["fail_closed"] for row in selected):
        raise ValueError("promoted specialist composition contains fail-closed cases")
    return sorted(selected, key=lambda row: row["case_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--hotpot-report", type=Path, required=True)
    parser.add_argument("--nq-report", type=Path, required=True)
    parser.add_argument("--fever-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    baseline_case_path = ROOT / baseline["artifacts"]["case_records_path"]
    if sha256_file(baseline_case_path) != baseline["artifacts"]["case_records_sha256"]:
        raise SystemExit("baseline case artifact hash mismatch")
    baseline_rows = load_jsonl(baseline_case_path)
    if len(baseline_rows) != 10000:
        raise SystemExit("baseline does not contain 10,000 cases")

    specialist_paths = {
        "hotpotqa": args.hotpot_report,
        "natural_questions": args.nq_report,
        "fever": args.fever_report,
    }
    specialist_reports = {}
    specialist_rows = {}
    for benchmark, path in specialist_paths.items():
        specialist_reports[benchmark], specialist_rows[benchmark] = verified_case_rows(
            path
        )
    selected = compose_rows(baseline_rows, specialist_rows)
    metrics = aggregate(selected)
    strict_macro = metrics["macro"]["joint_correct_rate"]
    accepted = strict_macro >= 0.60

    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, selected)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "policy": "disjoint-confirmation-promoted-specialist-per-benchmark",
        "expected_case_counts": EXPECTED_COUNTS,
        "case_ids_sha256": canonical_sha256(sorted(row["case_id"] for row in selected)),
        "fail_closed_guardrail": "zero_selected_fail_closed_cases_required",
        "promotion_gate": "strict_macro_joint_correct_rate_gte_0.60",
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if accepted else "below_promotion_gate",
        "promotion_ready": accepted,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "case_count": len(selected),
        "fail_closed_case_count": 0,
        "metrics": metrics,
        "inputs": {
            "baseline": {
                "path": args.baseline_report.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.baseline_report),
                "case_records_sha256": baseline["artifacts"][
                    "case_records_sha256"
                ],
            },
            "specialists": {
                benchmark: {
                    "path": path.resolve().relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(path),
                    "promotion_identity_sha256": specialist_reports[benchmark][
                        "identity_sha256"
                    ],
                }
                for benchmark, path in specialist_paths.items()
            },
        },
        "artifacts": {
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(selected),
        },
        "limitations": [
            "Each specialist was selected using its own declared development data and disjoint confirmation gate.",
            "The complete 10,000-case score includes those development and confirmation cases and is a reproducibility result, not another holdout.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": report["case_count"],
                "fail_closed_case_count": 0,
                "strict_macro_joint_correct_rate": strict_macro,
                "benchmarks": {
                    benchmark: metrics[benchmark]["joint_correct_rate"]
                    for benchmark in EXPECTED_COUNTS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
