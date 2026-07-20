"""Fail-closed audit of an accepted specialist-composed 10,000-case report."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.compose_specialist_full import EXPECTED_COUNTS  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    aggregate,
    canonical_sha256,
    load_jsonl,
    write_json,
)

RUNNER_VERSION = "1.0.0"


def audit(report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = {}
    checks["identity_hash_valid"] = (
        canonical_sha256(report["identity"]) == report["identity_sha256"]
    )
    case_path = ROOT / report["artifacts"]["case_records_path"]
    rows = load_jsonl(case_path)
    checks["case_file_hash_valid"] = (
        sha256_file(case_path) == report["artifacts"]["case_records_sha256"]
    )
    checks["canonical_case_hash_valid"] = (
        canonical_sha256(rows)
        == report["artifacts"]["case_results_canonical_sha256"]
    )
    checks["case_count_exact"] = len(rows) == 10000 == report["case_count"]
    checks["case_ids_unique"] = len({row["case_id"] for row in rows}) == 10000
    actual_counts = {
        benchmark: sum(row["benchmark_id"] == benchmark for row in rows)
        for benchmark in EXPECTED_COUNTS
    }
    checks["benchmark_counts_exact"] = actual_counts == EXPECTED_COUNTS
    checks["zero_fail_closed"] = (
        not any(row["fail_closed"] for row in rows)
        and report["fail_closed_case_count"] == 0
    )
    recomputed_metrics = aggregate(rows)
    checks["metrics_recompute_exactly"] = recomputed_metrics == report["metrics"]
    macro = recomputed_metrics["macro"]["joint_correct_rate"]
    checks["macro_gate_passed"] = macro >= 0.60
    checks["accepted_status_consistent"] = (
        report["status"] == "accepted" and report["promotion_ready"] is True
    )

    baseline_input = report["inputs"]["baseline"]
    baseline_path = ROOT / baseline_input["path"]
    checks["baseline_report_hash_valid"] = (
        sha256_file(baseline_path) == baseline_input["sha256"]
    )
    specialist_checks = {}
    for benchmark, item in report["inputs"]["specialists"].items():
        path = ROOT / item["path"]
        specialist = json.loads(path.read_text(encoding="utf-8"))
        specialist_checks[benchmark] = {
            "report_hash_valid": sha256_file(path) == item["sha256"],
            "promotion_ready": specialist.get("promotion_ready") is True,
            "identity_hash_valid": canonical_sha256(specialist["identity"])
            == specialist["identity_sha256"]
            == item["promotion_identity_sha256"],
        }
    checks["all_specialist_inputs_valid"] = all(
        all(values.values()) for values in specialist_checks.values()
    )
    passed = all(checks.values())
    return {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "auditor": {
            "runner_version": RUNNER_VERSION,
            "runner_source_sha256": sha256_file(Path(__file__)),
        },
        "audited_report": {
            "path": report_path.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(report_path),
        },
        "checks": checks,
        "specialist_checks": specialist_checks,
        "recomputed": {
            "case_count": len(rows),
            "benchmark_counts": actual_counts,
            "strict_macro_joint_correct_rate": macro,
            "metrics": recomputed_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
