"""Compose frozen benchmark lanes from compatible audited end-to-end reports."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.build_consistency_ensemble import (  # noqa: E402
    normalized_top_k_by_benchmark,
)
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    aggregate,
    canonical_sha256,
    case_results,
    load_jsonl,
    write_json,
    write_jsonl,
)
from scripts.benchmarks.run_verifier_cascade import reconstruct_work  # noqa: E402

RUNNER_VERSION = "1.0.0"


def compatible(primary: dict, derived: dict) -> None:
    for key in ("source_sha256", "sample_per_benchmark", "case_ids_sha256"):
        if primary["identity"][key] != derived["identity"][key]:
            raise SystemExit(f"report identity mismatch: {key}")
    if normalized_top_k_by_benchmark(
        primary["identity"]
    ) != normalized_top_k_by_benchmark(derived["identity"]):
        raise SystemExit("report identity mismatch: top_k_by_benchmark")
    if primary["case_count"] != derived["case_count"]:
        raise SystemExit("report case count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--primary-benchmarks", default="fever")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    derived = json.loads(args.derived.read_text(encoding="utf-8"))
    compatible(primary, derived)
    primary_benchmarks = {
        item.strip() for item in args.primary_benchmarks.split(",") if item.strip()
    }
    benchmark_universe = set(primary["identity"]["benchmarks"])
    if not primary_benchmarks or primary_benchmarks - benchmark_universe:
        raise SystemExit("unknown or empty primary benchmark selection")

    primary_batches = {
        batch["batch_id"]: batch
        for batch in load_jsonl(ROOT / primary["artifacts"]["batch_records_path"])
    }
    derived_batches = load_jsonl(
        ROOT / derived["artifacts"]["batch_records_path"]
    )
    if set(primary_batches) != {batch["batch_id"] for batch in derived_batches}:
        raise SystemExit("report batch universes differ")
    batches = [
        copy.deepcopy(primary_batches[batch["batch_id"]])
        if batch["benchmark_id"] in primary_benchmarks
        else copy.deepcopy(batch)
        for batch in derived_batches
    ]

    works = reconstruct_work(derived)
    work_by_id = {work["case_id"]: work for work in works}
    rows = case_results(batches, work_by_id)
    batch_path = args.out.with_suffix(".batches.jsonl")
    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(batch_path, batches)
    write_jsonl(case_path, rows)

    identity = {
        **derived["identity"],
        "upstream_runner_version": derived["identity"]["runner_version"],
        "upstream_runner_source_sha256": derived["identity"][
            "runner_source_sha256"
        ],
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "task_aware_composition_rule": (
            "primary_report_for_selected_benchmarks_derived_report_elsewhere"
        ),
        "primary_benchmarks": sorted(primary_benchmarks),
        "composition_reports": {
            "primary": {
                "path": args.primary.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.primary),
            },
            "derived": {
                "path": args.derived.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.derived),
            },
        },
    }
    report = copy.deepcopy(derived)
    report.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "metrics": aggregate(rows),
            "fail_closed_case_count": sum(row["fail_closed"] for row in rows),
            "task_aware_composition": {
                "primary_benchmarks": sorted(primary_benchmarks),
                "other_benchmarks_source": "derived",
            },
            "artifacts": {
                "primary_report_path": args.primary.resolve()
                .relative_to(ROOT)
                .as_posix(),
                "primary_report_sha256": sha256_file(args.primary),
                "derived_report_path": args.derived.resolve()
                .relative_to(ROOT)
                .as_posix(),
                "derived_report_sha256": sha256_file(args.derived),
                "batch_records_path": batch_path.resolve()
                .relative_to(ROOT)
                .as_posix(),
                "batch_records_sha256": sha256_file(batch_path),
                "case_records_path": case_path.resolve()
                .relative_to(ROOT)
                .as_posix(),
                "case_records_sha256": sha256_file(case_path),
                "case_results_canonical_sha256": canonical_sha256(rows),
            },
            "limitations": [
                *derived.get("limitations", []),
                "The task-aware composition rule is fixed by benchmark identity and never inspects case-level gold labels during selection.",
                "The selected primary benchmark lanes retain their original frozen batch records and execution provenance.",
            ],
        }
    )
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "case_count": len(rows),
                "metrics": report["metrics"],
                "task_aware_composition": report["task_aware_composition"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
