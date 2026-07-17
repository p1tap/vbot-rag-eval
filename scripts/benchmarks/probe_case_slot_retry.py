"""Replay the frozen HotpotQA batch that exposed case-slot retry need."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    RUNNER_VERSION,
    dense_work_items,
    run_batch,
    write_json,
)

TARGET_BATCH_INDEX = 466
TARGET_CASE_IDS = [
    "hotpotqa:dev_distractor:5a8ee4315542990e94052ba7",
    "hotpotqa:dev_distractor:5ac082535542992a796ded2f",
    "hotpotqa:dev_distractor:5ab6f7415542991d322236f4",
    "hotpotqa:dev_distractor:5ac19ff35542991316484b61",
]
DEFAULT_OUT = (
    ROOT
    / "reports"
    / "public-benchmarks"
    / "case-slot-retry-observed-failure-probe.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    works = dense_work_items(
        "hotpotqa", top_k=8, max_document_chars=6000, sample_per_benchmark=0
    )
    start = TARGET_BATCH_INDEX * 4
    batch = works[start : start + 4]
    case_ids = [item["case_id"] for item in batch]
    if case_ids != TARGET_CASE_IDS:
        raise SystemExit(f"observed-failure batch identity drifted: {case_ids}")
    record = run_batch(batch, "qwen3.5-9b-local", "hotpotqa:00466")
    if not record["valid"] or len(record["results"]) != 4:
        raise SystemExit(f"observed-failure probe did not recover: {record['errors']}")
    if record["execution_events"] not in ([], ["retried_invalid_case_slots"]):
        raise SystemExit("observed-failure probe used an unknown execution path")
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(
            ROOT / "scripts" / "benchmarks" / "run_end_to_end.py"
        ),
        "probe_source_sha256": sha256_file(Path(__file__)),
        "target": {
            "benchmark_id": "hotpotqa",
            "batch_id": "hotpotqa:00466",
            "case_ids": TARGET_CASE_IDS,
            "predecessor_failure": "v2.0.3 returned C4 without citation_ids on all three batch attempts",
        },
        "outcome": {
            "valid": True,
            "execution_events": record["execution_events"],
            "case_slot_retry_count": len(record["case_slot_retry_records"]),
            "result_count": len(record["results"]),
            "semantic_values_inferred": False,
        },
        "batch_record": record,
    }
    write_json(args.out, report)
    print(json.dumps(report["outcome"], indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
