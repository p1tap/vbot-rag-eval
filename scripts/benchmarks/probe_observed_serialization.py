"""Replay the seven frozen batches that exposed v2.2 serialization repairs."""
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
    fever_work_items,
    run_batch,
    write_json,
)

TARGETS = {
    "natural_questions": {
        410: [
            "natural_questions:dev:-5784633834792163973",
            "natural_questions:dev:4657479707979463328",
            "natural_questions:dev:-8235146511499235503",
            "natural_questions:dev:-8650104853556692961",
        ],
        498: [
            "natural_questions:dev:3912799558244992969",
            "natural_questions:dev:-1667661819585085026",
            "natural_questions:dev:-7168867178928877609",
            "natural_questions:dev:-5232469771534524427",
        ],
        588: [
            "natural_questions:dev:-2347826883564742884",
            "natural_questions:dev:8193934593691670178",
            "natural_questions:dev:-3615252403890478958",
            "natural_questions:dev:-6470661682628568770",
        ],
    },
    "fever": {
        97: [
            "fever:dev:132612",
            "fever:dev:53663",
            "fever:dev:79759",
            "fever:dev:156026",
        ],
        109: [
            "fever:dev:24311",
            "fever:dev:175449",
            "fever:dev:64685",
            "fever:dev:109617",
        ],
        113: [
            "fever:dev:228338",
            "fever:dev:79123",
            "fever:dev:50923",
            "fever:dev:201112",
        ],
        133: [
            "fever:dev:215278",
            "fever:dev:85093",
            "fever:dev:65625",
            "fever:dev:151163",
        ],
    },
}

DEFAULT_OUT = (
    ROOT
    / "reports"
    / "public-benchmarks"
    / "observed-serialization-v220-probe.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    work_by_benchmark = {
        "natural_questions": dense_work_items(
            "natural_questions",
            top_k=8,
            max_document_chars=6000,
            sample_per_benchmark=0,
        ),
        "fever": fever_work_items(
            top_k=8,
            max_document_chars=6000,
            sample_per_benchmark=0,
        ),
    }
    records = []
    outcomes = []
    for benchmark_id, targets in TARGETS.items():
        works = work_by_benchmark[benchmark_id]
        for batch_index, expected_case_ids in targets.items():
            start = batch_index * 4
            batch = works[start : start + 4]
            case_ids = [item["case_id"] for item in batch]
            batch_id = f"{benchmark_id}:{batch_index:05d}"
            if case_ids != expected_case_ids:
                raise SystemExit(
                    f"observed-failure batch identity drifted for {batch_id}: {case_ids}"
                )
            record = run_batch(batch, "qwen3.5-9b-local", batch_id)
            if not record["valid"] or len(record["results"]) != 4:
                raise SystemExit(
                    f"observed serialization probe failed for {batch_id}: "
                    f"{record['errors']}"
                )
            records.append(record)
            outcomes.append(
                {
                    "batch_id": batch_id,
                    "valid": True,
                    "normalization_events": record["normalization_events"],
                    "execution_events": record["execution_events"],
                    "case_slot_retry_count": len(record["case_slot_retry_records"]),
                    "result_count": len(record["results"]),
                }
            )

    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(
            ROOT / "scripts" / "benchmarks" / "run_end_to_end.py"
        ),
        "probe_source_sha256": sha256_file(Path(__file__)),
        "predecessor": {
            "rejection_path": "reports/public-benchmarks/end-to-end-10000-qwen35-local-v211-rejection.json",
            "invalid_natural_questions_batches": 3,
            "merged_fever_serialization_batches": 4,
        },
        "outcome": {
            "target_batch_count": len(outcomes),
            "valid_batch_count": sum(item["valid"] for item in outcomes),
            "semantic_values_inferred": False,
            "batches": outcomes,
        },
        "batch_records": records,
    }
    write_json(args.out, report)
    print(json.dumps(report["outcome"], indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
