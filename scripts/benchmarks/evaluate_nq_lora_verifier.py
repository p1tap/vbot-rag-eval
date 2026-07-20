"""Confirm and fully apply the frozen LoRA NQ answerability verifier."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.evaluate_nq_answerability_verifier import (  # noqa: E402
    apply_abstention,
    exact_paired_p_value,
)
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    SENTINEL,
    aggregate,
    canonical_sha256,
    load_jsonl,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.0.0"
FROZEN_NONEMPTY_SCORE_THRESHOLD = 0.02
FROZEN_NONEMPTY_MINUS_EMPTY_MARGIN = -0.9


def paired(selected: list[dict], baseline: list[dict]) -> dict:
    selected_by_id = {row["case_id"]: row for row in selected}
    selected_only = sum(
        selected_by_id[row["case_id"]]["joint_correct"]
        and not row["joint_correct"]
        for row in baseline
    )
    baseline_only = sum(
        row["joint_correct"]
        and not selected_by_id[row["case_id"]]["joint_correct"]
        for row in baseline
    )
    return {
        "selected_only_win_count": selected_only,
        "baseline_only_win_count": baseline_only,
        "net_win_count": selected_only - baseline_only,
        "exact_paired_two_sided_p_value": exact_paired_p_value(
            selected_only, baseline_only
        ),
    }


def apply_policy(
    raw_rows: list[dict], baseline_by_id: dict[str, dict]
) -> tuple[list[dict], int]:
    selected = []
    abstentions = 0
    for raw in raw_rows:
        baseline = baseline_by_id[raw["case_id"]]
        nonempty = [
            candidate["score"]
            for candidate in raw["candidates"]
            if candidate["answer"]
        ]
        empty = [
            candidate["score"]
            for candidate in raw["candidates"]
            if not candidate["answer"]
        ]
        max_nonempty = max(nonempty, default=0.0)
        max_empty = max(empty, default=0.0)
        confident = (
            max_nonempty >= FROZEN_NONEMPTY_SCORE_THRESHOLD
            and max_nonempty - max_empty >= FROZEN_NONEMPTY_MINUS_EMPTY_MARGIN
        )
        baseline_answered = baseline["prediction"].casefold() != SENTINEL.casefold()
        if baseline_answered and not confident:
            selected.append(apply_abstention(baseline, raw))
            abstentions += 1
        else:
            selected.append(dict(baseline))
    return selected, abstentions


def evaluate(
    raw_rows: list[dict], baseline_by_id: dict[str, dict]
) -> tuple[dict, list[dict]]:
    selected, abstentions = apply_policy(raw_rows, baseline_by_id)
    baseline = [baseline_by_id[row["case_id"]] for row in raw_rows]
    return (
        {
            "case_count": len(raw_rows),
            "abstention_count": abstentions,
            "baseline_metrics": aggregate(baseline),
            "selected_metrics": aggregate(selected),
            "paired": paired(selected, baseline),
        },
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-reader-report", type=Path, required=True)
    parser.add_argument("--confirmation-reader-report", type=Path, required=True)
    parser.add_argument("--full-reader-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reader_paths = {
        "development": args.development_reader_report,
        "confirmation": args.confirmation_reader_report,
        "full": args.full_reader_report,
    }
    reader_reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in reader_paths.items()
    }
    adapter_identities = [
        report["identity"]["adapter_artifacts"] for report in reader_reports.values()
    ]
    if not all(identity == adapter_identities[0] for identity in adapter_identities):
        raise SystemExit("reader reports do not use the same adapter artifacts")
    raw = {
        name: load_jsonl(ROOT / report["artifacts"]["records_path"])
        for name, report in reader_reports.items()
    }
    development_ids = {row["case_id"] for row in raw["development"]}
    confirmation_ids = {row["case_id"] for row in raw["confirmation"]}
    full_ids = {row["case_id"] for row in raw["full"]}
    if development_ids & confirmation_ids:
        raise SystemExit("development and confirmation cases overlap")
    if not development_ids | confirmation_ids <= full_ids:
        raise SystemExit("full reader report does not contain the declared windows")

    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    baseline_by_id = {
        row["case_id"]: row
        for row in load_jsonl(
            ROOT / baseline_report["artifacts"]["case_records_path"]
        )
        if row["benchmark_id"] == "natural_questions"
    }
    if full_ids != set(baseline_by_id):
        raise SystemExit("full reader and baseline NQ universes differ")

    development, _ = evaluate(raw["development"], baseline_by_id)
    confirmation, _ = evaluate(raw["confirmation"], baseline_by_id)
    full, full_selected = evaluate(raw["full"], baseline_by_id)
    promotion_ready = (
        confirmation["paired"]["net_win_count"] > 0
        and confirmation["paired"]["exact_paired_two_sided_p_value"] <= 0.05
        and not any(row["fail_closed"] for row in full_selected)
    )
    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, full_selected)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "policy": "keep_baseline_or_abstain_on_lora_reader_confidence",
        "frozen_nonempty_score_threshold": FROZEN_NONEMPTY_SCORE_THRESHOLD,
        "frozen_nonempty_minus_empty_margin": FROZEN_NONEMPTY_MINUS_EMPTY_MARGIN,
        "development_offset": reader_reports["development"]["identity"][
            "sample_offset_per_benchmark"
        ],
        "confirmation_offset": reader_reports["confirmation"]["identity"][
            "sample_offset_per_benchmark"
        ],
        "development_confirmation_overlap_count": 0,
        "case_ids_sha256": canonical_sha256(sorted(full_ids)),
        "adapter_artifacts": adapter_identities[0],
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_application_complete",
        "promotion_ready": promotion_ready,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "inputs": {
            name: {
                "path": path.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in {
                **reader_paths,
                "baseline": args.baseline_report,
            }.items()
        },
        "development": development,
        "confirmation": confirmation,
        "full": full,
        "artifacts": {
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(full_selected),
        },
        "limitations": [
            "LoRA training and threshold selection use only the declared development window.",
            "The disjoint confirmation window controls promotion.",
            "The full score includes development and confirmation cases and is not another holdout.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "promotion_ready": promotion_ready,
                "development": development["paired"],
                "confirmation": confirmation["paired"],
                "confirmation_joint": confirmation["selected_metrics"][
                    "natural_questions"
                ]["joint_correct_rate"],
                "full": full["paired"],
                "full_joint": full["selected_metrics"]["natural_questions"][
                    "joint_correct_rate"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
