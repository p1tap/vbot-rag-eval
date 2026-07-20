"""Evaluate the frozen NQ DeBERTa answerability-verification policy."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    SENTINEL,
    aggregate,
    answer_scores,
    canonical_sha256,
    load_jsonl,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.0.0"
FROZEN_NONEMPTY_SCORE_THRESHOLD = 0.36


def exact_paired_p_value(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, position)
        for position in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def apply_abstention(row: dict, raw: dict) -> dict:
    updated = dict(row)
    gold_documents = set(raw["gold_document_ids"])
    exact, f1 = answer_scores(SENTINEL, raw["gold"]["answers"])
    evidence_complete = not gold_documents
    updated.update(
        {
            "prediction": SENTINEL,
            "cited_document_ids": [],
            "citation_precision": None,
            "citation_recall": 0.0 if gold_documents else None,
            "answer_exact_match": exact,
            "answer_f1": f1,
            "answerability_correct": raw["gold"]["answerability"]
            == "unanswerable",
            "evidence_complete": evidence_complete,
            "legacy_union_evidence_complete": evidence_complete,
            "joint_correct": bool(exact and evidence_complete),
            "legacy_union_joint_correct": bool(exact and evidence_complete),
        }
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("development", "confirmation", "full"),
        required=True,
    )
    parser.add_argument("--development-reader-report", type=Path)
    parser.add_argument("--confirmation-policy-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "confirmation" and not args.development_reader_report:
        raise SystemExit("confirmation requires --development-reader-report")
    if args.stage == "full" and not args.confirmation_policy_report:
        raise SystemExit("full evaluation requires --confirmation-policy-report")

    reader_report = json.loads(args.reader_report.read_text(encoding="utf-8"))
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    raw_rows = load_jsonl(ROOT / reader_report["artifacts"]["records_path"])
    baseline_all = {
        row["case_id"]: row
        for row in load_jsonl(
            ROOT / baseline_report["artifacts"]["case_records_path"]
        )
    }
    raw_ids = {row["case_id"] for row in raw_rows}
    if len(raw_ids) != len(raw_rows) or not raw_ids <= set(baseline_all):
        raise SystemExit("reader cases are duplicated or absent from baseline")

    development_overlap_count = 0
    development_input = None
    if args.development_reader_report:
        development_report = json.loads(
            args.development_reader_report.read_text(encoding="utf-8")
        )
        development_rows = load_jsonl(
            ROOT / development_report["artifacts"]["records_path"]
        )
        development_ids = {row["case_id"] for row in development_rows}
        development_overlap_count = len(raw_ids & development_ids)
        if development_overlap_count:
            raise SystemExit("confirmation cases overlap development cases")
        development_input = {
            "path": args.development_reader_report.resolve()
            .relative_to(ROOT)
            .as_posix(),
            "sha256": sha256_file(args.development_reader_report),
            "case_ids_sha256": canonical_sha256(sorted(development_ids)),
        }

    confirmation_input = None
    if args.confirmation_policy_report:
        confirmation_report = json.loads(
            args.confirmation_policy_report.read_text(encoding="utf-8")
        )
        if not confirmation_report.get("promotion_ready"):
            raise SystemExit("confirmation policy was not promotion-ready")
        if (
            confirmation_report["identity"]["frozen_nonempty_score_threshold"]
            != FROZEN_NONEMPTY_SCORE_THRESHOLD
        ):
            raise SystemExit("confirmation used a different frozen threshold")
        confirmation_input = {
            "path": args.confirmation_policy_report.resolve()
            .relative_to(ROOT)
            .as_posix(),
            "sha256": sha256_file(args.confirmation_policy_report),
        }

    selected_rows = []
    abstention_count = 0
    for raw in raw_rows:
        baseline = baseline_all[raw["case_id"]]
        if baseline["benchmark_id"] != "natural_questions":
            raise SystemExit("verifier received a non-NQ case")
        nonempty_scores = [
            candidate["score"]
            for candidate in raw["candidates"]
            if candidate["answer"]
        ]
        max_nonempty_score = max(nonempty_scores, default=0.0)
        baseline_answered = baseline["prediction"].casefold() != SENTINEL.casefold()
        if baseline_answered and max_nonempty_score < FROZEN_NONEMPTY_SCORE_THRESHOLD:
            selected_rows.append(apply_abstention(baseline, raw))
            abstention_count += 1
        else:
            selected_rows.append(dict(baseline))

    baseline_rows = [baseline_all[row["case_id"]] for row in raw_rows]
    selected_metrics = aggregate(selected_rows)
    baseline_metrics = aggregate(baseline_rows)
    selected_by_id = {row["case_id"]: row for row in selected_rows}
    baseline_only = sum(
        baseline["joint_correct"]
        and not selected_by_id[baseline["case_id"]]["joint_correct"]
        for baseline in baseline_rows
    )
    selected_only = sum(
        selected_by_id[baseline["case_id"]]["joint_correct"]
        and not baseline["joint_correct"]
        for baseline in baseline_rows
    )
    paired = {
        "selected_only_win_count": selected_only,
        "baseline_only_win_count": baseline_only,
        "net_win_count": selected_only - baseline_only,
        "exact_paired_two_sided_p_value": exact_paired_p_value(
            selected_only, baseline_only
        ),
    }
    delta = (
        selected_metrics["natural_questions"]["joint_correct_rate"]
        - baseline_metrics["natural_questions"]["joint_correct_rate"]
    )
    promotion_ready = (
        args.stage in {"confirmation", "full"}
        and delta > 0
        and paired["exact_paired_two_sided_p_value"] <= 0.05
        and not any(row["fail_closed"] for row in selected_rows)
    )

    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, selected_rows)
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": f"{args.stage}_complete",
        "promotion_ready": promotion_ready,
        "identity": {
            "runner_version": RUNNER_VERSION,
            "runner_source_sha256": sha256_file(Path(__file__)),
            "policy": "keep_baseline_abstention_or_abstain_on_low_reader_score",
            "frozen_nonempty_score_threshold": FROZEN_NONEMPTY_SCORE_THRESHOLD,
            "threshold_selected_on_offset": 200,
            "sample_offset": reader_report["identity"][
                "sample_offset_per_benchmark"
            ],
            "sample_count": len(raw_rows),
            "case_ids_sha256": canonical_sha256(sorted(raw_ids)),
            "development_overlap_count": development_overlap_count,
        },
        "inputs": {
            "reader_report": {
                "path": args.reader_report.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.reader_report),
            },
            "baseline_report": {
                "path": args.baseline_report.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.baseline_report),
            },
            "development_reader_report": development_input,
            "confirmation_policy_report": confirmation_input,
        },
        "abstention_count": abstention_count,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "joint_correct_absolute_delta": delta,
        "paired": paired,
        "artifacts": {
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(selected_rows),
        },
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "baseline_joint": baseline_metrics["natural_questions"][
                    "joint_correct_rate"
                ],
                "selected_joint": selected_metrics["natural_questions"][
                    "joint_correct_rate"
                ],
                "delta": delta,
                "paired": paired,
                "promotion_ready": promotion_ready,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
