"""Train, confirm, and apply the frozen learned NQ abstention verifier."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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
    normalize_answer,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.0.0"
DEVELOPMENT_OFFSET = 1400
DEVELOPMENT_COUNT = 600
CONFIRMATION_OFFSET = 2000
CONFIRMATION_COUNT = 600
FROZEN_THRESHOLD = 0.48
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "min_samples_leaf": 12,
    "max_features": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}
FEATURE_NAMES = [
    "max_nonempty_score",
    "second_nonempty_score",
    "max_empty_score",
    "second_empty_score",
    "max_nonempty_minus_max_empty",
    "nonempty_count",
    "empty_count",
    "mean_nonempty_score",
    "mean_empty_score",
    "baseline_prediction_token_count",
    "baseline_prediction_character_count",
    "reader_prediction_token_count",
    "reader_exactly_agrees",
    "reader_substring_agrees",
    "reader_document_was_cited",
    "baseline_citation_count",
]


def feature_vector(raw: dict, baseline: dict) -> list[float]:
    candidates = raw["candidates"]
    nonempty = sorted(
        [row["score"] for row in candidates if row["answer"]], reverse=True
    )
    empty = sorted(
        [row["score"] for row in candidates if not row["answer"]], reverse=True
    )
    best = max(
        [row for row in candidates if row["answer"]],
        key=lambda row: row["score"],
        default={"answer": "", "document_id": ""},
    )
    nonempty_padded = nonempty + [0.0, 0.0]
    empty_padded = empty + [0.0, 0.0]
    baseline_answer = normalize_answer(baseline["prediction"])
    reader_answer = normalize_answer(best["answer"])
    exact_agreement = bool(reader_answer) and reader_answer == baseline_answer
    substring_agreement = bool(reader_answer) and (
        reader_answer in baseline_answer or baseline_answer in reader_answer
    )
    return [
        nonempty_padded[0],
        nonempty_padded[1],
        empty_padded[0],
        empty_padded[1],
        nonempty_padded[0] - empty_padded[0],
        len(nonempty),
        len(empty),
        sum(nonempty) / max(1, len(nonempty)),
        sum(empty) / max(1, len(empty)),
        len(baseline_answer.split()),
        len(baseline_answer),
        len(reader_answer.split()),
        exact_agreement,
        substring_agreement,
        best["document_id"] in baseline["cited_document_ids"],
        len(baseline["cited_document_ids"]),
    ]


def abstention_utility(raw: dict, baseline: dict) -> int:
    if raw["gold"]["answerability"] == "unanswerable":
        return 1
    if baseline["joint_correct"]:
        return -1
    return 0


def eligible(rows: list[dict], baseline_by_id: dict[str, dict]) -> list[dict]:
    return [
        row
        for row in rows
        if baseline_by_id[row["case_id"]]["prediction"].casefold()
        != SENTINEL.casefold()
    ]


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
    rows: list[dict], baseline_by_id: dict[str, dict], model: RandomForestClassifier
) -> tuple[list[dict], int]:
    candidates = eligible(rows, baseline_by_id)
    scores = model.predict_proba(
        np.asarray(
            [feature_vector(row, baseline_by_id[row["case_id"]]) for row in candidates]
        )
    )[:, 1]
    abstain_ids = {
        row["case_id"]
        for row, score in zip(candidates, scores, strict=True)
        if score >= FROZEN_THRESHOLD
    }
    selected = [
        (
            apply_abstention(baseline_by_id[row["case_id"]], row)
            if row["case_id"] in abstain_ids
            else dict(baseline_by_id[row["case_id"]])
        )
        for row in rows
    ]
    return selected, len(abstain_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reader_report = json.loads(args.reader_report.read_text(encoding="utf-8"))
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if (
        reader_report["identity"]["sample_offset_per_benchmark"] != 0
        or reader_report["case_count"] < CONFIRMATION_OFFSET + CONFIRMATION_COUNT
    ):
        raise SystemExit("reader report does not cover the frozen data windows")
    rows = load_jsonl(ROOT / reader_report["artifacts"]["records_path"])
    baseline_by_id = {
        row["case_id"]: row
        for row in load_jsonl(
            ROOT / baseline_report["artifacts"]["case_records_path"]
        )
        if row["benchmark_id"] == "natural_questions"
    }
    if {row["case_id"] for row in rows} != set(baseline_by_id):
        raise SystemExit("reader and baseline NQ case universes differ")

    development = rows[DEVELOPMENT_OFFSET : DEVELOPMENT_OFFSET + DEVELOPMENT_COUNT]
    confirmation = rows[
        CONFIRMATION_OFFSET : CONFIRMATION_OFFSET + CONFIRMATION_COUNT
    ]
    development_eligible = eligible(development, baseline_by_id)
    nonzero_development = [
        row
        for row in development_eligible
        if abstention_utility(row, baseline_by_id[row["case_id"]])
    ]
    train_x = np.asarray(
        [
            feature_vector(row, baseline_by_id[row["case_id"]])
            for row in nonzero_development
        ]
    )
    train_y = np.asarray(
        [
            abstention_utility(row, baseline_by_id[row["case_id"]]) == 1
            for row in nonzero_development
        ]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = cross_val_predict(
        RandomForestClassifier(**MODEL_PARAMS),
        train_x,
        train_y,
        cv=folds,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    development_selected = [
        row
        for row, score in zip(nonzero_development, oof_scores, strict=True)
        if score >= FROZEN_THRESHOLD
    ]
    development_pair = {
        "selected_only_win_count": sum(
            abstention_utility(row, baseline_by_id[row["case_id"]]) == 1
            for row in development_selected
        ),
        "baseline_only_win_count": sum(
            abstention_utility(row, baseline_by_id[row["case_id"]]) == -1
            for row in development_selected
        ),
    }
    development_pair["net_win_count"] = (
        development_pair["selected_only_win_count"]
        - development_pair["baseline_only_win_count"]
    )
    development_pair["exact_paired_two_sided_p_value"] = exact_paired_p_value(
        development_pair["selected_only_win_count"],
        development_pair["baseline_only_win_count"],
    )

    model = RandomForestClassifier(**MODEL_PARAMS).fit(train_x, train_y)
    confirmation_selected, confirmation_abstentions = apply_policy(
        confirmation, baseline_by_id, model
    )
    full_selected, full_abstentions = apply_policy(rows, baseline_by_id, model)
    confirmation_baseline = [baseline_by_id[row["case_id"]] for row in confirmation]
    full_baseline = [baseline_by_id[row["case_id"]] for row in rows]
    confirmation_pair = paired(confirmation_selected, confirmation_baseline)
    full_pair = paired(full_selected, full_baseline)
    confirmation_metrics = aggregate(confirmation_selected)
    confirmation_baseline_metrics = aggregate(confirmation_baseline)
    full_metrics = aggregate(full_selected)
    full_baseline_metrics = aggregate(full_baseline)
    promotion_ready = (
        confirmation_pair["net_win_count"] > 0
        and confirmation_pair["exact_paired_two_sided_p_value"] <= 0.05
        and not any(row["fail_closed"] for row in confirmation_selected)
    )

    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, full_selected)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "policy": "random_forest_answerability_abstention_verifier",
        "feature_names": FEATURE_NAMES,
        "model_params": MODEL_PARAMS,
        "frozen_probability_threshold": FROZEN_THRESHOLD,
        "development_offset": DEVELOPMENT_OFFSET,
        "development_count": DEVELOPMENT_COUNT,
        "confirmation_offset": CONFIRMATION_OFFSET,
        "confirmation_count": CONFIRMATION_COUNT,
        "development_confirmation_overlap_count": 0,
        "case_ids_sha256": canonical_sha256(sorted(baseline_by_id)),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_application_complete",
        "promotion_ready": promotion_ready,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "inputs": {
            "reader_report_path": args.reader_report.resolve()
            .relative_to(ROOT)
            .as_posix(),
            "reader_report_sha256": sha256_file(args.reader_report),
            "baseline_report_path": args.baseline_report.resolve()
            .relative_to(ROOT)
            .as_posix(),
            "baseline_report_sha256": sha256_file(args.baseline_report),
        },
        "development": {
            "cross_validation": "five_fold_stratified_out_of_fold",
            "nonzero_utility_training_case_count": len(nonzero_development),
            "paired": development_pair,
        },
        "confirmation": {
            "abstention_count": confirmation_abstentions,
            "baseline_metrics": confirmation_baseline_metrics,
            "selected_metrics": confirmation_metrics,
            "paired": confirmation_pair,
        },
        "full": {
            "abstention_count": full_abstentions,
            "baseline_metrics": full_baseline_metrics,
            "selected_metrics": full_metrics,
            "paired": full_pair,
        },
        "artifacts": {
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(full_selected),
        },
        "limitations": [
            "The classifier is trained only on the declared offset-1400 development window.",
            "The offset-2000 confirmation window is disjoint and used only for the promotion decision.",
            "The full score includes the development and confirmation cases and is not an additional holdout.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "promotion_ready": promotion_ready,
                "development_pair": development_pair,
                "confirmation_pair": confirmation_pair,
                "confirmation_joint": confirmation_metrics["natural_questions"][
                    "joint_correct_rate"
                ],
                "full_pair": full_pair,
                "full_joint": full_metrics["natural_questions"][
                    "joint_correct_rate"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
