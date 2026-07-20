"""Select, confirm, and fully apply a provenance-preserving NQ abstention model."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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
QUESTION_TYPES = [
    "what",
    "who",
    "when",
    "where",
    "which",
    "how",
    "is",
    "are",
    "did",
    "does",
    "was",
    "were",
]
FEATURE_NAMES = [
    "max_nonempty_score",
    "max_empty_score",
    "nonempty_minus_empty",
    "top_two_nonempty_gap",
    "nonempty_to_empty_ratio",
    "max_baseline_answer_match_score",
    "baseline_answer_match_count",
    "nonempty_candidate_count",
    "largest_candidate_answer_consensus",
    "baseline_citation_count",
    "baseline_prediction_token_count",
    "baseline_prediction_character_count",
    "question_token_count",
    "baseline_prediction_in_question",
    *[f"question_type_{value}" for value in QUESTION_TYPES],
]
THRESHOLDS = [value / 100 for value in range(5, 96)]


def features(raw: dict, baseline: dict) -> list[float]:
    nonempty = [row for row in raw["candidates"] if row["answer"]]
    empty = [row for row in raw["candidates"] if not row["answer"]]
    scores = sorted((float(row["score"]) for row in nonempty), reverse=True)
    max_nonempty = scores[0] if scores else 0.0
    max_empty = max((float(row["score"]) for row in empty), default=0.0)
    normalized_prediction = normalize_answer(baseline["prediction"])
    matches = [
        float(row["score"])
        for row in nonempty
        if normalize_answer(row["answer"]) == normalized_prediction
    ]
    normalized_candidates = [normalize_answer(row["answer"]) for row in nonempty]
    first_word_match = re.match(r"[a-z0-9]+", raw["query"].casefold())
    first_word = first_word_match.group(0) if first_word_match else ""
    return [
        max_nonempty,
        max_empty,
        max_nonempty - max_empty,
        scores[0] - scores[1] if len(scores) > 1 else max_nonempty,
        max_nonempty / (max_empty + 1e-4),
        max(matches, default=0.0),
        float(len(matches)),
        float(len(nonempty)),
        float(
            max(
                (normalized_candidates.count(value) for value in set(normalized_candidates)),
                default=0,
            )
        ),
        float(len(baseline["cited_document_ids"])),
        float(len(normalized_prediction.split())),
        float(len(baseline["prediction"])),
        float(len(raw["query"].split())),
        float(normalized_prediction in normalize_answer(raw["query"])),
        *[float(first_word == value) for value in QUESTION_TYPES],
    ]


def model_factory(name: str):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            max_features=0.8,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            max_features=0.8,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        )
    raise ValueError(f"unknown model candidate: {name}")


def training_examples(
    raw_rows: list[dict], baseline_by_id: dict[str, dict]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    matrix = []
    labels = []
    case_ids = []
    for raw in raw_rows:
        baseline = baseline_by_id[raw["case_id"]]
        if baseline["prediction"].casefold() == SENTINEL.casefold():
            continue
        keep_correct = bool(baseline["joint_correct"])
        abstain_correct = raw["gold"]["answerability"] == "unanswerable"
        if keep_correct == abstain_correct:
            continue
        matrix.append(features(raw, baseline))
        labels.append(int(abstain_correct))
        case_ids.append(raw["case_id"])
    return (
        np.asarray(matrix, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        case_ids,
    )


def probabilities_for_rows(
    model, raw_rows: list[dict], baseline_by_id: dict[str, dict]
) -> dict[str, float]:
    selected = [
        raw
        for raw in raw_rows
        if baseline_by_id[raw["case_id"]]["prediction"].casefold()
        != SENTINEL.casefold()
    ]
    if not selected:
        return {}
    matrix = np.asarray(
        [features(raw, baseline_by_id[raw["case_id"]]) for raw in selected],
        dtype=np.float64,
    )
    return {
        raw["case_id"]: float(probability)
        for raw, probability in zip(
            selected, model.predict_proba(matrix)[:, 1], strict=True
        )
    }


def apply_policy(
    raw_rows: list[dict],
    baseline_by_id: dict[str, dict],
    probabilities: dict[str, float],
    threshold: float,
) -> list[dict]:
    output = []
    for raw in raw_rows:
        baseline = baseline_by_id[raw["case_id"]]
        if probabilities.get(raw["case_id"], 0.0) >= threshold:
            output.append(apply_abstention(baseline, raw))
        else:
            output.append(dict(baseline))
    return output


def paired(selected: list[dict], baseline_by_id: dict[str, dict]) -> dict:
    wins = sum(
        row["joint_correct"] and not baseline_by_id[row["case_id"]]["joint_correct"]
        for row in selected
    )
    losses = sum(
        baseline_by_id[row["case_id"]]["joint_correct"] and not row["joint_correct"]
        for row in selected
    )
    return {
        "selected_only_win_count": wins,
        "baseline_only_win_count": losses,
        "net_win_count": wins - losses,
        "exact_paired_two_sided_p_value": exact_paired_p_value(wins, losses),
    }


def evaluate(
    raw_rows: list[dict],
    baseline_by_id: dict[str, dict],
    probabilities: dict[str, float],
    threshold: float,
) -> tuple[dict, list[dict]]:
    selected = apply_policy(raw_rows, baseline_by_id, probabilities, threshold)
    baseline = [baseline_by_id[row["case_id"]] for row in raw_rows]
    return (
        {
            "case_count": len(selected),
            "abstention_count": sum(
                row["prediction"].casefold() == SENTINEL.casefold()
                and baseline_by_id[row["case_id"]]["prediction"].casefold()
                != SENTINEL.casefold()
                for row in selected
            ),
            "baseline_metrics": aggregate(baseline),
            "selected_metrics": aggregate(selected),
            "paired": paired(selected, baseline_by_id),
        },
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-reader-report", type=Path, required=True)
    parser.add_argument("--confirmation-reader-report", type=Path, required=True)
    parser.add_argument("--full-reader-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "development": args.development_reader_report,
        "confirmation": args.confirmation_reader_report,
        "full": args.full_reader_report,
    }
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    adapters = [report["identity"]["adapter_artifacts"] for report in reports.values()]
    if not all(adapter == adapters[0] for adapter in adapters):
        raise SystemExit("reader reports do not use the same adapter")
    raw = {
        name: load_jsonl(ROOT / report["artifacts"]["records_path"])
        for name, report in reports.items()
    }
    ids = {name: {row["case_id"] for row in rows} for name, rows in raw.items()}
    if ids["development"] & ids["confirmation"]:
        raise SystemExit("development and confirmation cases overlap")
    if not ids["development"] | ids["confirmation"] <= ids["full"]:
        raise SystemExit("full report does not contain the declared windows")

    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    baseline_by_id = {
        row["case_id"]: row
        for row in load_jsonl(
            ROOT / baseline_report["artifacts"]["case_records_path"]
        )
        if row["benchmark_id"] == "natural_questions"
    }
    if ids["full"] != set(baseline_by_id):
        raise SystemExit("full reader and baseline NQ universes differ")

    matrix, labels, training_ids = training_examples(
        raw["development"], baseline_by_id
    )
    candidate_names = [
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
    ]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_results = []
    for name in candidate_names:
        out_of_fold = np.zeros(len(labels), dtype=np.float64)
        for train_indices, validation_indices in folds.split(matrix, labels):
            candidate = model_factory(name)
            candidate.fit(matrix[train_indices], labels[train_indices])
            out_of_fold[validation_indices] = candidate.predict_proba(
                matrix[validation_indices]
            )[:, 1]
        probabilities = dict(zip(training_ids, out_of_fold, strict=True))
        sweep = []
        for threshold in THRESHOLDS:
            evaluation, _ = evaluate(
                raw["development"], baseline_by_id, probabilities, threshold
            )
            sweep.append(
                {
                    "threshold": threshold,
                    "joint_correct_rate": evaluation["selected_metrics"][
                        "natural_questions"
                    ]["joint_correct_rate"],
                    "abstention_count": evaluation["abstention_count"],
                    **evaluation["paired"],
                }
            )
        chosen = sorted(
            sweep,
            key=lambda row: (
                -row["joint_correct_rate"],
                row["baseline_only_win_count"],
                row["abstention_count"],
                row["threshold"],
            ),
        )[0]
        candidate_results.append(
            {"model_name": name, "selected_oof": chosen, "threshold_sweep": sweep}
        )
    selected_candidate = sorted(
        candidate_results,
        key=lambda row: (
            -row["selected_oof"]["joint_correct_rate"],
            row["selected_oof"]["baseline_only_win_count"],
            row["selected_oof"]["abstention_count"],
            candidate_names.index(row["model_name"]),
        ),
    )[0]
    selected_name = selected_candidate["model_name"]
    threshold = selected_candidate["selected_oof"]["threshold"]
    model = model_factory(selected_name)
    model.fit(matrix, labels)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    probabilities = {
        name: probabilities_for_rows(model, rows, baseline_by_id)
        for name, rows in raw.items()
    }
    evaluations = {}
    selected_rows = {}
    for name in raw:
        evaluations[name], selected_rows[name] = evaluate(
            raw[name], baseline_by_id, probabilities[name], threshold
        )
    promotion_ready = (
        evaluations["confirmation"]["paired"]["net_win_count"] > 0
        and evaluations["confirmation"]["paired"][
            "exact_paired_two_sided_p_value"
        ]
        <= 0.05
        and not any(row["fail_closed"] for row in selected_rows["full"])
    )

    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, selected_rows["full"])
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "policy": "keep_frozen_deepseek_answer_or_abstain_only",
        "feature_names": FEATURE_NAMES,
        "candidate_models": candidate_names,
        "selection": "five-fold out-of-fold development joint correctness",
        "selected_model": selected_name,
        "selected_probability_threshold": threshold,
        "random_state": 42,
        "sklearn_version": sklearn.__version__,
        "training_example_count": len(labels),
        "training_positive_abstain_count": int(labels.sum()),
        "development_confirmation_overlap_count": 0,
        "case_ids_sha256": canonical_sha256(sorted(ids["full"])),
        "adapter_artifacts": adapters[0],
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
            for name, path in {**paths, "baseline": args.baseline_report}.items()
        },
        "model_selection": candidate_results,
        **evaluations,
        "artifacts": {
            "model_path": args.model_out.resolve().relative_to(ROOT).as_posix(),
            "model_sha256": sha256_file(args.model_out),
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(selected_rows["full"]),
        },
        "limitations": [
            "Features, candidate-family choice, and threshold selection use only out-of-fold predictions within the declared development window.",
            "The policy can only preserve a frozen DeepSeek answer or replace it with the fail-closed abstention sentinel; it never creates an answer or citation.",
            "The disjoint confirmation window controls promotion.",
            "The full score includes development and confirmation cases and is not another holdout.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "promotion_ready": promotion_ready,
                "selected_model": selected_name,
                "selected_threshold": threshold,
                "selected_oof": selected_candidate["selected_oof"],
                "confirmation": evaluations["confirmation"],
                "full_joint": evaluations["full"]["selected_metrics"][
                    "natural_questions"
                ]["joint_correct_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
