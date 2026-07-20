"""Train a reproducible Hotpot bridge-document selector on development data only."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    canonical_sha256,
    dense_work_items,
    load_jsonl,
    normalize_answer,
    write_json,
)

RUNNER_VERSION = "1.0.0"
FEATURE_NAMES = [
    "reciprocal_dense_rank",
    "dense_rank",
    "query_title_token_recall",
    "query_document_token_recall",
    "answer_document_token_recall",
    "normalized_answer_in_document",
    "cited_by_batch1",
    "cited_by_batch4",
    "cited_by_both",
    "answers_agree",
    "title_tokens_referenced_elsewhere",
    "title_token_count",
    "document_token_count",
]
ABSOLUTE_DEVELOPMENT_PRECISION_FLOOR = 0.94
MIN_SELECTOR_PROBABILITY_THRESHOLD = 0.45
THRESHOLDS = [value / 100 for value in range(5, 96)]


def tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def document_features(work: dict, batch1: dict, batch4: dict) -> list[list[float]]:
    query_tokens = tokens(work["query"])
    answer_tokens = tokens(batch1["prediction"])
    normalized_prediction = normalize_answer(batch1["prediction"])
    answers_agree = normalized_prediction == normalize_answer(batch4["prediction"])
    output = []
    for rank, context in enumerate(work["contexts"], start=1):
        title_tokens = tokens(context["title"])
        document_tokens = tokens(context["text"])
        other_tokens = tokens(
            " ".join(
                row["text"]
                for row in work["contexts"]
                if row["document_id"] != context["document_id"]
            )
        )
        document_id = context["document_id"]
        output.append(
            [
                1.0 / rank,
                float(rank),
                len(query_tokens & title_tokens) / max(1, len(query_tokens)),
                len(query_tokens & document_tokens) / max(1, len(query_tokens)),
                len(answer_tokens & document_tokens) / max(1, len(answer_tokens)),
                float(normalized_prediction in normalize_answer(context["text"])),
                float(document_id in batch1["cited_document_ids"]),
                float(document_id in batch4["cited_document_ids"]),
                float(
                    document_id in batch1["cited_document_ids"]
                    and document_id in batch4["cited_document_ids"]
                ),
                float(answers_agree),
                len(title_tokens & other_tokens) / max(1, len(title_tokens)),
                float(len(title_tokens)),
                float(len(document_tokens)),
            ]
        )
    return output


def predict_probabilities(
    features: list[list[float]], coefficients: list[float], intercept: float
) -> list[float]:
    output = []
    for row in features:
        logit = intercept + sum(
            value * coefficient
            for value, coefficient in zip(row, coefficients, strict=True)
        )
        output.append(1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, logit)))))
    return output


def apply_selector(
    work: dict,
    batch1: dict,
    batch4: dict,
    probabilities: list[float],
    threshold: float,
) -> dict:
    selected = set(batch1["cited_document_ids"])
    answers_agree = normalize_answer(batch1["prediction"]) == normalize_answer(
        batch4["prediction"]
    )
    if answers_agree:
        selected.update(batch4["cited_document_ids"])
        selected.update(
            context["document_id"]
            for context, probability in zip(
                work["contexts"], probabilities, strict=True
            )
            if probability >= threshold
        )
    gold = set(work["gold_document_ids"])
    evidence_complete = not gold or gold <= selected
    return {
        "joint_correct": bool(batch1["answer_exact_match"] and evidence_complete),
        "citation_precision": len(selected & gold) / len(selected) if selected else None,
        "citation_recall": len(selected & gold) / len(gold) if gold else None,
        "selected_document_ids": sorted(selected),
        "evidence_complete": evidence_complete,
    }


def threshold_metrics(
    works: list[dict],
    batch1_by_id: dict[str, dict],
    batch4_by_id: dict[str, dict],
    probabilities_by_id: dict[str, list[float]],
    threshold: float,
) -> dict:
    rows = [
        apply_selector(
            work,
            batch1_by_id[work["case_id"]],
            batch4_by_id[work["case_id"]],
            probabilities_by_id[work["case_id"]],
            threshold,
        )
        for work in works
    ]
    precision = [row["citation_precision"] for row in rows if row["citation_precision"] is not None]
    recall = [row["citation_recall"] for row in rows if row["citation_recall"] is not None]
    return {
        "threshold": threshold,
        "joint_correct_rate": sum(row["joint_correct"] for row in rows) / len(rows),
        "citation_precision": sum(precision) / len(precision),
        "citation_recall": sum(recall) / len(recall),
        "added_document_count": sum(
            len(
                set(row["selected_document_ids"])
                - set(batch1_by_id[work["case_id"]]["cited_document_ids"])
                - set(batch4_by_id[work["case_id"]]["cited_document_ids"])
            )
            for row, work in zip(rows, works, strict=True)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch1-report", type=Path, required=True)
    parser.add_argument("--batch4-report", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--offset", type=int, default=800)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.batch1_report, args.batch4_report)
    ]
    if reports[0]["identity"]["batch_size"] != 1 or reports[1]["identity"]["batch_size"] != 4:
        raise SystemExit("expected paired batch-size-1 and batch-size-4 reports")
    for report in reports:
        if report["identity"]["top_k_by_benchmark"]["hotpotqa"] != 8:
            raise SystemExit("selector training requires frozen top-k 8")
    if args.sample < 1 or args.offset < 0:
        raise SystemExit("selector sample must be positive and offset nonnegative")

    case_rows = [
        load_jsonl(ROOT / report["artifacts"]["case_records_path"])
        for report in reports
    ]
    batch1_all_by_id = {row["case_id"]: row for row in case_rows[0]}
    batch4_all_by_id = {row["case_id"]: row for row in case_rows[1]}
    works = dense_work_items(
        "hotpotqa",
        top_k=8,
        max_document_chars=6000,
        sample_per_benchmark=args.sample,
        sample_offset_per_benchmark=args.offset,
    )
    work_ids = {row["case_id"] for row in works}
    if not work_ids <= set(batch1_all_by_id) or not work_ids <= set(
        batch4_all_by_id
    ):
        raise SystemExit("development case universes differ")
    batch1_by_id = {case_id: batch1_all_by_id[case_id] for case_id in work_ids}
    batch4_by_id = {case_id: batch4_all_by_id[case_id] for case_id in work_ids}

    matrix = []
    labels = []
    groups = []
    case_slices = {}
    for work in works:
        start = len(matrix)
        features = document_features(
            work, batch1_by_id[work["case_id"]], batch4_by_id[work["case_id"]]
        )
        matrix.extend(features)
        gold = set(work["gold_document_ids"])
        labels.extend(
            context["document_id"] in gold for context in work["contexts"]
        )
        groups.extend([work["case_id"]] * len(features))
        case_slices[work["case_id"]] = (start, len(matrix))

    matrix_array = np.asarray(matrix, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups)
    out_of_fold = np.zeros(len(label_array), dtype=np.float64)
    for train_indices, validation_indices in GroupKFold(n_splits=5).split(
        matrix_array, label_array, group_array
    ):
        fold_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ).fit(matrix_array[train_indices], label_array[train_indices])
        out_of_fold[validation_indices] = fold_model.predict_proba(
            matrix_array[validation_indices]
        )[:, 1]
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ).fit(matrix_array, label_array)
    coefficients = [float(value) for value in model.coef_[0]]
    intercept = float(model.intercept_[0])
    all_probabilities = out_of_fold.tolist()
    probabilities_by_id = {
        case_id: all_probabilities[start:stop]
        for case_id, (start, stop) in case_slices.items()
    }
    sweep = [
        threshold_metrics(
            works, batch1_by_id, batch4_by_id, probabilities_by_id, threshold
        )
        for threshold in THRESHOLDS
    ]
    agreement_policy_baseline = threshold_metrics(
        works,
        batch1_by_id,
        batch4_by_id,
        probabilities_by_id,
        1.01,
    )
    required_development_precision = ABSOLUTE_DEVELOPMENT_PRECISION_FLOOR
    eligible = [
        row
        for row in sweep
        if row["citation_precision"] >= required_development_precision
        and row["threshold"] >= MIN_SELECTOR_PROBABILITY_THRESHOLD
    ]
    if not eligible:
        raise SystemExit("no selector threshold meets the citation-precision guardrail")
    selected = sorted(
        eligible,
        key=lambda row: (
            -row["joint_correct_rate"],
            -row["citation_recall"],
            row["threshold"],
        ),
    )[0]

    model_artifact = {
        "schema_version": "1.0.0",
        "feature_names": FEATURE_NAMES,
        "coefficients": coefficients,
        "intercept": intercept,
        "selected_probability_threshold": selected["threshold"],
        "absolute_development_precision_floor": ABSOLUTE_DEVELOPMENT_PRECISION_FLOOR,
        "agreement_policy_baseline_precision": agreement_policy_baseline[
            "citation_precision"
        ],
        "required_development_precision": required_development_precision,
        "minimum_selector_probability_threshold": MIN_SELECTOR_PROBABILITY_THRESHOLD,
    }
    write_json(args.model_out, model_artifact)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "model_type": "sklearn_logistic_regression_balanced",
        "sklearn_version": sklearn.__version__,
        "random_state": 42,
        "feature_names": FEATURE_NAMES,
        "training_case_count": len(works),
        "training_sample_offset": args.offset,
        "training_pair_count": len(matrix),
        "positive_pair_count": sum(labels),
        "case_ids_sha256": canonical_sha256(sorted(batch1_by_id)),
        "threshold_selection_folds": 5,
        "threshold_selection": (
            "maximize case-grouped out-of-fold joint, then citation recall, then choose the lowest threshold, "
            "subject to the predeclared development citation-precision floor of 0.94 and a conservative "
            "minimum selector probability of 0.45"
        ),
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "training_complete",
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "inputs": {
            "batch1": {
                "path": args.batch1_report.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.batch1_report),
            },
            "batch4": {
                "path": args.batch4_report.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.batch4_report),
            },
        },
        "selected_development_metrics": selected,
        "agreement_policy_development_baseline": agreement_policy_baseline,
        "required_development_precision": required_development_precision,
        "threshold_sweep": sweep,
        "artifacts": {
            "model_path": args.model_out.resolve().relative_to(ROOT).as_posix(),
            "model_sha256": sha256_file(args.model_out),
            "model_canonical_sha256": canonical_sha256(model_artifact),
        },
        "limitations": [
            f"The selector and threshold use only the declared {len(works)}-case development window.",
            "The selector may add documents only when independent batch-size routes agree on the normalized answer.",
            "A disjoint confirmation window is required before promotion.",
        ],
    }
    write_json(args.report_out, report)
    print(json.dumps({"selected": selected, "identity": identity}, indent=2))


if __name__ == "__main__":
    main()
