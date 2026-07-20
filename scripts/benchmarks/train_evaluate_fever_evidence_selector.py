"""Train and audit a provenance-preserving FEVER evidence selector."""
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
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.evaluate_nq_answerability_verifier import (  # noqa: E402
    exact_paired_p_value,
)
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    aggregate,
    canonical_sha256,
    fever_work_items,
    load_jsonl,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.0.0"
FROZEN_LABEL_THRESHOLD = 0.75
DEVELOPMENT_PRECISION_SAFETY_MARGIN = 0.01
THRESHOLDS = [value / 100 for value in range(5, 96)]
FEATURE_NAMES = [
    "reciprocal_retrieval_rank",
    "retrieval_rank",
    "entailment_probability",
    "contradiction_probability",
    "neutral_probability",
    "selected_relation_probability",
    "maximum_case_relation_probability",
    "relative_selected_relation_probability",
    "cited_by_baseline",
    "label_agrees_with_baseline",
    "predicted_supports",
    "predicted_refutes",
    "claim_title_token_recall",
    "claim_document_token_recall",
    "title_token_count",
    "document_token_count",
]


def model_factory(name: str):
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=42
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            max_features=0.8,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
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


def tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def local_label(raw: dict) -> tuple[str, str | None, float]:
    candidates = raw["candidates"]
    if not candidates:
        return "not_enough_info", None, 0.0
    entailment = max(float(row["entailment"]) for row in candidates)
    contradiction = max(float(row["contradiction"]) for row in candidates)
    top = max(entailment, contradiction)
    if top < FROZEN_LABEL_THRESHOLD:
        return "not_enough_info", None, top
    if entailment >= contradiction:
        return "supports", "entailment", top
    return "refutes", "contradiction", top


def candidate_features(
    raw: dict, work: dict, baseline: dict
) -> tuple[list[list[float]], list[str]]:
    prediction, score_key, top = local_label(raw)
    query_tokens = tokens(raw["query"])
    context_by_id = {row["document_id"]: row for row in work["contexts"]}
    matrix = []
    document_ids = []
    for rank, candidate in enumerate(raw["candidates"], start=1):
        context = context_by_id[candidate["document_id"]]
        title_tokens = tokens(context["title"])
        document_tokens = tokens(context["text"])
        relation = (
            float(candidate[score_key])
            if score_key
            else max(
                float(candidate["entailment"]),
                float(candidate["contradiction"]),
            )
        )
        matrix.append(
            [
                1.0 / rank,
                float(rank),
                float(candidate["entailment"]),
                float(candidate["contradiction"]),
                float(candidate["neutral"]),
                relation,
                top,
                relation / max(top, 1e-9),
                float(candidate["document_id"] in baseline["cited_document_ids"]),
                float(prediction == baseline["prediction"]),
                float(prediction == "supports"),
                float(prediction == "refutes"),
                len(query_tokens & title_tokens) / max(1, len(query_tokens)),
                len(query_tokens & document_tokens) / max(1, len(query_tokens)),
                float(len(title_tokens)),
                float(len(document_tokens)),
            ]
        )
        document_ids.append(candidate["document_id"])
    return matrix, document_ids


def reconstruct_works(report: dict) -> dict[str, dict]:
    identity = report["identity"]
    return {
        row["case_id"]: row
        for row in fever_work_items(
            top_k=identity["top_k"],
            max_document_chars=identity["max_document_chars"],
            sample_per_benchmark=identity["sample_per_benchmark"],
            sample_offset_per_benchmark=identity["sample_offset_per_benchmark"],
        )
    }


def predict_cases(
    model,
    raw_rows: list[dict],
    works: dict[str, dict],
    baseline_by_id: dict[str, dict],
) -> dict[str, tuple[list[str], list[float]]]:
    output = {}
    for raw in raw_rows:
        matrix, document_ids = candidate_features(
            raw, works[raw["case_id"]], baseline_by_id[raw["case_id"]]
        )
        probabilities = (
            model.predict_proba(np.asarray(matrix, dtype=np.float64))[:, 1].tolist()
            if matrix
            else []
        )
        output[raw["case_id"]] = (document_ids, probabilities)
    return output


def score_case(
    raw: dict,
    baseline: dict,
    prediction_scores: tuple[list[str], list[float]],
    threshold: float,
) -> dict:
    prediction, _, _ = local_label(raw)
    cited = set()
    document_ids, probabilities = prediction_scores
    if prediction != "not_enough_info":
        if prediction == baseline["prediction"]:
            cited.update(baseline["cited_document_ids"])
        cited.update(
            document_id
            for document_id, probability in zip(
                document_ids, probabilities, strict=True
            )
            if probability >= threshold
        )
        if not cited and document_ids:
            cited.add(document_ids[int(np.argmax(probabilities))])
    cited_documents = sorted(cited)
    cited_set = set(cited_documents)
    gold_documents = set(raw["gold_document_ids"])
    label_correct = prediction == raw["gold"]["label"]
    if prediction == "not_enough_info":
        evidence_complete = (
            raw["gold"]["label"] == "not_enough_info" and not cited_documents
        )
    else:
        evidence_complete = any(
            set(evidence_set) <= cited_set for evidence_set in raw["gold_evidence_sets"]
        )
    return {
        "case_id": raw["case_id"],
        "benchmark_id": "fever",
        "prediction": prediction,
        "fail_closed": False,
        "gold": raw["gold"],
        "retrieval_any_gold": bool(
            gold_documents & set(raw["retrieved_document_ids"])
        )
        if gold_documents
        else None,
        "retrieval_complete_gold": gold_documents
        <= set(raw["retrieved_document_ids"])
        if gold_documents
        else None,
        "citation_precision": len(cited_set & gold_documents) / len(cited_set)
        if cited_set
        else None,
        "citation_recall": len(cited_set & gold_documents) / len(gold_documents)
        if gold_documents
        else None,
        "cited_document_ids": cited_documents,
        "label_correct": label_correct,
        "evidence_complete": evidence_complete,
        "joint_correct": bool(label_correct and evidence_complete),
    }


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
    predictions: dict[str, tuple[list[str], list[float]]],
    threshold: float,
) -> tuple[dict, list[dict]]:
    selected = [
        score_case(
            raw,
            baseline_by_id[raw["case_id"]],
            predictions[raw["case_id"]],
            threshold,
        )
        for raw in raw_rows
    ]
    baseline = [baseline_by_id[row["case_id"]] for row in raw_rows]
    return (
        {
            "case_count": len(selected),
            "baseline_metrics": aggregate(baseline),
            "selected_metrics": aggregate(selected),
            "paired": paired(selected, baseline_by_id),
        },
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-nli-report", type=Path, required=True)
    parser.add_argument("--confirmation-nli-report", type=Path, required=True)
    parser.add_argument("--replication-nli-report", type=Path, required=True)
    parser.add_argument("--full-nli-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "development": args.development_nli_report,
        "confirmation": args.confirmation_nli_report,
        "replication": args.replication_nli_report,
        "full": args.full_nli_report,
    }
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    adapters = [report["identity"]["adapter_artifacts"] for report in reports.values()]
    if not all(adapter == adapters[0] for adapter in adapters):
        raise SystemExit("NLI reports do not use the same adapter")
    raw = {
        name: load_jsonl(ROOT / report["artifacts"]["records_path"])
        for name, report in reports.items()
    }
    ids = {name: {row["case_id"] for row in rows} for name, rows in raw.items()}
    holdout_union = ids["confirmation"] | ids["replication"]
    if ids["development"] & holdout_union or ids["confirmation"] & ids["replication"]:
        raise SystemExit("development and confirmation windows overlap")
    if not ids["development"] | holdout_union <= ids["full"]:
        raise SystemExit("full NLI report does not contain the declared windows")

    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    baseline_by_id = {
        row["case_id"]: row
        for row in load_jsonl(
            ROOT / baseline_report["artifacts"]["case_records_path"]
        )
        if row["benchmark_id"] == "fever"
    }
    if ids["full"] != set(baseline_by_id):
        raise SystemExit("full NLI and baseline FEVER universes differ")

    works = {name: reconstruct_works(report) for name, report in reports.items()}
    matrix = []
    labels = []
    groups = []
    development_document_ids = {}
    case_slices = {}
    for row in raw["development"]:
        start = len(matrix)
        case_matrix, document_ids = candidate_features(
            row, works["development"][row["case_id"]], baseline_by_id[row["case_id"]]
        )
        matrix.extend(case_matrix)
        gold = set(row["gold_document_ids"])
        labels.extend(document_id in gold for document_id in document_ids)
        groups.extend([row["case_id"]] * len(case_matrix))
        development_document_ids[row["case_id"]] = document_ids
        case_slices[row["case_id"]] = (start, len(matrix))
    matrix_array = np.asarray(matrix, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups)
    development_baseline = aggregate(
        [baseline_by_id[row["case_id"]] for row in raw["development"]]
    )["fever"]
    precision_floor = (
        development_baseline["citation_precision"]
        + DEVELOPMENT_PRECISION_SAFETY_MARGIN
    )
    candidate_names = [
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
    ]
    folds = GroupKFold(n_splits=5)
    candidate_results = []
    for model_name in candidate_names:
        out_of_fold = np.zeros(len(label_array), dtype=np.float64)
        for train_indices, validation_indices in folds.split(
            matrix_array, label_array, group_array
        ):
            candidate = model_factory(model_name)
            candidate.fit(matrix_array[train_indices], label_array[train_indices])
            out_of_fold[validation_indices] = candidate.predict_proba(
                matrix_array[validation_indices]
            )[:, 1]
        development_predictions = {
            case_id: (
                development_document_ids[case_id],
                out_of_fold[start:stop].tolist(),
            )
            for case_id, (start, stop) in case_slices.items()
        }
        sweep = []
        for threshold in THRESHOLDS:
            evaluation, _ = evaluate(
                raw["development"],
                baseline_by_id,
                development_predictions,
                threshold,
            )
            metrics = evaluation["selected_metrics"]["fever"]
            sweep.append(
                {
                    "threshold": threshold,
                    "joint_correct_rate": metrics["joint_correct_rate"],
                    "citation_precision": metrics["citation_precision"],
                    "citation_recall": metrics["citation_recall"],
                }
            )
        eligible = [
            row for row in sweep if row["citation_precision"] >= precision_floor
        ]
        if not eligible:
            continue
        selected_policy = sorted(
            eligible,
            key=lambda row: (
                -row["joint_correct_rate"],
                -row["citation_recall"],
                row["threshold"],
                -row["citation_precision"],
            ),
        )[0]
        candidate_results.append(
            {
                "model_name": model_name,
                "selected_oof": selected_policy,
                "threshold_sweep": sweep,
            }
        )
    if not candidate_results:
        raise SystemExit("no evidence model preserves development citation precision")
    selected_candidate = sorted(
        candidate_results,
        key=lambda row: (
            -row["selected_oof"]["joint_correct_rate"],
            -row["selected_oof"]["citation_recall"],
            row["selected_oof"]["threshold"],
            candidate_names.index(row["model_name"]),
        ),
    )[0]
    selected_model_name = selected_candidate["model_name"]
    selected_threshold = selected_candidate["selected_oof"]
    threshold = selected_threshold["threshold"]
    model = model_factory(selected_model_name)
    model.fit(matrix_array, label_array)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    predictions = {
        name: predict_cases(model, rows, works[name], baseline_by_id)
        for name, rows in raw.items()
    }

    evaluations = {}
    selected_rows = {}
    for name in raw:
        evaluations[name], selected_rows[name] = evaluate(
            raw[name], baseline_by_id, predictions[name], threshold
        )
    combined_rows = selected_rows["confirmation"] + selected_rows["replication"]
    combined_confirmation = {
        "case_count": len(combined_rows),
        "baseline_metrics": aggregate(
            [baseline_by_id[row["case_id"]] for row in combined_rows]
        ),
        "selected_metrics": aggregate(combined_rows),
        "paired": paired(combined_rows, baseline_by_id),
    }
    full_selected_metrics = evaluations["full"]["selected_metrics"]["fever"]
    full_baseline_metrics = evaluations["full"]["baseline_metrics"]["fever"]
    promotion_ready = (
        combined_confirmation["paired"]["net_win_count"] > 0
        and combined_confirmation["paired"]["exact_paired_two_sided_p_value"] <= 0.05
        and full_selected_metrics["citation_precision"]
        >= full_baseline_metrics["citation_precision"]
        and not any(row["fail_closed"] for row in selected_rows["full"])
    )

    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, selected_rows["full"])
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "label_policy": "maximum_document_relation_with_frozen_threshold_0.75",
        "evidence_policy": "baseline_evidence_on_label_agreement_plus_selected_document_classifier",
        "feature_names": FEATURE_NAMES,
        "candidate_models": candidate_names,
        "model_selection": "five-fold case-grouped out-of-fold development evaluation",
        "selected_model": selected_model_name,
        "selected_evidence_probability_threshold": threshold,
        "threshold_selection": "maximize out-of-fold development joint and recall, then choose the lowest eligible threshold, subject to baseline citation precision plus a one-point safety margin",
        "development_precision_safety_margin": DEVELOPMENT_PRECISION_SAFETY_MARGIN,
        "random_state": 42,
        "sklearn_version": sklearn.__version__,
        "training_pair_count": len(labels),
        "training_positive_evidence_count": int(sum(labels)),
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
        "development_precision_floor": precision_floor,
        "selected_development_policy": selected_threshold,
        "model_selection": candidate_results,
        **evaluations,
        "combined_confirmation": combined_confirmation,
        "artifacts": {
            "model_path": args.model_out.resolve().relative_to(ROOT).as_posix(),
            "model_sha256": sha256_file(args.model_out),
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(selected_rows["full"]),
        },
        "limitations": [
            "The evidence model family and threshold use only case-grouped out-of-fold predictions in the declared development window.",
            "The disjoint confirmation windows jointly control promotion.",
            "Promotion requires full-run citation precision no lower than the frozen DeepSeek baseline.",
            "The full score includes development and confirmation cases and is not another holdout.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "promotion_ready": promotion_ready,
                "selected_model": selected_model_name,
                "selected_development_policy": selected_threshold,
                "combined_confirmation": combined_confirmation,
                "full": evaluations["full"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
