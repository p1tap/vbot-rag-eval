"""Train and audit the promoted Hotpot pair-and-route specialist."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
    dense_work_items,
    load_jsonl,
    normalize_answer,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.0.0"
DEVELOPMENT_SAMPLE = 2400
DEVELOPMENT_OFFSET = 0
CONFIRMATION_WINDOWS = {"confirmation": (300, 2400), "replication": (300, 2700)}
PAIR_THRESHOLD = 0.15
ROUTE_THRESHOLD = 0.63
CITATION_PRECISION_FLOOR = 0.94


def tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def load_cases(path: Path) -> tuple[dict, dict[str, dict]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = load_jsonl(ROOT / report["artifacts"]["case_records_path"])
    return report, {row["case_id"]: row for row in rows}


def pair_features(work: dict, batch1: dict, batch4: dict, left: int, right: int) -> list[float]:
    documents = [work["contexts"][left], work["contexts"][right]]
    query_tokens = tokens(work["query"])
    answer_tokens = tokens(batch1["prediction"])
    values = []
    for document, rank, other in (
        (documents[0], left, documents[1]),
        (documents[1], right, documents[0]),
    ):
        title_tokens = tokens(document["title"])
        document_tokens = tokens(document["text"])
        other_tokens = tokens(other["text"])
        values.extend(
            [
                1.0 / (rank + 1),
                float(rank + 1),
                len(query_tokens & title_tokens) / max(1, len(query_tokens)),
                len(query_tokens & document_tokens) / max(1, len(query_tokens)),
                len(answer_tokens & document_tokens) / max(1, len(answer_tokens)),
                float(normalize_answer(batch1["prediction"]) in normalize_answer(document["text"])),
                float(document["document_id"] in batch1["cited_document_ids"]),
                float(document["document_id"] in batch4["cited_document_ids"]),
                len(title_tokens & other_tokens) / max(1, len(title_tokens)),
                float(normalize_answer(document["title"]) in normalize_answer(other["text"])),
            ]
        )
    pair = {row["document_id"] for row in documents}
    batch1_citations = set(batch1["cited_document_ids"])
    batch4_citations = set(batch4["cited_document_ids"])
    comparison_tokens = {"both", "older", "younger", "first", "earlier", "later", "more", "less"}
    values.extend(
        [
            float(normalize_answer(batch1["prediction"]) == normalize_answer(batch4["prediction"])),
            float(len(pair & batch1_citations)),
            float(len(pair & batch4_citations)),
            float(batch1_citations <= pair),
            float(batch4_citations <= pair),
            float(bool(pair & batch1_citations)),
            float(bool(pair & batch4_citations)),
            float(left == 0 or right == 0),
            float(bool(comparison_tokens & query_tokens)),
        ]
    )
    return values


def route_features(
    work: dict,
    batch1: dict,
    batch4: dict,
    pair_probability: float,
    pair_margin: float,
) -> list[float]:
    query_tokens = tokens(work["query"])

    def per_route(row: dict) -> list[float]:
        answer_tokens = tokens(row["prediction"])
        cited = set(row["cited_document_ids"])
        documents = [item for item in work["contexts"] if item["document_id"] in cited]
        return [
            float(len(row["prediction"])),
            float(len(answer_tokens)),
            float(len(cited)),
            float(sum(normalize_answer(row["prediction"]) in normalize_answer(item["text"]) for item in work["contexts"])),
            float(sum(normalize_answer(row["prediction"]) in normalize_answer(item["text"]) for item in documents)),
            sum(
                len(answer_tokens & tokens(item["text"])) / max(1, len(answer_tokens))
                for item in documents
            )
            / max(1, len(documents)),
        ]

    left = normalize_answer(batch1["prediction"])
    right = normalize_answer(batch4["prediction"])
    comparison_tokens = {"both", "older", "younger", "first", "earlier", "later", "more", "less"}
    return [
        float(left == right),
        float(left in right or right in left),
        float(len(set(batch1["cited_document_ids"]) & set(batch4["cited_document_ids"]))),
        pair_probability,
        pair_margin,
        float(bool(comparison_tokens & query_tokens)),
        *per_route(batch1),
        *per_route(batch4),
    ]


def pair_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )


def route_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_features=0.9,
    )


def pair_candidates(work: dict, batch1: dict, batch4: dict) -> tuple[np.ndarray, list[tuple[str, str]]]:
    matrix = []
    pairs = []
    for left, right in itertools.combinations(range(len(work["contexts"])), 2):
        matrix.append(pair_features(work, batch1, batch4, left, right))
        pairs.append((work["contexts"][left]["document_id"], work["contexts"][right]["document_id"]))
    return np.asarray(matrix, dtype=np.float64), pairs


def apply_pair(base: dict, batch1: dict, batch4: dict, work: dict, probabilities: np.ndarray, pairs: list[tuple[str, str]]) -> tuple[dict, float, float]:
    selected = dict(base)
    top_index = int(np.argmax(probabilities))
    pair = set(pairs[top_index])
    cited = set(base["cited_document_ids"])
    answers_agree = normalize_answer(batch1["prediction"]) == normalize_answer(batch4["prediction"])
    if answers_agree and probabilities[top_index] >= PAIR_THRESHOLD and pair & cited:
        cited = pair
    gold = set(work["gold_document_ids"])
    evidence_complete = gold <= cited
    selected.update(
        {
            "cited_document_ids": sorted(cited),
            "citation_precision": len(cited & gold) / len(cited) if cited else None,
            "citation_recall": len(cited & gold) / len(gold) if gold else None,
            "evidence_complete": evidence_complete,
            "joint_correct": bool(batch1["answer_exact_match"] and evidence_complete),
            "legacy_union_evidence_complete": evidence_complete,
            "legacy_union_joint_correct": bool(batch1["answer_exact_match"] and evidence_complete),
        }
    )
    ordered = sorted((float(value) for value in probabilities), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    return selected, ordered[0], margin


def paired(selected: list[dict], baseline: dict[str, dict]) -> dict:
    wins = sum(row["joint_correct"] and not baseline[row["case_id"]]["joint_correct"] for row in selected)
    losses = sum(baseline[row["case_id"]]["joint_correct"] and not row["joint_correct"] for row in selected)
    return {
        "selected_only_win_count": wins,
        "baseline_only_win_count": losses,
        "net_win_count": wins - losses,
        "exact_paired_two_sided_p_value": exact_paired_p_value(wins, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch1-report", type=Path, required=True)
    parser.add_argument("--batch4-report", type=Path, required=True)
    parser.add_argument("--baseline-policy-report", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    loaded = [load_cases(path) for path in (args.batch1_report, args.batch4_report, args.baseline_policy_report)]
    reports = [item[0] for item in loaded]
    batch1, batch4, baseline = [item[1] for item in loaded]
    works = dense_work_items("hotpotqa", top_k=8, max_document_chars=6000, sample_per_benchmark=3000, sample_offset_per_benchmark=0)
    work_by_id = {row["case_id"]: row for row in works}
    if not set(work_by_id) == set(batch1) == set(batch4) == set(baseline):
        raise SystemExit("Hotpot full report universes differ")
    development_ids = {
        row["case_id"]
        for row in dense_work_items(
            "hotpotqa",
            top_k=8,
            max_document_chars=6000,
            sample_per_benchmark=DEVELOPMENT_SAMPLE,
            sample_offset_per_benchmark=DEVELOPMENT_OFFSET,
        )
    }
    window_ids = {
        name: {
            row["case_id"]
            for row in dense_work_items(
                "hotpotqa",
                top_k=8,
                max_document_chars=6000,
                sample_per_benchmark=sample,
                sample_offset_per_benchmark=offset,
            )
        }
        for name, (sample, offset) in CONFIRMATION_WINDOWS.items()
    }
    if any(development_ids & ids for ids in window_ids.values()) or window_ids["confirmation"] & window_ids["replication"]:
        raise SystemExit("development and confirmation windows overlap")

    matrix = []
    labels = []
    groups = []
    slices = {}
    for case_id in sorted(development_ids):
        start = len(matrix)
        case_matrix, pairs = pair_candidates(work_by_id[case_id], batch1[case_id], batch4[case_id])
        matrix.extend(case_matrix.tolist())
        gold = set(work_by_id[case_id]["gold_document_ids"])
        labels.extend(set(pair) == gold for pair in pairs)
        groups.extend([case_id] * len(pairs))
        slices[case_id] = (start, len(matrix), pairs)
    matrix_array = np.asarray(matrix, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups)
    oof = np.zeros(len(label_array), dtype=np.float64)
    for train, validation in GroupKFold(5).split(matrix_array, label_array, group_array):
        model = pair_model().fit(matrix_array[train], label_array[train])
        oof[validation] = model.predict_proba(matrix_array[validation])[:, 1]
    oof_pair_rows = {}
    oof_pair_metadata = {}
    for case_id, (start, stop, pairs) in slices.items():
        row, top, margin = apply_pair(
            baseline[case_id],
            batch1[case_id],
            batch4[case_id],
            work_by_id[case_id],
            oof[start:stop],
            pairs,
        )
        oof_pair_rows[case_id] = row
        oof_pair_metadata[case_id] = (top, margin)
    action_matrix = np.asarray(
        [
            route_features(
                work_by_id[case_id],
                batch1[case_id],
                batch4[case_id],
                *oof_pair_metadata[case_id],
            )
            for case_id in sorted(development_ids)
        ],
        dtype=np.float64,
    )
    ordered_development = sorted(development_ids)
    action_labels = np.asarray(
        [
            int(batch4[case_id]["joint_correct"] and not oof_pair_rows[case_id]["joint_correct"])
            for case_id in ordered_development
        ],
        dtype=np.int64,
    )
    action_losses = np.asarray(
        [
            int(oof_pair_rows[case_id]["joint_correct"] and not batch4[case_id]["joint_correct"])
            for case_id in ordered_development
        ],
        dtype=np.int64,
    )
    informative = (action_labels + action_losses) > 0
    final_pair_model = pair_model().fit(matrix_array, label_array)
    final_route_model = route_model().fit(action_matrix[informative], action_labels[informative])
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pair_model": final_pair_model,
            "route_model": final_route_model,
            "pair_threshold": PAIR_THRESHOLD,
            "route_threshold": ROUTE_THRESHOLD,
        },
        args.model_out,
    )

    selected_by_id = {}
    route_b_count = 0
    for case_id in sorted(work_by_id):
        case_matrix, pairs = pair_candidates(work_by_id[case_id], batch1[case_id], batch4[case_id])
        probabilities = final_pair_model.predict_proba(case_matrix)[:, 1]
        pair_row, top, margin = apply_pair(
            baseline[case_id], batch1[case_id], batch4[case_id], work_by_id[case_id], probabilities, pairs
        )
        action_probability = final_route_model.predict_proba(
            np.asarray(
                [route_features(work_by_id[case_id], batch1[case_id], batch4[case_id], top, margin)],
                dtype=np.float64,
            )
        )[0, 1]
        if action_probability >= ROUTE_THRESHOLD:
            selected_by_id[case_id] = dict(batch4[case_id])
            route_b_count += 1
        else:
            selected_by_id[case_id] = pair_row
    selected = [selected_by_id[case_id] for case_id in sorted(selected_by_id)]
    evaluations = {}
    combined = []
    for name, ids in window_ids.items():
        rows = [selected_by_id[case_id] for case_id in sorted(ids)]
        combined.extend(rows)
        evaluations[name] = {
            "case_count": len(rows),
            "baseline_metrics": aggregate([baseline[row["case_id"]] for row in rows]),
            "selected_metrics": aggregate(rows),
            "paired": paired(rows, baseline),
        }
    combined_evaluation = {
        "case_count": len(combined),
        "baseline_metrics": aggregate([baseline[row["case_id"]] for row in combined]),
        "selected_metrics": aggregate(combined),
        "paired": paired(combined, baseline),
    }
    metrics = aggregate(selected)
    promotion_ready = (
        combined_evaluation["paired"]["net_win_count"] > 0
        and combined_evaluation["paired"]["exact_paired_two_sided_p_value"] <= 0.05
        and metrics["hotpotqa"]["citation_precision"] >= CITATION_PRECISION_FLOOR
        and all(
            evaluation["selected_metrics"]["hotpotqa"]["citation_precision"] >= CITATION_PRECISION_FLOOR
            for evaluation in evaluations.values()
        )
        and not any(row["fail_closed"] for row in selected)
    )
    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, selected)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "policy": "grouped_oof_pair_selector_then_frozen_batch1_or_batch4_route_selector",
        "development_sample": DEVELOPMENT_SAMPLE,
        "development_offset": DEVELOPMENT_OFFSET,
        "confirmation_windows": CONFIRMATION_WINDOWS,
        "pair_threshold": PAIR_THRESHOLD,
        "route_threshold": ROUTE_THRESHOLD,
        "citation_precision_floor": CITATION_PRECISION_FLOOR,
        "random_state": 42,
        "sklearn_version": sklearn.__version__,
        "development_case_ids_sha256": canonical_sha256(sorted(development_ids)),
        "full_case_ids_sha256": canonical_sha256(sorted(work_by_id)),
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_ready" if promotion_ready else "rejected",
        "promotion_ready": promotion_ready,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "case_count": len(selected),
        "fail_closed_case_count": sum(row["fail_closed"] for row in selected),
        "metrics": metrics,
        "selected_batch4_route_count": route_b_count,
        "combined_confirmation": combined_evaluation,
        **evaluations,
        "inputs": {
            name: {
                "path": path.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "identity_sha256": source["identity_sha256"],
            }
            for name, path, source in zip(
                ("batch1", "batch4", "baseline_policy"),
                (args.batch1_report, args.batch4_report, args.baseline_policy_report),
                reports,
                strict=True,
            )
        },
        "artifacts": {
            "model_path": args.model_out.resolve().relative_to(ROOT).as_posix(),
            "model_sha256": sha256_file(args.model_out),
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(selected),
        },
        "limitations": [
            "The complete 3,000-case score includes development and confirmation cases and is not another holdout.",
            "The route selector may only retain the batch-size-1 DeepSeek result or select the existing batch-size-4 DeepSeek result.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "promotion_ready": promotion_ready,
                "combined_confirmation": combined_evaluation["paired"],
                "full_joint": metrics["hotpotqa"]["joint_correct_rate"],
                "full_citation_precision": metrics["hotpotqa"]["citation_precision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
