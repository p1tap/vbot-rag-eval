"""Audit a frozen document-level FEVER LoRA aggregation policy."""
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
    exact_paired_p_value,
)
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    aggregate,
    canonical_sha256,
    load_jsonl,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.0.0"
FROZEN_RELATION_SCORE_THRESHOLD = 0.75
FROZEN_CITATION_SCORE_RATIO = 20.0


def select_prediction(raw: dict) -> tuple[str, list[str]]:
    """Map document NLI scores to one FEVER label and a bounded evidence set."""
    candidates = raw.get("candidates", [])
    if not candidates:
        # An empty retrieval result has no evidence for supports/refutes. This is
        # a deterministic NEI decision, not a missing model response.
        return "not_enough_info", []
    if len({row["document_id"] for row in candidates}) != len(candidates):
        raise ValueError(f"{raw['case_id']} has duplicate candidate documents")

    entailment = max(float(row["entailment"]) for row in candidates)
    contradiction = max(float(row["contradiction"]) for row in candidates)
    relation_score = max(entailment, contradiction)
    if relation_score < FROZEN_RELATION_SCORE_THRESHOLD:
        return "not_enough_info", []

    score_key = "entailment" if entailment >= contradiction else "contradiction"
    prediction = "supports" if score_key == "entailment" else "refutes"
    citation_floor = relation_score / FROZEN_CITATION_SCORE_RATIO
    cited = sorted(
        {
            row["document_id"]
            for row in candidates
            if float(row[score_key]) >= citation_floor
        }
    )
    if not cited:
        raise RuntimeError(f"{raw['case_id']} selected a label without evidence")
    return prediction, cited


def score_case(raw: dict) -> dict:
    prediction, cited_documents = select_prediction(raw)
    cited_set = set(cited_documents)
    gold_documents = set(raw["gold_document_ids"])
    retrieved_documents = set(raw["retrieved_document_ids"])
    gold_label = raw["gold"]["label"]
    label_correct = prediction == gold_label
    if prediction == "not_enough_info":
        evidence_complete = gold_label == "not_enough_info" and not cited_documents
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
        "retrieval_any_gold": bool(gold_documents & retrieved_documents)
        if gold_documents
        else None,
        "retrieval_complete_gold": gold_documents <= retrieved_documents
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
    selected_only = sum(
        row["joint_correct"] and not baseline_by_id[row["case_id"]]["joint_correct"]
        for row in selected
    )
    baseline_only = sum(
        baseline_by_id[row["case_id"]]["joint_correct"] and not row["joint_correct"]
        for row in selected
    )
    return {
        "selected_only_win_count": selected_only,
        "baseline_only_win_count": baseline_only,
        "net_win_count": selected_only - baseline_only,
        "exact_paired_two_sided_p_value": exact_paired_p_value(
            selected_only, baseline_only
        ),
    }


def evaluate(raw_rows: list[dict], baseline_by_id: dict[str, dict]) -> tuple[dict, list[dict]]:
    selected = [score_case(row) for row in raw_rows]
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    nli_paths = {
        "development": args.development_nli_report,
        "confirmation": args.confirmation_nli_report,
        "replication": args.replication_nli_report,
        "full": args.full_nli_report,
    }
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in nli_paths.items()
    }
    adapter_identities = [
        report["identity"]["adapter_artifacts"] for report in reports.values()
    ]
    if not all(identity == adapter_identities[0] for identity in adapter_identities):
        raise SystemExit("NLI reports do not use the same adapter artifacts")
    for name, report in reports.items():
        if report["identity"]["top_k"] != 12:
            raise SystemExit(f"{name} report does not use frozen top-k 12")

    raw = {
        name: load_jsonl(ROOT / report["artifacts"]["records_path"])
        for name, report in reports.items()
    }
    ids = {name: {row["case_id"] for row in rows} for name, rows in raw.items()}
    if ids["development"] & ids["confirmation"]:
        raise SystemExit("development and confirmation cases overlap")
    if ids["development"] & ids["replication"]:
        raise SystemExit("development and replication cases overlap")
    if ids["confirmation"] & ids["replication"]:
        raise SystemExit("confirmation and replication cases overlap")
    if not (
        ids["development"] | ids["confirmation"] | ids["replication"]
    ) <= ids["full"]:
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

    evaluations = {}
    selected = {}
    for name in raw:
        evaluations[name], selected[name] = evaluate(raw[name], baseline_by_id)
    confirmation_rows = selected["confirmation"] + selected["replication"]
    combined_confirmation = {
        "case_count": len(confirmation_rows),
        "baseline_metrics": aggregate(
            [baseline_by_id[row["case_id"]] for row in confirmation_rows]
        ),
        "selected_metrics": aggregate(confirmation_rows),
        "paired": paired(confirmation_rows, baseline_by_id),
    }
    promotion_ready = (
        combined_confirmation["paired"]["net_win_count"] > 0
        and combined_confirmation["paired"]["exact_paired_two_sided_p_value"] <= 0.05
        and not any(row["fail_closed"] for row in selected["full"])
    )

    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, selected["full"])
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "policy": "maximum_document_relation_with_thresholded_same_relation_evidence",
        "frozen_relation_score_threshold": FROZEN_RELATION_SCORE_THRESHOLD,
        "frozen_citation_score_ratio": FROZEN_CITATION_SCORE_RATIO,
        "development_offset": reports["development"]["identity"][
            "sample_offset_per_benchmark"
        ],
        "confirmation_offsets": [
            reports[name]["identity"]["sample_offset_per_benchmark"]
            for name in ("confirmation", "replication")
        ],
        "all_window_overlap_count": 0,
        "case_ids_sha256": canonical_sha256(sorted(ids["full"])),
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
            for name, path in {**nli_paths, "baseline": args.baseline_report}.items()
        },
        **evaluations,
        "combined_confirmation": combined_confirmation,
        "artifacts": {
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(selected["full"]),
        },
        "limitations": [
            "LoRA training and aggregation-policy selection use only the declared development window.",
            "Two disjoint untouched windows jointly control promotion.",
            "The full score includes development and confirmation cases and is not another holdout.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "promotion_ready": promotion_ready,
                "combined_confirmation": combined_confirmation["paired"],
                "combined_confirmation_joint": combined_confirmation[
                    "selected_metrics"
                ]["fever"]["joint_correct_rate"],
                "full": evaluations["full"]["paired"],
                "full_joint": evaluations["full"]["selected_metrics"]["fever"][
                    "joint_correct_rate"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
