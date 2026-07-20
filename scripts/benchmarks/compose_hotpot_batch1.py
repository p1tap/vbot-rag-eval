"""Audit the frozen single-case Hotpot route with agreement-only citation union."""
from __future__ import annotations

import argparse
import copy
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
    dense_work_items,
    load_jsonl,
    normalize_answer,
    write_json,
    write_jsonl,
)
from scripts.benchmarks.train_hotpot_citation_selector import (  # noqa: E402
    FEATURE_NAMES,
    document_features,
    predict_probabilities,
)

RUNNER_VERSION = "1.0.0"
CITATION_PRECISION_FLOOR = 0.94


def paired(selected: list[dict], baseline: list[dict]) -> dict:
    baseline_by_id = {row["case_id"]: row for row in baseline}
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


def compose_case(
    batch1: dict, batch4: dict, work: dict, selector: dict | None = None
) -> dict:
    if batch1["case_id"] != batch4["case_id"]:
        raise ValueError("cannot compose different cases")
    selected = dict(batch1)
    answer_agrees = normalize_answer(batch1["prediction"]) == normalize_answer(
        batch4["prediction"]
    )
    cited = set(batch1["cited_document_ids"])
    if answer_agrees:
        cited.update(batch4["cited_document_ids"])
        if selector is not None:
            probabilities = predict_probabilities(
                document_features(work, batch1, batch4),
                selector["coefficients"],
                selector["intercept"],
            )
            cited.update(
                context["document_id"]
                for context, probability in zip(
                    work["contexts"], probabilities, strict=True
                )
                if probability >= selector["selected_probability_threshold"]
            )
    cited_documents = sorted(cited)
    cited_set = set(cited_documents)
    gold_documents = set(work["gold_document_ids"])
    evidence_complete = not gold_documents or gold_documents <= cited_set
    selected.update(
        {
            "cited_document_ids": cited_documents,
            "citation_precision": len(cited_set & gold_documents) / len(cited_set)
            if cited_set
            else None,
            "citation_recall": len(cited_set & gold_documents) / len(gold_documents)
            if gold_documents
            else None,
            "evidence_complete": evidence_complete,
            "legacy_union_evidence_complete": evidence_complete,
            "joint_correct": bool(batch1["answer_exact_match"] and evidence_complete),
            "legacy_union_joint_correct": bool(
                batch1["answer_exact_match"] and evidence_complete
            ),
        }
    )
    return selected


def load_report_cases(path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = load_jsonl(ROOT / report["artifacts"]["case_records_path"])
    return report, rows


def slice_development_window(
    reports: dict[str, dict],
    cases: dict[str, list[dict]],
    *,
    sample: int,
    offset: int,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Materialize the selector's ranked development window from superset reports."""
    identity = reports["batch1"]["identity"]
    works = dense_work_items(
        "hotpotqa",
        top_k=8,
        max_document_chars=identity["max_document_chars"],
        sample_per_benchmark=sample,
        sample_offset_per_benchmark=offset,
        retrieval_mode=identity["retrieval"]["bounded_candidate_mode"],
        compression_mode=identity["compression"]["mode"],
    )
    expected_ids = {row["case_id"] for row in works}
    sliced_reports = copy.deepcopy(reports)
    sliced_cases = {}
    for size in ("batch1", "batch4"):
        by_id = {row["case_id"]: row for row in cases[size]}
        if not expected_ids <= set(by_id):
            raise SystemExit(
                f"development {size} report does not contain the selector training window"
            )
        sliced_cases[size] = [by_id[case_id] for case_id in sorted(expected_ids)]
        sliced_reports[size]["identity"]["sample_per_benchmark"] = sample
        sliced_reports[size]["identity"]["sample_offset_per_benchmark"] = offset
    return sliced_reports, sliced_cases


def validate_pair(batch1_report: dict, batch4_report: dict, name: str) -> None:
    for report in (batch1_report, batch4_report):
        identity = report["identity"]
        if identity["benchmarks"] != ["hotpotqa"]:
            raise SystemExit(f"{name} contains a non-Hotpot benchmark")
        if identity["top_k_by_benchmark"]["hotpotqa"] != 8:
            raise SystemExit(f"{name} does not use frozen Hotpot top-k 8")
        if identity["citation_policy"] != "complete_hotpot_chain":
            raise SystemExit(f"{name} does not use complete-chain citations")
    if batch1_report["identity"]["batch_size"] != 1:
        raise SystemExit(f"{name} selected report is not batch size 1")
    if batch4_report["identity"]["batch_size"] != 4:
        raise SystemExit(f"{name} comparison report is not batch size 4")
    if (
        batch1_report["identity"]["sample_offset_per_benchmark"]
        != batch4_report["identity"]["sample_offset_per_benchmark"]
    ):
        raise SystemExit(f"{name} report offsets differ")


def evaluate_pair(
    batch1_report: dict,
    batch1_rows: list[dict],
    batch4_rows: list[dict],
    selector: dict,
) -> tuple[dict, list[dict]]:
    identity = batch1_report["identity"]
    works = dense_work_items(
        "hotpotqa",
        top_k=8,
        max_document_chars=identity["max_document_chars"],
        sample_per_benchmark=identity["sample_per_benchmark"],
        sample_offset_per_benchmark=identity["sample_offset_per_benchmark"],
        retrieval_mode=identity["retrieval"]["bounded_candidate_mode"],
        compression_mode=identity["compression"]["mode"],
    )
    work_by_id = {row["case_id"]: row for row in works}
    batch1_by_id = {row["case_id"]: row for row in batch1_rows}
    batch4_all_by_id = {row["case_id"]: row for row in batch4_rows}
    if len(batch1_by_id) != len(batch1_rows) or len(batch4_all_by_id) != len(
        batch4_rows
    ):
        raise SystemExit("duplicate case IDs in input reports")
    if set(batch1_by_id) != set(work_by_id) or not set(batch1_by_id) <= set(
        batch4_all_by_id
    ):
        raise SystemExit("paired Hotpot case universes differ")
    batch4_by_id = {
        case_id: batch4_all_by_id[case_id] for case_id in batch1_by_id
    }
    selected = [
        compose_case(
            batch1_by_id[case_id],
            batch4_by_id[case_id],
            work_by_id[case_id],
            selector,
        )
        for case_id in sorted(batch1_by_id)
    ]
    baseline = [batch4_by_id[row["case_id"]] for row in selected]
    return (
        {
            "case_count": len(selected),
            "answer_agreement_count": sum(
                normalize_answer(batch1_by_id[case_id]["prediction"])
                == normalize_answer(batch4_by_id[case_id]["prediction"])
                for case_id in batch1_by_id
            ),
            "baseline_metrics": aggregate(baseline),
            "batch1_metrics": aggregate(list(batch1_by_id.values())),
            "selected_metrics": aggregate(selected),
            "paired": paired(selected, baseline),
        },
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for window in ("development", "confirmation", "full"):
        parser.add_argument(f"--{window}-batch1-report", type=Path, required=True)
        parser.add_argument(f"--{window}-batch4-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selector-training-report", type=Path, required=True)
    args = parser.parse_args()

    selector_training = json.loads(
        args.selector_training_report.read_text(encoding="utf-8")
    )
    selector_path = ROOT / selector_training["artifacts"]["model_path"]
    if sha256_file(selector_path) != selector_training["artifacts"]["model_sha256"]:
        raise SystemExit("citation selector artifact hash mismatch")
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    if selector["feature_names"] != FEATURE_NAMES:
        raise SystemExit("citation selector feature contract mismatch")
    if canonical_sha256(selector) != selector_training["artifacts"][
        "model_canonical_sha256"
    ]:
        raise SystemExit("citation selector canonical hash mismatch")

    paths = {
        window: {
            size: getattr(args, f"{window}_{size}_report")
            for size in ("batch1", "batch4")
        }
        for window in ("development", "confirmation", "full")
    }
    reports = {}
    cases = {}
    for window, pair in paths.items():
        reports[window] = {}
        cases[window] = {}
        for size, path in pair.items():
            reports[window][size], cases[window][size] = load_report_cases(path)
        validate_pair(reports[window]["batch1"], reports[window]["batch4"], window)
    if selector_training["inputs"]["batch1"]["sha256"] != sha256_file(
        paths["development"]["batch1"]
    ) or selector_training["inputs"]["batch4"]["sha256"] != sha256_file(
        paths["development"]["batch4"]
    ):
        raise SystemExit("citation selector was not trained on the declared development pair")

    training_sample = selector_training["identity"]["training_case_count"]
    training_offset = selector_training["identity"].get(
        "training_sample_offset",
        reports["development"]["batch1"]["identity"][
            "sample_offset_per_benchmark"
        ],
    )
    reports["development"], cases["development"] = slice_development_window(
        reports["development"],
        cases["development"],
        sample=training_sample,
        offset=training_offset,
    )

    development_ids = {row["case_id"] for row in cases["development"]["batch1"]}
    confirmation_ids = {row["case_id"] for row in cases["confirmation"]["batch1"]}
    full_ids = {row["case_id"] for row in cases["full"]["batch1"]}
    if development_ids & confirmation_ids:
        raise SystemExit("development and confirmation cases overlap")
    if not development_ids | confirmation_ids <= full_ids:
        raise SystemExit("full reports do not contain the declared windows")

    evaluations = {}
    selected = {}
    for window in ("development", "confirmation", "full"):
        evaluations[window], selected[window] = evaluate_pair(
            reports[window]["batch1"],
            cases[window]["batch1"],
            cases[window]["batch4"],
            selector,
        )
    promotion_ready = (
        evaluations["confirmation"]["paired"]["net_win_count"] > 0
        and evaluations["confirmation"]["paired"][
            "exact_paired_two_sided_p_value"
        ]
        <= 0.05
        and not any(row["fail_closed"] for row in selected["full"])
        and evaluations["confirmation"]["selected_metrics"]["hotpotqa"][
            "citation_precision"
        ]
        >= CITATION_PRECISION_FLOOR
        and evaluations["full"]["selected_metrics"]["hotpotqa"][
            "citation_precision"
        ]
        >= CITATION_PRECISION_FLOOR
    )

    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, selected["full"])
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "policy": "batch1_answer_with_agreement_gated_batch4_and_logistic_bridge_citations",
        "development_confirmation_overlap_count": 0,
        "development_sample_count": training_sample,
        "development_sample_offset": training_offset,
        "case_ids_sha256": canonical_sha256(sorted(full_ids)),
        "selector_training_report_sha256": sha256_file(args.selector_training_report),
        "selector_model_sha256": sha256_file(selector_path),
        "frozen_selector_probability_threshold": selector[
            "selected_probability_threshold"
        ],
        "confirmation_and_full_citation_precision_floor": CITATION_PRECISION_FLOOR,
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_application_complete",
        "promotion_ready": promotion_ready,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "inputs": {
            "selector_training": {
                "path": args.selector_training_report.resolve()
                .relative_to(ROOT)
                .as_posix(),
                "sha256": sha256_file(args.selector_training_report),
                "model_path": selector_path.resolve().relative_to(ROOT).as_posix(),
                "model_sha256": sha256_file(selector_path),
            },
            "windows": {
            window: {
                size: {
                    "path": path.resolve().relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
                for size, path in pair.items()
            }
            for window, pair in paths.items()
            },
        },
        **evaluations,
        "artifacts": {
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(selected["full"]),
        },
        "limitations": [
            "The one-case generation policy, agreement-only citation union, logistic selector, and selector threshold were selected on the declared development window.",
            "The disjoint confirmation window controls promotion; both confirmation and full application must retain at least 94% citation precision.",
            "The full score includes development and confirmation cases and is not another holdout.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "promotion_ready": promotion_ready,
                "development_joint": evaluations["development"]["selected_metrics"][
                    "hotpotqa"
                ]["joint_correct_rate"],
                "confirmation": evaluations["confirmation"]["paired"],
                "confirmation_joint": evaluations["confirmation"][
                    "selected_metrics"
                ]["hotpotqa"]["joint_correct_rate"],
                "full_joint": evaluations["full"]["selected_metrics"]["hotpotqa"][
                    "joint_correct_rate"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
