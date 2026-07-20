"""Compose and evaluate the frozen Hotpot top-8 answer/citation policy."""
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
    aggregate,
    canonical_sha256,
    dense_work_items,
    load_jsonl,
    normalize_answer,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.0.0"


def exact_paired_p_value(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, position)
        for position in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def rescore_citations(row: dict, cited_documents: set[str], work: dict) -> dict:
    updated = dict(row)
    gold_documents = set(work["gold_document_ids"])
    evidence_complete = any(
        set(evidence_set) <= cited_documents
        for evidence_set in work["gold_evidence_sets"]
    )
    updated.update(
        {
            "cited_document_ids": sorted(cited_documents),
            "citation_precision": (
                len(cited_documents & gold_documents) / len(cited_documents)
                if cited_documents
                else None
            ),
            "citation_recall": (
                len(cited_documents & gold_documents) / len(gold_documents)
                if gold_documents
                else None
            ),
            "evidence_complete": evidence_complete,
            "legacy_union_evidence_complete": evidence_complete,
            "joint_correct": bool(row["answer_exact_match"] and evidence_complete),
            "legacy_union_joint_correct": bool(
                row["answer_exact_match"] and evidence_complete
            ),
        }
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top8-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--development-top8-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "confirmation" and not args.development_top8_report:
        raise SystemExit("confirmation requires --development-top8-report")

    top8_report = json.loads(args.top8_report.read_text(encoding="utf-8"))
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    top8_rows = load_jsonl(ROOT / top8_report["artifacts"]["case_records_path"])
    baseline_all = {
        row["case_id"]: row
        for row in load_jsonl(
            ROOT / baseline_report["artifacts"]["case_records_path"]
        )
    }
    top8_ids = {row["case_id"] for row in top8_rows}
    if len(top8_ids) != len(top8_rows) or not top8_ids <= set(baseline_all):
        raise SystemExit("top-8 cases are duplicated or absent from baseline")
    identity = top8_report["identity"]
    if identity["top_k"] != 8 or identity["benchmarks"] != ["hotpotqa"]:
        raise SystemExit("candidate is not the declared Hotpot top-8 route")
    works = dense_work_items(
        "hotpotqa",
        top_k=8,
        max_document_chars=identity["max_document_chars"],
        sample_per_benchmark=identity["sample_per_benchmark"],
        sample_offset_per_benchmark=identity["sample_offset_per_benchmark"],
    )
    work_by_id = {row["case_id"]: row for row in works}
    if set(work_by_id) != top8_ids:
        raise SystemExit("reconstructed work universe differs from candidate")

    development_overlap_count = 0
    development_input = None
    if args.development_top8_report:
        development_report = json.loads(
            args.development_top8_report.read_text(encoding="utf-8")
        )
        development_rows = load_jsonl(
            ROOT / development_report["artifacts"]["case_records_path"]
        )
        development_ids = {row["case_id"] for row in development_rows}
        development_overlap_count = len(top8_ids & development_ids)
        if development_overlap_count:
            raise SystemExit("confirmation overlaps development")
        development_input = {
            "path": args.development_top8_report.resolve()
            .relative_to(ROOT)
            .as_posix(),
            "sha256": sha256_file(args.development_top8_report),
            "case_ids_sha256": canonical_sha256(sorted(development_ids)),
        }

    selected = []
    agreement_count = 0
    for row in top8_rows:
        baseline = baseline_all[row["case_id"]]
        if normalize_answer(row["prediction"]) == normalize_answer(
            baseline["prediction"]
        ):
            agreement_count += 1
            cited = set(row["cited_document_ids"]) | set(
                baseline["cited_document_ids"]
            )
            selected.append(rescore_citations(row, cited, work_by_id[row["case_id"]]))
        else:
            selected.append(dict(row))
    baseline_rows = [baseline_all[row["case_id"]] for row in top8_rows]
    selected_by_id = {row["case_id"]: row for row in selected}
    selected_only = sum(
        selected_by_id[row["case_id"]]["joint_correct"]
        and not row["joint_correct"]
        for row in baseline_rows
    )
    baseline_only = sum(
        row["joint_correct"]
        and not selected_by_id[row["case_id"]]["joint_correct"]
        for row in baseline_rows
    )
    pair = {
        "selected_only_win_count": selected_only,
        "baseline_only_win_count": baseline_only,
        "net_win_count": selected_only - baseline_only,
        "exact_paired_two_sided_p_value": exact_paired_p_value(
            selected_only, baseline_only
        ),
    }
    baseline_metrics = aggregate(baseline_rows)
    selected_metrics = aggregate(selected)
    delta = (
        selected_metrics["hotpotqa"]["joint_correct_rate"]
        - baseline_metrics["hotpotqa"]["joint_correct_rate"]
    )
    promotion_ready = (
        args.stage == "confirmation"
        and delta > 0
        and pair["exact_paired_two_sided_p_value"] <= 0.05
        and not any(row["fail_closed"] for row in selected)
    )
    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, selected)
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": f"{args.stage}_complete",
        "promotion_ready": promotion_ready,
        "identity": {
            "runner_version": RUNNER_VERSION,
            "runner_source_sha256": sha256_file(Path(__file__)),
            "policy": "top8_answer_union_top4_citations_only_on_answer_agreement",
            "sample_offset": identity["sample_offset_per_benchmark"],
            "sample_count": len(selected),
            "development_overlap_count": development_overlap_count,
            "case_ids_sha256": canonical_sha256(sorted(top8_ids)),
        },
        "inputs": {
            "top8_report": {
                "path": args.top8_report.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.top8_report),
            },
            "baseline_report": {
                "path": args.baseline_report.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.baseline_report),
            },
            "development_top8_report": development_input,
        },
        "answer_agreement_count": agreement_count,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "joint_correct_absolute_delta": delta,
        "paired": pair,
        "artifacts": {
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(selected),
        },
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "baseline_joint": baseline_metrics["hotpotqa"][
                    "joint_correct_rate"
                ],
                "selected_joint": selected_metrics["hotpotqa"][
                    "joint_correct_rate"
                ],
                "paired": pair,
                "promotion_ready": promotion_ready,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
