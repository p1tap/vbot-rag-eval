"""Build an auditable majority ensemble from frozen compatible E2E reports."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    aggregate,
    canonical_sha256,
    case_results,
    load_jsonl,
    normalize_answer,
    write_json,
    write_jsonl,
)
from scripts.benchmarks.run_verifier_cascade import reconstruct_work  # noqa: E402

RUNNER_VERSION = "1.2.0"


def load_case_rows(report: dict) -> dict[str, dict]:
    path = ROOT / report["artifacts"]["case_records_path"]
    return {row["case_id"]: row for row in load_jsonl(path)}


def normalized_prediction(row: dict) -> str:
    if row["benchmark_id"] == "fever":
        return row["prediction"].strip().casefold()
    return normalize_answer(row["prediction"])


def normalized_top_k_by_benchmark(identity: dict) -> dict[str, int]:
    """Normalize reports written before per-benchmark top-k was recorded."""
    configured = identity.get("top_k_by_benchmark")
    if configured is not None:
        return configured
    return {benchmark: identity["top_k"] for benchmark in identity["benchmarks"]}


def citation_component_indexes(
    benchmark_id: str,
    agreeing: list[int],
    component_count: int,
    hotpot_rule: str,
) -> list[int]:
    if benchmark_id != "hotpotqa":
        return [agreeing[0]]
    if hotpot_rule == "agreement_group":
        return agreeing
    if hotpot_rule == "all_components":
        return list(range(component_count))
    raise ValueError(f"unknown Hotpot citation rule: {hotpot_rule}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--hotpot-citation-rule",
        choices=("agreement_group", "all_components"),
        default="agreement_group",
    )
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    primary = reports[0]
    for position, report in enumerate(reports[1:], 2):
        for key in ("source_sha256", "sample_per_benchmark"):
            if report["identity"][key] != primary["identity"][key]:
                raise SystemExit(f"report {position} identity mismatch: {key}")
        if normalized_top_k_by_benchmark(
            report["identity"]
        ) != normalized_top_k_by_benchmark(primary["identity"]):
            raise SystemExit(f"report {position} identity mismatch: top_k_by_benchmark")
        if report["case_count"] != primary["case_count"]:
            raise SystemExit(f"report {position} case count mismatch")

    works = reconstruct_work(primary)
    work_by_id = {work["case_id"]: work for work in works}
    component_rows = [load_case_rows(report) for report in reports]
    expected_ids = set(work_by_id)
    if any(set(rows) != expected_ids for rows in component_rows):
        raise SystemExit("component reports do not contain the reconstructed case IDs")

    selected_results = {}
    agreement_counts = Counter()
    for case_id, work in work_by_id.items():
        rows = [component[case_id] for component in component_rows]
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(normalized_prediction(row), []).append(index)
        agreeing = sorted(
            groups.values(),
            key=lambda indexes: (
                -len(indexes),
                0 if 0 in indexes else 1,
                indexes,
            ),
        )[0]
        required = len(rows) // 2 + 1
        if len(agreeing) < required:
            agreeing = [0]
            agreement_counts["primary_fallback"] += 1
        else:
            agreement_counts[f"majority_{len(agreeing)}"] += 1
        chosen = rows[agreeing[0]]
        citation_indexes = citation_component_indexes(
            work["benchmark_id"], agreeing, len(reports), args.hotpot_citation_rule
        )
        cited_documents = {
            document_id
            for index in citation_indexes
            for document_id in rows[index]["cited_document_ids"]
        }
        document_to_citation = {
            context["document_id"]: context["citation_id"]
            for context in work["contexts"]
        }
        selected_results[case_id] = {
            "case_id": case_id,
            "prediction": chosen["prediction"],
            "citation_ids": sorted(
                document_to_citation[document_id]
                for document_id in cited_documents
                if document_id in document_to_citation
            ),
        }

    primary_batch_path = ROOT / primary["artifacts"]["batch_records_path"]
    batch_history = load_jsonl(primary_batch_path)
    batch_by_id = {batch["batch_id"]: batch for batch in batch_history}
    batches = copy.deepcopy(
        sorted(batch_by_id.values(), key=lambda batch: batch["batch_id"])
    )
    for batch in batches:
        batch["frozen_primary_results_sha256"] = canonical_sha256(batch["results"])
        batch["results"] = [selected_results[case_id] for case_id in batch["case_ids"]]
        batch["consistency_ensemble"] = {
            "component_count": len(reports),
            "selection_rule": "strict_normalized_majority_primary_fallback",
            "citation_rule": (
                f"hotpot_union_{args.hotpot_citation_rule}_"
                "other_tasks_selected_result_only"
            ),
        }

    rows = case_results(batches, work_by_id)
    batch_path = args.out.with_suffix(".batches.jsonl")
    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(batch_path, batches)
    write_jsonl(case_path, rows)
    identity = {
        **primary["identity"],
        "generator_runner_version": primary["identity"]["runner_version"],
        "generator_runner_source_sha256": primary["identity"]["runner_source_sha256"],
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "selection_rule": "strict_normalized_majority_primary_fallback",
        "citation_rule": (
            f"hotpot_union_{args.hotpot_citation_rule}_"
            "other_tasks_selected_result_only"
        ),
        "component_reports": [
            {
                "path": path.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "case_results_sha256": report["artifacts"][
                    "case_results_canonical_sha256"
                ],
            }
            for path, report in zip(args.reports, reports, strict=True)
        ],
        "nq_evidence_semantics": "any_complete_non_null_human_annotation",
        "legacy_nq_evidence_semantics": "union_of_consensus_annotations",
    }
    report = copy.deepcopy(primary)
    report.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "metrics": aggregate(rows),
            "fail_closed_case_count": sum(row["fail_closed"] for row in rows),
            "consistency_ensemble": {
                "component_count": len(reports),
                "agreement_counts": dict(sorted(agreement_counts.items())),
            },
            "artifacts": {
                "batch_records_path": batch_path.resolve().relative_to(ROOT).as_posix(),
                "batch_records_sha256": sha256_file(batch_path),
                "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
                "case_records_sha256": sha256_file(case_path),
                "case_results_canonical_sha256": canonical_sha256(rows),
            },
            "limitations": [
                *primary.get("limitations", []),
                "Natural Questions evidence correctness accepts any one complete non-null human annotation, matching Google's official alternative-annotation semantics; the former union metric remains reported as a legacy diagnostic.",
                "The consistency ensemble uses only frozen model outputs and a predeclared majority rule; it has no access to gold labels during selection.",
            ],
        }
    )
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "case_count": len(rows),
                "metrics": report["metrics"],
                "consistency_ensemble": report["consistency_ensemble"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
