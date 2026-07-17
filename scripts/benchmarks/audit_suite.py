from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER_SOURCE_PATH = ROOT / "scripts" / "benchmarks" / "run_end_to_end.py"

from scripts.benchmarks.prepare import load_registry  # noqa: E402

DEFAULT_INPUTS = {
    "hotpotqa": ROOT
    / "artifacts"
    / "benchmarks"
    / "hotpotqa"
    / "normalized"
    / "hotpotqa-dev-distractor-3000.jsonl",
    "natural_questions": ROOT
    / "artifacts"
    / "benchmarks"
    / "natural_questions"
    / "normalized"
    / "natural-questions-dev-5000.jsonl",
    "fever": ROOT
    / "artifacts"
    / "benchmarks"
    / "fever"
    / "normalized"
    / "fever-dev-2000.jsonl",
}

DEFAULT_REPORTS = {
    benchmark_id: path.parents[1] / "reports" / path.with_suffix(".json").name
    for benchmark_id, path in DEFAULT_INPUTS.items()
}
AI_SPOT_AUDIT = (
    ROOT / "reports" / "public-benchmarks" / "public-suite-spot-audit-ai.json"
)
MAX_NORMALIZED_BATCH_RATE = 0.05
MAX_CASE_SLOT_RETRY_BATCH_RATE = 0.01
PUBLIC_E2E_RESPONSE_CONTRACT = "public-rag-batch-1.4.0-observed-serialization"
ALLOWED_NORMALIZATION_EVENTS = {
    "stripped_markdown_fence",
    "stripped_trailing_backtick",
    "restored_missing_array_opener",
    "wrapped_results_array",
    "wrapped_single_result",
    "restored_empty_abstention_citations",
    "canonicalized_fever_abstention_label",
    "stripped_extra_trailing_closer",
    "restored_near_case_id_from_frozen_order",
    "regrouped_flattened_result_fields",
    "wrapped_comma_separated_results",
    "restored_missing_root_closer",
    "split_merged_result_fields",
    "serialized_nq_prediction_string_array",
}


def _artifact_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f"invalid end-to-end artifact path: {relative}")
    return path


def validate_end_to_end(
    path: Path, expected_source_sha256: dict[str, str], expected_case_ids: set[str]
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.get("identity", {})
    if value.get("status") != "complete" or value.get("resume_claim_ready") is not True:
        raise ValueError("end-to-end report is not complete and claim-ready")
    if not expected_case_ids or value.get("case_count") != len(expected_case_ids):
        raise ValueError("end-to-end report case count mismatch")
    if value.get("batch_count") != value.get("valid_batch_count"):
        raise ValueError("end-to-end report contains invalid batches")
    raw_valid_batches = value.get("raw_contract_valid_batch_count")
    normalized_batches = value.get("deterministically_normalized_batch_count")
    slot_retry_batches = value.get("case_slot_retry_batch_count")
    if not all(
        isinstance(item, int)
        for item in (raw_valid_batches, normalized_batches, slot_retry_batches)
    ):
        raise ValueError("end-to-end report omits raw/normalized/slot-retry counts")
    if raw_valid_batches + normalized_batches + slot_retry_batches != value.get("batch_count"):
        raise ValueError("end-to-end execution-path counts are inconsistent")
    normalization_rate = normalized_batches / value["batch_count"]
    slot_retry_rate = slot_retry_batches / value["batch_count"]
    if normalization_rate > MAX_NORMALIZED_BATCH_RATE:
        raise ValueError("end-to-end deterministic normalization rate exceeds guardrail")
    guardrails = value.get("contract_guardrails") or {}
    if (
        guardrails.get("max_deterministic_normalization_rate")
        != MAX_NORMALIZED_BATCH_RATE
        or guardrails.get("normalization_rate_pass") is not True
        or guardrails.get("max_case_slot_retry_batch_rate")
        != MAX_CASE_SLOT_RETRY_BATCH_RATE
        or guardrails.get("case_slot_retry_rate_pass") is not True
    ):
        raise ValueError("end-to-end report contract guardrail declaration mismatch")
    if value.get("fail_closed_case_count") != 0:
        raise ValueError("end-to-end report contains fail-closed cases")
    if set(identity.get("benchmarks", [])) != set(DEFAULT_INPUTS):
        raise ValueError("end-to-end report benchmark universe mismatch")
    if identity.get("sample_per_benchmark") != 0:
        raise ValueError("end-to-end report is a sample, not the full suite")
    if identity.get("response_contract_version") != PUBLIC_E2E_RESPONSE_CONTRACT:
        raise ValueError("end-to-end response contract is not the promoted local-slot contract")
    if identity.get("runner_source_sha256") != file_sha256(RUNNER_SOURCE_PATH):
        raise ValueError("end-to-end runner source hash drifted")
    if identity.get("source_sha256") != expected_source_sha256:
        raise ValueError("end-to-end report normalized inputs drifted")

    artifacts = value.get("artifacts", {})
    case_path = _artifact_path(artifacts.get("case_records_path", ""))
    batch_path = _artifact_path(artifacts.get("batch_records_path", ""))
    if file_sha256(case_path) != artifacts.get("case_records_sha256"):
        raise ValueError("end-to-end case-record hash mismatch")
    if file_sha256(batch_path) != artifacts.get("batch_records_sha256"):
        raise ValueError("end-to-end batch-record hash mismatch")
    batch_rows = [
        json.loads(line)
        for line in batch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(batch_rows) != value["batch_count"]:
        raise ValueError("end-to-end batch-record count mismatch")
    batch_case_ids = []
    observed_raw = observed_normalized = observed_slot_retry = 0
    for batch in batch_rows:
        case_ids = batch.get("case_ids")
        results = batch.get("results")
        mapping = batch.get("wire_case_id_mapping")
        if (
            batch.get("schema_version") != "1.1.0"
            or batch.get("valid") is not True
            or batch.get("errors")
            or not isinstance(case_ids, list)
            or not isinstance(results, list)
            or not isinstance(mapping, list)
            or len(results) != len(case_ids)
            or len(mapping) != len(case_ids)
        ):
            raise ValueError("end-to-end batch failed the promoted record contract")
        expected_mapping = [
            {"wire_case_id": f"C{index}", "case_id": case_id}
            for index, case_id in enumerate(case_ids, 1)
        ]
        if mapping != expected_mapping:
            raise ValueError("end-to-end wire/source case mapping mismatch")
        if [result.get("case_id") for result in results] != case_ids:
            raise ValueError("end-to-end restored result order/identity mismatch")
        events = batch.get("normalization_events") or []
        execution_events = batch.get("execution_events") or []
        if not isinstance(events, list) or set(events) - ALLOWED_NORMALIZATION_EVENTS:
            raise ValueError("end-to-end batch contains unknown normalization events")
        if execution_events not in ([], ["retried_invalid_case_slots"]):
            raise ValueError("end-to-end batch contains unknown execution events")
        if batch.get("raw_contract_valid") is True:
            if events or execution_events:
                raise ValueError("raw-valid batch also declares repair execution")
            observed_raw += 1
        elif execution_events:
            if events:
                raise ValueError("slot-retry batch is double-counted as normalized")
            slot_records = batch.get("case_slot_retry_records")
            if (
                not isinstance(slot_records, list)
                or not slot_records
                or any(record.get("valid") is not True for record in slot_records)
            ):
                raise ValueError("slot-retry batch lacks valid retry provenance")
            for record in slot_records:
                slot_events = record.get("normalization_events") or []
                if (
                    not isinstance(slot_events, list)
                    or set(slot_events) - ALLOWED_NORMALIZATION_EVENTS
                ):
                    raise ValueError(
                        "slot-retry record contains unknown normalization events"
                    )
            observed_slot_retry += 1
        else:
            if not events:
                raise ValueError("normalized batch omits normalization event")
            observed_normalized += 1
        batch_case_ids.extend(case_ids)
    if (
        observed_raw != raw_valid_batches
        or observed_normalized != normalized_batches
        or observed_slot_retry != slot_retry_batches
    ):
        raise ValueError("end-to-end batch execution counts disagree with report")
    if len(batch_case_ids) != len(expected_case_ids) or set(batch_case_ids) != expected_case_ids:
        raise ValueError("end-to-end batch case universe mismatch")
    if len(batch_case_ids) != len(set(batch_case_ids)):
        raise ValueError("end-to-end batch case IDs are duplicated")
    rows = [
        json.loads(line)
        for line in case_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [row.get("case_id") for row in rows]
    if len(case_ids) != len(expected_case_ids) or set(case_ids) != expected_case_ids:
        raise ValueError("end-to-end case-record identity mismatch")
    if len(case_ids) != len(set(case_ids)) or any(row.get("fail_closed") for row in rows):
        raise ValueError("end-to-end case records are duplicated or fail-closed")
    provenance = value.get("provenance", {})
    if provenance.get("locally_human_reviewed_public_cases") != 0:
        raise ValueError("end-to-end report misstates local human provenance")
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": file_sha256(path),
        "profile_id": identity.get("profile_id"),
        "case_count": len(case_ids),
        "batch_count": value.get("batch_count"),
        "raw_contract_valid_batch_count": raw_valid_batches,
        "deterministically_normalized_batch_count": normalized_batches,
        "case_slot_retry_batch_count": slot_retry_batches,
        "deterministic_normalization_rate": normalization_rate,
        "case_slot_retry_batch_rate": slot_retry_rate,
        "metrics": value.get("metrics"),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cross_references(record: dict[str, Any]) -> int:
    documents = {document["id"]: document for document in record["documents"]}
    if len(documents) != len(record["documents"]):
        raise ValueError(f"duplicate document ID in {record['id']}")
    evidence_count = 0
    for evidence in record["supporting_evidence"]:
        document = documents.get(evidence["document_id"])
        if document is None:
            raise ValueError(f"unknown evidence document in {record['id']}")
        sentence_index = evidence["sentence_index"]
        if sentence_index >= len(document["sentences"]):
            raise ValueError(f"unknown evidence sentence in {record['id']}")
        if document["sentences"][sentence_index] != evidence["text"]:
            raise ValueError(f"evidence text drift in {record['id']}")
        evidence_count += 1
    for vote in record["annotation_votes"]:
        for evidence in vote["evidence"]:
            document = documents.get(evidence["document_id"])
            if document is None:
                raise ValueError(f"unknown vote evidence document in {record['id']}")
            if evidence["sentence_index"] >= len(document["sentences"]):
                raise ValueError(f"unknown vote evidence sentence in {record['id']}")
    if set(record["distractor_document_ids"]) - documents.keys():
        raise ValueError(f"unknown distractor document in {record['id']}")
    return evidence_count


def audit_file(
    path: Path,
    expected_benchmark_id: str,
    validator: Draft202012Validator,
    global_ids: set[str],
) -> dict[str, Any]:
    source_ids: set[str] = set()
    count = 0
    evidence_count = 0
    vote_count = 0
    answerability: Counter[str] = Counter()
    labels: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            errors = sorted(
                validator.iter_errors(record), key=lambda error: list(error.path)
            )
            if errors:
                raise ValueError(
                    f"{path.name}:{line_number} failed schema: {errors[0].message}"
                )
            if record["benchmark_id"] != expected_benchmark_id:
                raise ValueError(
                    f"{path.name}:{line_number} has benchmark {record['benchmark_id']}"
                )
            if record["id"] in global_ids:
                raise ValueError(f"duplicate suite case ID: {record['id']}")
            global_ids.add(record["id"])
            source_id = record["provenance"]["source_record_id"]
            if source_id in source_ids:
                raise ValueError(
                    f"duplicate {expected_benchmark_id} source ID: {source_id}"
                )
            source_ids.add(source_id)
            evidence_count += validate_cross_references(record)
            vote_count += len(record["annotation_votes"])
            answerability[record["gold"]["answerability"]] += 1
            if record["gold"]["label"] is not None:
                labels[record["gold"]["label"]] += 1
            count += 1

    return {
        "cases": count,
        "unique_source_ids": len(source_ids),
        "evidence_references_resolved": evidence_count,
        "annotation_votes_preserved": vote_count,
        "answerability": dict(sorted(answerability.items())),
        "labels": dict(sorted(labels.items())),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def audit_suite(
    include_ai_spot_audit: bool = True,
    end_to_end_report: Path | None = None,
) -> dict[str, Any]:
    registry = load_registry()
    targets = {
        entry["id"]: entry["target_cases"] for entry in registry["benchmarks"]
    }
    schema = json.loads(
        (ROOT / "benchmarks" / "schemas" / "normalized-case.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema)
    global_ids: set[str] = set()
    benchmarks: dict[str, Any] = {}

    for benchmark_id, path in DEFAULT_INPUTS.items():
        result = audit_file(path, benchmark_id, validator, global_ids)
        if result["cases"] != targets[benchmark_id]:
            raise ValueError(
                f"{benchmark_id} has {result['cases']} cases; expected {targets[benchmark_id]}"
            )
        source_report = json.loads(
            DEFAULT_REPORTS[benchmark_id].read_text(encoding="utf-8")
        )
        if source_report["conversion"]["output_sha256"] != result["sha256"]:
            raise ValueError(f"{benchmark_id} output/report checksum mismatch")
        if source_report["conversion"]["normalized_cases"] != result["cases"]:
            raise ValueError(f"{benchmark_id} output/report count mismatch")
        benchmarks[benchmark_id] = {
            **result,
            "source_manifest_sha256": source_report["source"]["sha256"],
            "conversion_loss_rate": source_report["conversion"][
                "conversion_loss_rate"
            ],
        }

    total = sum(result["cases"] for result in benchmarks.values())
    expected_total = sum(targets.values())
    if total != expected_total:
        raise ValueError(f"suite has {total} cases; expected {expected_total}")
    report = {
        "suite_report_version": "1.1.0",
        "status": "machine_validated_human_spot_audit_pending",
        "public_case_count": total,
        "unique_case_ids": len(global_ids),
        "human_annotation_provenance": "inherited_from_pinned_publishers",
        "locally_human_reviewed_case_count": 0,
        "resume_claim_ready": False,
        "resume_claim_blockers": [
            "stratified adapter spot audit is pending",
            "RAG retrieval/generation evaluation has not yet run across this suite",
        ],
        "benchmarks": benchmarks,
    }
    if include_ai_spot_audit and AI_SPOT_AUDIT.exists():
        from scripts.benchmarks.build_spot_audit import build

        ai_audit = json.loads(AI_SPOT_AUDIT.read_text(encoding="utf-8"))
        expected_audit = build()
        expected_hashes = {
            key: value["sha256"] for key, value in benchmarks.items()
        }
        if ai_audit.get("status") != "ai_model_spot_audit_passed":
            raise ValueError("AI spot audit is not passing")
        if ai_audit.get("normalized_suite", {}).get("benchmark_sha256") != expected_hashes:
            raise ValueError("AI spot audit targets different normalized benchmark files")
        provenance = ai_audit.get("review_provenance", {})
        if (
            provenance.get("reviewer_kind") != "ai_agent"
            or provenance.get("locally_human_reviewed") is not False
        ):
            raise ValueError("AI spot audit has ambiguous or false review provenance")
        expected_cases = {
            row["audit_id"]: (row["case_id"], row["benchmark_id"], row["stratum"])
            for row in expected_audit["cases"]
        }
        decisions = ai_audit.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != 48:
            raise ValueError("AI spot audit must contain exactly 48 decisions")
        if len({row.get("audit_id") for row in decisions}) != 48:
            raise ValueError("AI spot audit decision IDs are missing or duplicated")
        if any(row.get("status") != "approved" for row in decisions):
            raise ValueError("AI spot audit contains a non-approved conversion case")
        actual_cases = {
            row["audit_id"]: (row.get("case_id"), row.get("benchmark_id"), row.get("stratum"))
            for row in decisions
        }
        if actual_cases != expected_cases:
            raise ValueError("AI spot audit decisions do not match the frozen sample")
        report.update(
            {
                "status": "machine_validated_ai_spot_audit_passed",
                "locally_ai_reviewed_case_count": 48,
                "local_review_provenance": "ai_agent_not_human",
                "resume_claim_blockers": [
                    "RAG retrieval/generation evaluation has not yet run across this suite"
                ],
            }
        )
    if end_to_end_report is not None:
        if report["status"] != "machine_validated_ai_spot_audit_passed":
            raise ValueError("end-to-end promotion requires the passing AI spot audit")
        end_to_end = validate_end_to_end(
            end_to_end_report,
            {key: value["sha256"] for key, value in benchmarks.items()},
            global_ids,
        )
        report.update(
            {
                "status": "end_to_end_evaluated",
                "resume_claim_ready": True,
                "resume_claim_blockers": [],
                "end_to_end_evaluation": end_to_end,
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the cross-benchmark integrity audit over all 10K cases."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "public-suite-10000.json",
    )
    parser.add_argument("--end-to-end-report", type=Path)
    args = parser.parse_args()
    report = audit_suite(end_to_end_report=args.end_to_end_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "public suite integrity: "
        f"cases={report['public_case_count']} unique={report['unique_case_ids']} "
        f"status={report['status']}"
    )


if __name__ == "__main__":
    main()
