"""Validate corpus and V2 evaluation data contracts without model/API access."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import holdout_ledger  # noqa: E402

EXPECTED_SPLITS = {"dev", "release", "adversarial", "real_user", "reserve"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_schema(instance: object, schema_path: Path, label: str) -> None:
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        rendered = []
        for error in errors:
            location = ".".join(str(part) for part in error.absolute_path) or "<root>"
            rendered.append(f"{label}:{location}: {error.message}")
        raise ValueError("\n".join(rendered))


def slug(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def heading_keys(path: Path, document_id: str) -> set[str]:
    keys: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            continue
        match = None if in_fence else re.match(r"^#{1,4}\s+(.*)$", line)
        if match:
            keys.add(f"{document_id}::{slug(match.group(1).strip())}")
    return keys


def validate_corpus() -> tuple[dict, set[str], set[str]]:
    manifest_path = ROOT / "corpus" / "manifest.json"
    manifest = load_json(manifest_path)
    validate_schema(
        manifest, ROOT / "corpus" / "manifest.schema.json", "corpus/manifest.json"
    )

    ids: set[str] = set()
    paths: set[str] = set()
    content_hashes: set[str] = set()
    evidence: set[str] = set()
    for document in manifest["documents"]:
        document_id = document["document_id"]
        relative = document["path"]
        if document_id in ids:
            raise ValueError(f"duplicate document_id: {document_id}")
        if relative in paths:
            raise ValueError(f"duplicate corpus path: {relative}")
        ids.add(document_id)
        paths.add(relative)

        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError(f"invalid corpus path: {relative}")
        digest = sha256_file(path)
        if digest != document["content_sha256"]:
            raise ValueError(f"content hash mismatch: {relative}")
        if digest in content_hashes:
            raise ValueError(f"duplicate corpus content hash: {relative}")
        content_hashes.add(digest)
        text = path.read_text(encoding="utf-8")
        if len(text.splitlines()) != document["line_count"]:
            raise ValueError(f"line count mismatch: {relative}")
        if len(text.split()) != document["word_count"]:
            raise ValueError(f"word count mismatch: {relative}")
        evidence.update(heading_keys(path, document_id))

    documents_by_id = {item["document_id"]: item for item in manifest["documents"]}
    for document in manifest["documents"]:
        authority = document["authority"]
        unknown = (set(authority["supersedes"]) | {authority["superseded_by"]}) - (
            ids | {None}
        )
        if unknown:
            raise ValueError(
                f"unknown authority reference for {document['document_id']}: "
                f"{sorted(unknown)}"
            )
        if (
            authority["valid_from"] is not None
            and authority["valid_until"] is not None
            and authority["valid_until"] < authority["valid_from"]
        ):
            raise ValueError(f"invalid authority interval: {document['document_id']}")
        if document["status"] in {"deleted", "superseded"} and document["ingestion"]["allowed"]:
            raise ValueError(
                f"inactive document cannot be ingestible: {document['document_id']}"
            )

    disk_markdown = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "corpus").glob("*.md")
    }
    if paths != disk_markdown:
        raise ValueError(
            f"corpus manifest/file mismatch: declared={sorted(paths)}, "
            f"on_disk={sorted(disk_markdown)}"
        )
    fingerprint = canonical_json_sha256(
        sorted(manifest["documents"], key=lambda item: item["document_id"])
    )
    if fingerprint != manifest["fingerprint_sha256"]:
        raise ValueError("corpus fingerprint mismatch")

    catalog_meta = manifest["evidence_catalog"]
    catalog_path = ROOT / catalog_meta["path"]
    if sha256_file(catalog_path) != catalog_meta["sha256"]:
        raise ValueError("evidence catalog hash mismatch")
    catalog_schema = load_json(ROOT / catalog_meta["schema"])
    Draft202012Validator.check_schema(catalog_schema)
    catalog_validator = Draft202012Validator(catalog_schema)
    span_ids: set[str] = set()
    catalog_lines = [
        line for line in catalog_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(catalog_lines) != catalog_meta["records"]:
        raise ValueError("evidence catalog record count mismatch")
    for line_number, line in enumerate(catalog_lines, 1):
        record = json.loads(line)
        errors = list(catalog_validator.iter_errors(record))
        if errors:
            raise ValueError(
                f"{catalog_meta['path']}:{line_number}: {errors[0].message}"
            )
        evidence_id = record["evidence_id"]
        if evidence_id in span_ids:
            raise ValueError(f"duplicate span evidence ID: {evidence_id}")
        span_ids.add(evidence_id)
        document = documents_by_id.get(record["document_id"])
        if document is None or record["document_sha256"] != document["content_sha256"]:
            raise ValueError(f"span document mismatch: {evidence_id}")
        if record["heading_key"] not in evidence:
            raise ValueError(f"span heading mismatch: {evidence_id}")
        if record["end_line"] < record["start_line"]:
            raise ValueError(f"span line range invalid: {evidence_id}")
        if sha256_bytes(record["text"].encode("utf-8")) != record["content_sha256"]:
            raise ValueError(f"span content hash mismatch: {evidence_id}")
    return manifest, evidence, span_ids


def source_v1_records(path: Path) -> tuple[str, dict[str, tuple[dict, str]]]:
    raw = path.read_bytes()
    records = {}
    for line in (line for line in raw.splitlines() if line.strip()):
        record = json.loads(line)
        records[record["id"]] = (record, sha256_bytes(line))
    return sha256_bytes(raw), records


def validate_case_integrity(
    case: dict,
    valid_legacy_evidence: set[str],
    valid_span_evidence: set[str],
) -> None:
    acceptable = set(case["acceptable_evidence"])
    distractors = set(case["distractor_evidence"])
    if acceptable & distractors:
        raise ValueError(f"evidence also labelled distractor: {case['id']}")

    claim_evidence = {
        evidence_id
        for claim in case["required_claims"]
        for evidence_id in claim["evidence_ids"]
    }
    if not claim_evidence.issubset(acceptable):
        raise ValueError(f"claim evidence not acceptable: {case['id']}")
    claim_ids = [claim["id"] for claim in case["required_claims"]]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError(f"duplicate claim ID: {case['id']}")

    unknown_distractors = distractors - (
        valid_legacy_evidence | valid_span_evidence
    )
    if unknown_distractors:
        raise ValueError(
            f"unknown distractor evidence for {case['id']}: "
            f"{sorted(unknown_distractors)}"
        )
    if case["evidence_granularity"] == "legacy_heading":
        missing = acceptable - valid_legacy_evidence
        if missing:
            raise ValueError(
                f"unknown legacy heading evidence for {case['id']}: {sorted(missing)}"
            )
    elif case["evidence_granularity"] == "span":
        missing = acceptable - valid_span_evidence
        if missing:
            raise ValueError(
                f"unknown span evidence for {case['id']}: {sorted(missing)}"
            )
        non_span_distractors = distractors - valid_span_evidence
        if non_span_distractors:
            raise ValueError(
                f"non-span distractor evidence for {case['id']}: "
                f"{sorted(non_span_distractors)}"
            )

    action_reason = {
        "abstain_absent": {"absent_from_corpus", "subjective"},
        "clarify_ambiguous": {"ambiguous", "malformed"},
        "abstain_conflict": {"conflicting_sources"},
        "abstain_obsolete": {"obsolete_only"},
        "abstain_unauthorized": {"unauthorized"},
        "reject_injection": {"prompt_injection"},
    }
    if case["answerability"] == "unanswerable":
        allowed_reasons = action_reason[case["answer_action"]]
        if case["unanswerable"]["reason"] not in allowed_reasons:
            raise ValueError(f"answer action/reason mismatch: {case['id']}")

    has_conversation = case["conversation_id"] is not None
    has_turn = case["turn_index"] is not None
    if has_conversation != has_turn:
        raise ValueError(f"incomplete conversation identity: {case['id']}")

    review = case["review"]
    if review["status"] == "approved":
        if not review["reviewers"] or review["reviewed_at"] is None:
            raise ValueError(f"approved case lacks review audit: {case['id']}")
        if case["severity"] == "unassigned":
            raise ValueError(f"approved case lacks severity: {case['id']}")
        if case["authoring"]["method"] == "model_proposed_ai_verified":
            if review.get("reviewer_kinds") != ["ai_agent"]:
                raise ValueError(
                    f"AI-reviewed case has ambiguous reviewer provenance: {case['id']}"
                )
        if case["answerability"] == "answerable" and (
            case["evidence_granularity"] != "span"
            or any(
                claim["status"] != "atomic_verified"
                for claim in case["required_claims"]
            )
        ):
            raise ValueError(
                f"approved answerable case lacks atomic span labels: {case['id']}"
            )


def validate_dataset(
    corpus_manifest: dict,
    valid_legacy_evidence: set[str],
    valid_span_evidence: set[str],
) -> tuple[dict, list[dict]]:
    manifest_path = ROOT / "evals" / "v2" / "dataset-manifest.json"
    manifest = load_json(manifest_path)
    validate_schema(
        manifest,
        ROOT / "evals" / "schema" / "dataset-manifest.schema.json",
        "evals/v2/dataset-manifest.json",
    )
    case_schema = ROOT / manifest["case_schema"]
    schema = load_json(case_schema)
    Draft202012Validator.check_schema(schema)
    case_validator = Draft202012Validator(schema, format_checker=FormatChecker())

    expected_corpus = {
        "path": "corpus/manifest.json",
        "corpus_id": corpus_manifest["corpus_id"],
        "corpus_version": corpus_manifest["corpus_version"],
        "fingerprint_sha256": corpus_manifest["fingerprint_sha256"],
        "evidence_catalog_sha256": corpus_manifest["evidence_catalog"]["sha256"],
    }
    if manifest["corpus_manifest"] != expected_corpus:
        raise ValueError("dataset/corpus manifest fingerprint mismatch")

    partitions = manifest["partitions"]
    allowed_case_versions = set(manifest["case_record_versions"])
    if manifest["dataset_version"] not in allowed_case_versions:
        raise ValueError("current dataset version is absent from case_record_versions")
    split_names = [partition["split"] for partition in partitions]
    if set(split_names) != EXPECTED_SPLITS or len(split_names) != len(EXPECTED_SPLITS):
        raise ValueError("dataset manifest must contain every split exactly once")

    source_path = ROOT / "evals" / "golden.jsonl"
    v1_sha, v1_records = source_v1_records(source_path)
    source_entry = next(
        (item for item in manifest["source_datasets"] if item["path"] == "evals/golden.jsonl"),
        None,
    )
    if source_entry is None or source_entry["sha256"] != v1_sha:
        raise ValueError("V1 source dataset fingerprint mismatch")

    all_cases: list[dict] = []
    case_ids: set[str] = set()
    question_texts: set[str] = set()
    fingerprint_rows = []
    for partition in partitions:
        split_count = 0
        for file_entry in partition["files"]:
            relative = file_entry["path"]
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
                raise ValueError(f"invalid dataset path: {relative}")
            digest = sha256_file(path)
            if digest != file_entry["sha256"]:
                raise ValueError(f"dataset file hash mismatch: {relative}")
            fingerprint_rows.append(f"{partition['split']}\t{relative}\t{digest}\n")
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            if len(lines) != file_entry["cases"]:
                raise ValueError(f"dataset case count mismatch: {relative}")
            split_count += len(lines)
            for line_number, line in enumerate(lines, 1):
                case = json.loads(line)
                errors = sorted(
                    case_validator.iter_errors(case), key=lambda error: list(error.path)
                )
                if errors:
                    location = ".".join(str(part) for part in errors[0].absolute_path)
                    raise ValueError(
                        f"{relative}:{line_number}:{location}: {errors[0].message}"
                    )
                if case["dataset_version"] not in allowed_case_versions:
                    raise ValueError(f"dataset version mismatch: {case['id']}")
                if case["split"] != partition["split"]:
                    raise ValueError(f"split mismatch: {case['id']}")
                if case["id"] in case_ids:
                    raise ValueError(f"duplicate case ID: {case['id']}")
                if case["question"] in question_texts:
                    raise ValueError(f"duplicate exact question: {case['id']}")
                case_ids.add(case["id"])
                question_texts.add(case["question"])

                validate_case_integrity(
                    case, valid_legacy_evidence, valid_span_evidence
                )

                legacy = case.get("legacy")
                if legacy:
                    original_id = legacy["v1_record"]["id"]
                    original = v1_records.get(original_id)
                    if original is None:
                        raise ValueError(f"missing V1 source record: {case['id']}")
                    if legacy["v1_record"] != original[0]:
                        raise ValueError(f"V1 round-trip mismatch: {case['id']}")
                    if legacy["source_record_sha256"] != original[1]:
                        raise ValueError(f"V1 record hash mismatch: {case['id']}")
                    if legacy["source_dataset_sha256"] != v1_sha:
                        raise ValueError(f"V1 dataset hash mismatch: {case['id']}")
                all_cases.append(case)
        if split_count != partition["expected_cases"]:
            raise ValueError(f"partition case count mismatch: {partition['split']}")

    fingerprint = sha256_bytes("".join(sorted(fingerprint_rows)).encode("utf-8"))
    if fingerprint != manifest["fingerprint_sha256"]:
        raise ValueError("dataset fingerprint mismatch")
    return manifest, all_cases


def validate_derived_artifacts(corpus: dict, dataset: dict, cases: list[dict]) -> dict:
    coverage_path = ROOT / "evals" / "v2" / "coverage-matrix.json"
    coverage = load_json(coverage_path)
    validate_schema(
        coverage,
        ROOT / "evals" / "schema" / "coverage-matrix.schema.json",
        "evals/v2/coverage-matrix.json",
    )
    source_map_path = ROOT / "evals" / "v2" / "v1-review-source-map.json"
    source_map = load_json(source_map_path)
    validate_schema(
        source_map,
        ROOT / "evals" / "schema" / "v1-review-source-map.schema.json",
        "evals/v2/v1-review-source-map.json",
    )

    expected_fingerprints = {
        "dataset_fingerprint_sha256": dataset["fingerprint_sha256"],
        "corpus_fingerprint_sha256": corpus["fingerprint_sha256"],
        "evidence_catalog_sha256": corpus["evidence_catalog"]["sha256"],
    }
    for artifact_name, artifact in (("coverage", coverage), ("review source map", source_map)):
        for field, expected in expected_fingerprints.items():
            if artifact[field] != expected:
                raise ValueError(f"{artifact_name} {field} mismatch")

    if coverage["summary"]["cases"] != len(cases):
        raise ValueError("coverage matrix case count mismatch")
    legacy_ids = {case["id"] for case in cases if case.get("legacy")}
    source_map_ids = {case["case_id"] for case in source_map["cases"]}
    if source_map_ids != legacy_ids or source_map["case_count"] != len(legacy_ids):
        raise ValueError("V1 review source-map case identity mismatch")
    if any(not case["human_decision_required"] for case in source_map["cases"]):
        raise ValueError("V1 review source map contains a non-human decision")

    ledger_events = holdout_ledger.validate_events()
    return {
        "coverage_lanes": coverage["summary"]["target_lanes"],
        "coverage_seeded_lanes": coverage["summary"]["lanes_with_approved_seed"],
        "review_source_cases": source_map["case_count"],
        "holdout_ledger_events": len(ledger_events),
    }


def validate_all() -> dict:
    corpus, legacy_evidence, span_evidence = validate_corpus()
    dataset, cases = validate_dataset(corpus, legacy_evidence, span_evidence)
    derived = validate_derived_artifacts(corpus, dataset, cases)
    splits = Counter(case["split"] for case in cases)
    reviews = Counter(case["review"]["status"] for case in cases)
    return {
        "corpus_version": corpus["corpus_version"],
        "documents": len(corpus["documents"]),
        "legacy_heading_ids": len(legacy_evidence),
        "evidence_spans": len(span_evidence),
        "dataset_version": dataset["dataset_version"],
        "cases": len(cases),
        "splits": dict(sorted(splits.items())),
        "review_status": dict(sorted(reviews.items())),
        **derived,
    }


def main() -> None:
    summary = validate_all()
    print("data contracts: OK")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
