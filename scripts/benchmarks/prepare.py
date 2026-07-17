from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.benchmarks import (  # noqa: E402
    HotpotQAAdapter,
    FeverAdapter,
    NaturalQuestionsAdapter,
    canonical_json_sha256,
    stable_hash_sample,
)
from rag.benchmarks.base import render_jsonl  # noqa: E402

ADAPTERS = {
    "hotpotqa": HotpotQAAdapter,
    "natural_questions": NaturalQuestionsAdapter,
    "fever": FeverAdapter,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry() -> dict[str, Any]:
    registry_path = ROOT / "benchmarks" / "registry.json"
    schema_path = ROOT / "benchmarks" / "registry.schema.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(registry)
    ids = [entry["id"] for entry in registry["benchmarks"]]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark registry contains duplicate IDs")
    return registry


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError(
                "reading parquet requires pyarrow; JSON fixtures do not"
            ) from error
        rows = parquet.read_table(path).to_pylist()
        records: list[dict[str, Any]] = []
        for row in rows:
            context = row["context"]
            supporting = row["supporting_facts"]
            records.append(
                {
                    "_id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "type": row["type"],
                    "level": row["level"],
                    "supporting_facts": [
                        [title, sentence_id]
                        for title, sentence_id in zip(
                            supporting["title"], supporting["sent_id"]
                        )
                    ],
                    "context": [
                        [title, sentences]
                        for title, sentences in zip(
                            context["title"], context["sentences"]
                        )
                    ],
                }
            )
        return records

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("source benchmark must be a JSON array of objects")
    return value


def source_artifacts(path: Path) -> list[dict[str, Any]]:
    files = sorted(file for file in path.iterdir() if file.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no source artifacts found under {path}")
    return [
        {
            "name": file.name,
            "bytes": file.stat().st_size,
            "sha256": file_sha256(file),
        }
        for file in files
    ]


def source_manifest_sha256(artifacts: list[dict[str, Any]]) -> str:
    if len(artifacts) == 1:
        return artifacts[0]["sha256"]
    return canonical_json_sha256(artifacts)


def compact_natural_questions_record(record: dict[str, Any]) -> dict[str, Any]:
    """Retain task semantics while hashing the complete source row first."""
    source_hash = canonical_json_sha256(record)
    document = record["document"]
    tokens = document["tokens"]
    candidates = record["long_answer_candidates"]
    annotations = record["annotations"]
    return {
        "id": record["id"],
        "document": {
            "title": document["title"],
            "url": document["url"],
            "tokens": {
                "token": tokens["token"],
                "is_html": tokens["is_html"],
            },
        },
        "question": {"text": record["question"]["text"]},
        "long_answer_candidates": {
            "start_token": candidates["start_token"],
            "end_token": candidates["end_token"],
            "top_level": candidates["top_level"],
        },
        "annotations": {
            "id": annotations["id"],
            "long_answer": [
                {"candidate_index": answer["candidate_index"]}
                for answer in annotations["long_answer"]
            ],
            "short_answers": [
                {"text": answer["text"]} for answer in annotations["short_answers"]
            ],
            "yes_no_answer": annotations["yes_no_answer"],
        },
        "_source_record_sha256": source_hash,
    }


def fever_sentences(lines: str, page_id: str) -> list[str]:
    indexed: dict[int, str] = {}
    for raw_line in lines.splitlines():
        fields = raw_line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"FEVER page {page_id} has malformed sentence data")
        try:
            sentence_id = int(fields[0])
        except ValueError as error:
            raise ValueError(
                f"FEVER page {page_id} has non-integer sentence ID"
            ) from error
        if sentence_id in indexed:
            raise ValueError(f"FEVER page {page_id} has duplicate sentence ID")
        indexed[sentence_id] = fields[1]
    if not indexed:
        return []
    sentences = ["" for _ in range(max(indexed) + 1)]
    for sentence_id, text in indexed.items():
        sentences[sentence_id] = text
    return sentences


def load_fever_source(
    path: Path, limit: int | None
) -> tuple[list[dict[str, Any]], int, bool, dict[str, Any]]:
    import unicodedata
    import zipfile

    claims_path = path / "shared_task_dev.jsonl"
    wiki_path = path / "wiki-pages.zip"
    claims = [
        json.loads(line)
        for line in claims_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    ranked = stable_hash_sample(claims, None, "id")
    desired = len(ranked) if limit is None else limit
    pool_size = min(len(ranked), desired + max(100, desired // 20))
    pool = ranked[:pool_size]
    needed_pages = {
        item[2]
        for claim in pool
        for evidence_set in claim["evidence"]
        for item in evidence_set
        if item[2] is not None
    }

    cache_key = canonical_json_sha256(
        {
            "wiki_sha256": file_sha256(wiki_path),
            "pages": sorted(needed_pages),
            "title_normalization": "NFC-v1",
        }
    )
    cache_path = path / "cache" / f"selected-pages-{cache_key}.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        pages = cache["pages"]
    else:
        pages: dict[str, dict[str, Any]] = {}
        normalized_requests: dict[str, list[str]] = {}
        for requested_page in needed_pages:
            normalized_requests.setdefault(
                unicodedata.normalize("NFC", requested_page), []
            ).append(requested_page)
        with zipfile.ZipFile(wiki_path) as archive:
            entries = sorted(
                name
                for name in archive.namelist()
                if name.startswith("wiki-pages/") and name.endswith(".jsonl")
            )
            for name in entries:
                with archive.open(name) as handle:
                    for raw_line in handle:
                        page = json.loads(raw_line)
                        page_id = page["id"]
                        requested_ids = normalized_requests.get(
                            unicodedata.normalize("NFC", page_id), []
                        )
                        for requested_id in requested_ids:
                            pages[requested_id] = {
                                "sentences": fever_sentences(
                                    page["lines"], requested_id
                                )
                            }
                if len(pages) == len(needed_pages):
                    break
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "pages": pages,
                    "missing_pages": sorted(needed_pages - pages.keys()),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    considered = 0
    for claim in pool:
        considered += 1
        claim_pages = {
            item[2]
            for evidence_set in claim["evidence"]
            for item in evidence_set
            if item[2] is not None
        }
        missing_claim_pages = sorted(claim_pages - pages.keys())
        if missing_claim_pages:
            excluded.append(
                {
                    "source_record_id": str(claim["id"]),
                    "reason": "referenced_page_missing_from_official_wikipedia_dump",
                    "missing_pages": missing_claim_pages,
                }
            )
            continue
        selected.append(claim)
        if len(selected) == desired:
            break
    if len(selected) != desired:
        raise ValueError(
            f"FEVER selection produced {len(selected)} valid cases; expected {desired}"
        )

    records: list[dict[str, Any]] = []
    for claim in selected:
        source_hash = canonical_json_sha256(claim)
        page_ids = {
            item[2]
            for evidence_set in claim["evidence"]
            for item in evidence_set
            if item[2] is not None
        }
        records.append(
            {
                **claim,
                "_source_record_sha256": source_hash,
                "_pages": {page_id: pages[page_id] for page_id in page_ids},
            }
        )
    return (
        records,
        len(claims),
        True,
        {
            "method": (
                "sha256(source_record_id), ascending; NFC-normalized page-title "
                "resolution; skip and report only claims whose evidence pages "
                "remain absent from the pinned official corpus"
            ),
            "considered_cases": considered,
            "excluded_cases": len(excluded),
            "exclusions": excluded,
        },
    )


def load_source(
    path: Path, benchmark_id: str, limit: int | None
) -> tuple[list[dict[str, Any]], int, bool, dict[str, Any]]:
    """Load a source, preselecting large sharded NQ data before Python expansion."""
    if not path.is_dir():
        records = load_records(path)
        return records, len(records), False, {
            "method": "sha256(source_record_id), ascending",
            "considered_cases": len(records) if limit is None else min(limit, len(records)),
            "excluded_cases": 0,
            "exclusions": [],
        }
    if benchmark_id == "fever":
        return load_fever_source(path, limit)
    if benchmark_id != "natural_questions":
        raise ValueError(f"directory input is not implemented for {benchmark_id}")

    import pyarrow as arrow
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    files = sorted(path.glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet shards found under {path}")
    source_ids: list[str] = []
    for file in files:
        source_ids.extend(parquet.read_table(file, columns=["id"])["id"].to_pylist())
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("sharded source contains duplicate IDs")
    selected = stable_hash_sample(
        [{"id": source_id} for source_id in source_ids], limit, "id"
    )
    selected_ids = {record["id"] for record in selected}

    records: list[dict[str, Any]] = []
    value_set = arrow.array(sorted(selected_ids))
    for file in files:
        table = parquet.read_table(file)
        table = table.filter(compute.is_in(table["id"], value_set=value_set))
        records.extend(compact_natural_questions_record(row) for row in table.to_pylist())
    records = list(stable_hash_sample(records, None, "id"))
    if len(records) != len(selected_ids):
        raise ValueError(
            f"selected {len(selected_ids)} Natural Questions IDs but loaded {len(records)}"
        )
    return records, len(source_ids), True, {
        "method": "sha256(source_record_id), ascending",
        "considered_cases": len(records),
        "excluded_cases": 0,
        "exclusions": [],
    }


def validate_normalized(records: list[dict[str, Any]]) -> dict[str, int]:
    schema = json.loads(
        (ROOT / "benchmarks" / "schemas" / "normalized-case.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema)
    normalized_ids: set[str] = set()
    source_ids: set[str] = set()
    evidence_references = 0

    for position, record in enumerate(records):
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        if errors:
            raise ValueError(
                f"normalized record {position} failed schema: {errors[0].message}"
            )
        if record["id"] in normalized_ids:
            raise ValueError(f"duplicate normalized ID: {record['id']}")
        normalized_ids.add(record["id"])
        source_id = record["provenance"]["source_record_id"]
        if source_id in source_ids:
            raise ValueError(f"duplicate source ID after normalization: {source_id}")
        source_ids.add(source_id)

        documents = {document["id"]: document for document in record["documents"]}
        if len(documents) != len(record["documents"]):
            raise ValueError(f"duplicate document ID in {record['id']}")
        for evidence in record["supporting_evidence"]:
            document = documents.get(evidence["document_id"])
            if document is None:
                raise ValueError(f"unknown evidence document in {record['id']}")
            sentence_index = evidence["sentence_index"]
            if sentence_index >= len(document["sentences"]):
                raise ValueError(f"unknown evidence sentence in {record['id']}")
            if document["sentences"][sentence_index] != evidence["text"]:
                raise ValueError(f"evidence text drift in {record['id']}")
            evidence_references += 1
        for vote in record["annotation_votes"]:
            for evidence in vote["evidence"]:
                document = documents.get(evidence["document_id"])
                if document is None:
                    raise ValueError(f"unknown vote evidence document in {record['id']}")
                if evidence["sentence_index"] >= len(document["sentences"]):
                    raise ValueError(f"unknown vote evidence sentence in {record['id']}")
            if vote["answerability"] == "unanswerable" and vote["evidence"]:
                raise ValueError(f"unanswerable vote contains evidence in {record['id']}")
        if set(record["distractor_document_ids"]) - documents.keys():
            raise ValueError(f"unknown distractor document in {record['id']}")
        if record["gold"]["answerability"] == "answerable" and not record["supporting_evidence"]:
            raise ValueError(f"answerable case has no supporting evidence in {record['id']}")

    return {
        "normalized_cases": len(records),
        "unique_normalized_ids": len(normalized_ids),
        "unique_source_ids": len(source_ids),
        "evidence_references_resolved": evidence_references,
    }


def build_report(
    benchmark: dict[str, Any],
    source_path: Path,
    source_record_count: int,
    artifacts: list[dict[str, Any]],
    selection_info: dict[str, Any],
    normalized: list[dict[str, Any]],
    output_bytes: bytes,
) -> dict[str, Any]:
    validation = validate_normalized(normalized)
    return {
        "report_version": "1.0.0",
        "benchmark_id": benchmark["id"],
        "source": {
            "version": benchmark["source_version"],
            "split": benchmark["source_split"],
            "url": benchmark["source_url"],
            "upstream_url": benchmark["upstream_source_url"],
            "file": source_path.name,
            "sha256": source_manifest_sha256(artifacts),
            "records": source_record_count,
            "artifacts": artifacts,
            "license": benchmark["license"],
        },
        "selection": {
            "method": selection_info["method"],
            "selected_cases": len(normalized),
            "target_cases": benchmark["target_cases"],
            "considered_cases": selection_info["considered_cases"],
        },
        "conversion": {
            **validation,
            "excluded_cases": selection_info["excluded_cases"],
            "conversion_loss_rate": (
                selection_info["excluded_cases"]
                / selection_info["considered_cases"]
                if selection_info["considered_cases"]
                else 0.0
            ),
            "exclusions": selection_info["exclusions"],
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        },
        "distribution": {
            "answerability": dict(
                sorted(Counter(case["gold"]["answerability"] for case in normalized).items())
            ),
            "question_type": dict(
                sorted(
                    Counter(
                        value
                        for case in normalized
                        if (value := case["metadata"].get("question_type")) is not None
                    ).items()
                )
            ),
            "difficulty": dict(
                sorted(
                    Counter(
                        value
                        for case in normalized
                        if (value := case["metadata"].get("difficulty")) is not None
                    ).items()
                )
            ),
            "label": dict(
                sorted(
                    Counter(
                        value
                        for case in normalized
                        if (value := case["gold"].get("label")) is not None
                    ).items()
                )
            ),
        },
        "annotation_provenance": benchmark["human_annotation_provenance"],
        "manual_audit": {
            "status": "pending",
            "note": "Adapter validation is complete; a stratified human spot audit is still required before publication.",
        },
    }


def prepare(
    benchmark_id: str,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    limit: int | None,
    *,
    verify_source_checksum: bool = True,
) -> dict[str, Any]:
    registry = load_registry()
    benchmark = next(
        (entry for entry in registry["benchmarks"] if entry["id"] == benchmark_id),
        None,
    )
    if benchmark is None:
        raise ValueError(f"benchmark not registered: {benchmark_id}")
    if benchmark_id not in ADAPTERS:
        raise ValueError(f"benchmark adapter not implemented: {benchmark_id}")

    source_records, source_record_count, preselected, selection_info = load_source(
        input_path, benchmark_id, limit
    )
    artifacts = source_artifacts(input_path)
    raw_sha256 = source_manifest_sha256(artifacts)
    if (
        verify_source_checksum
        and benchmark["raw_sha256"]
        and raw_sha256 != benchmark["raw_sha256"]
    ):
        raise ValueError(
            f"raw checksum mismatch: expected {benchmark['raw_sha256']}, got {raw_sha256}"
        )

    adapter = ADAPTERS[benchmark_id]()
    normalized = adapter.normalize_many(
        source_records,
        source_split=benchmark["source_split"],
        limit=None if preselected else limit,
    )
    output_bytes = render_jsonl(normalized)
    report = build_report(
        benchmark,
        input_path,
        source_record_count,
        artifacts,
        selection_info,
        normalized,
        output_bytes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output_bytes)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and validate an official public benchmark artifact."
    )
    parser.add_argument("--benchmark", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare(
        args.benchmark, args.input, args.output, args.report, args.limit
    )
    print(
        "public benchmark validated: "
        f"{report['benchmark_id']} cases={report['conversion']['normalized_cases']} "
        f"evidence={report['conversion']['evidence_references_resolved']} "
        f"loss={report['conversion']['conversion_loss_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
