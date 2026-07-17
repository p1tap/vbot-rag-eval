"""Build deterministic corpus/evaluation coverage for Phase 1 planning.

The artifact is descriptive. A seed target proves that a lane's authoring,
fixture, scoring, and review path exists; it is not a statistical sample-size
claim.

Run:
  python scripts/build_coverage_matrix.py --write
  python scripts/build_coverage_matrix.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_MANIFEST = ROOT / "corpus" / "manifest.json"
DATASET_MANIFEST = ROOT / "evals" / "v2" / "dataset-manifest.json"
DEFAULT_OUT = ROOT / "evals" / "v2" / "coverage-matrix.json"
SEED_MINIMUM = 3
TARGET_LANES = (
    "single_hop",
    "exact_identifier",
    "multi_section",
    "multi_document",
    "comparison_aggregation",
    "unanswerable",
    "false_premise",
    "hard_negative",
    "temporal_conflict",
    "structured_content",
    "noisy_user",
    "prompt_injection",
    "multi_turn",
    "authorization",
    "global_corpus",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def load_cases(dataset: dict) -> list[dict]:
    cases = []
    for partition in dataset["partitions"]:
        for entry in partition["files"]:
            cases.extend(load_jsonl(ROOT / entry["path"]))
    return cases


def build_matrix() -> dict:
    corpus = load_json(CORPUS_MANIFEST)
    dataset = load_json(DATASET_MANIFEST)
    catalog = load_jsonl(ROOT / corpus["evidence_catalog"]["path"])
    cases = load_cases(dataset)

    catalog_by_id = {record["evidence_id"]: record for record in catalog}
    heading_records: dict[str, list[dict]] = defaultdict(list)
    for record in catalog:
        heading_records[record["heading_key"]].append(record)

    case_headings: dict[str, set[str]] = {}
    for case in cases:
        headings = set()
        for evidence_id in case["acceptable_evidence"]:
            record = catalog_by_id.get(evidence_id)
            headings.add(record["heading_key"] if record else evidence_id)
        case_headings[case["id"]] = headings

    documents = []
    for document in sorted(corpus["documents"], key=lambda item: item["document_id"]):
        document_id = document["document_id"]
        document_records = [
            record for record in catalog if record["document_id"] == document_id
        ]
        headings = []
        for heading_key in sorted(
            key for key in heading_records if key.startswith(f"{document_id}::")
        ):
            records = heading_records[heading_key]
            linked = sorted(
                case["id"]
                for case in cases
                if heading_key in case_headings[case["id"]]
            )
            approved = sorted(
                case["id"]
                for case in cases
                if heading_key in case_headings[case["id"]]
                and case["review"]["status"] == "approved"
            )
            span_labelled = sorted(
                case["id"]
                for case in cases
                if heading_key in case_headings[case["id"]]
                and case["evidence_granularity"] == "span"
            )
            headings.append(
                {
                    "heading_key": heading_key,
                    "heading_path": records[0]["heading_path"],
                    "evidence_blocks": len(records),
                    "block_types": dict(
                        sorted(Counter(record["block_type"] for record in records).items())
                    ),
                    "linked_case_ids": linked,
                    "approved_case_ids": approved,
                    "span_labelled_case_ids": span_labelled,
                    "has_current_case": bool(linked),
                }
            )
        documents.append(
            {
                "document_id": document_id,
                "status": document["status"],
                "access_class": document["ingestion"]["access_class"],
                "evidence_blocks": len(document_records),
                "block_types": dict(
                    sorted(
                        Counter(record["block_type"] for record in document_records).items()
                    )
                ),
                "headings": headings,
                "headings_total": len(headings),
                "headings_with_current_cases": sum(
                    heading["has_current_case"] for heading in headings
                ),
                "headings_without_current_cases": [
                    heading["heading_key"]
                    for heading in headings
                    if not heading["has_current_case"]
                ],
            }
        )

    lane_counts = Counter(case["lane"] for case in cases)
    approved_lane_counts = Counter(
        case["lane"] for case in cases if case["review"]["status"] == "approved"
    )
    lanes = []
    for lane in TARGET_LANES:
        current = lane_counts[lane]
        approved = approved_lane_counts[lane]
        lanes.append(
            {
                "lane": lane,
                "provisional_seed_minimum": SEED_MINIMUM,
                "current_cases": current,
                "approved_cases": approved,
                "approved_seed_gap": max(0, SEED_MINIMUM - approved),
                "status": "seed_met" if approved >= SEED_MINIMUM else "missing_seed",
            }
        )

    return {
        "schema_version": "1.0.0",
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["dataset_version"],
        "dataset_fingerprint_sha256": dataset["fingerprint_sha256"],
        "corpus_id": corpus["corpus_id"],
        "corpus_version": corpus["corpus_version"],
        "corpus_fingerprint_sha256": corpus["fingerprint_sha256"],
        "evidence_catalog_sha256": corpus["evidence_catalog"]["sha256"],
        "summary": {
            "documents": len(documents),
            "evidence_blocks": len(catalog),
            "headings": sum(document["headings_total"] for document in documents),
            "headings_with_current_cases": sum(
                document["headings_with_current_cases"] for document in documents
            ),
            "cases": len(cases),
            "approved_cases": sum(
                case["review"]["status"] == "approved" for case in cases
            ),
            "span_labelled_answerable_cases": sum(
                case["answerability"] == "answerable"
                and case["evidence_granularity"] == "span"
                for case in cases
            ),
            "target_lanes": len(TARGET_LANES),
            "lanes_with_approved_seed": sum(
                lane["status"] == "seed_met" for lane in lanes
            ),
            "provisional_total_seed_gap": sum(
                lane["approved_seed_gap"] for lane in lanes
            ),
        },
        "documents": documents,
        "lanes": lanes,
        "legacy_lane_counts": dict(sorted(lane_counts.items())),
        "limitations": [
            "A three-case lane seed exercises workflow only; it is not a statistical target.",
            "Legacy intent-family IDs have not been adjudicated for independence.",
            "Heading coverage does not imply span sufficiency or answer quality.",
            "Model-proposed candidates do not count until human approval.",
        ],
    }


def render(matrix: dict) -> bytes:
    return (json.dumps(matrix, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    expected = render(build_matrix())
    output = args.out.resolve()
    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(expected)
        action = "wrote"
    else:
        if not output.is_file() or output.read_bytes() != expected:
            raise ValueError(
                "coverage matrix is missing or stale; regenerate with "
                "python scripts/build_coverage_matrix.py --write"
            )
        action = "verified"
    matrix = json.loads(expected)
    print(f"{action} coverage matrix for {matrix['summary']['cases']} cases")
    print(
        "heading coverage: "
        f"{matrix['summary']['headings_with_current_cases']}/"
        f"{matrix['summary']['headings']}"
    )
    print(
        "approved lane seeds: "
        f"{matrix['summary']['lanes_with_approved_seed']}/"
        f"{matrix['summary']['target_lanes']}"
    )
    print(f"output sha256: {sha256_bytes(expected)}")


if __name__ == "__main__":
    main()

