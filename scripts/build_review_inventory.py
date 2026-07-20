"""Build a deterministic V2 annotation/review coverage inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "evals" / "v2" / "dataset-manifest.json"
DEFAULT_OUT = ROOT / "evals" / "v2" / "review-inventory.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_cases() -> tuple[dict, list[dict]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = []
    for partition in manifest["partitions"]:
        for file_entry in partition["files"]:
            path = ROOT / file_entry["path"]
            cases.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
    return manifest, cases


def ids_where(cases: list[dict], predicate) -> list[str]:
    return sorted(case["id"] for case in cases if predicate(case))


def build_inventory() -> dict:
    manifest, cases = load_cases()
    partitions = {item["split"]: item for item in manifest["partitions"]}
    pending_review = ids_where(
        cases, lambda case: case["review"]["status"] != "approved"
    )
    legacy_claims = ids_where(
        cases,
        lambda case: any(
            claim["status"] == "legacy_unatomized"
            for claim in case["required_claims"]
        ),
    )
    legacy_evidence = ids_where(
        cases, lambda case: case["evidence_granularity"] == "legacy_heading"
    )
    severity_unassigned = ids_where(
        cases, lambda case: case["severity"] == "unassigned"
    )
    independence_unknown = ids_where(
        cases,
        lambda case: case["authoring"]["independent_of_corpus_authors"] is None,
    )
    document_counts = Counter(
        evidence_id.split("::", 1)[0]
        for case in cases
        for evidence_id in case["acceptable_evidence"]
    )
    freeze_checks = {
        "all_cases_approved": not pending_review and bool(cases),
        "all_answerable_claims_atomic": not legacy_claims,
        "all_answerable_evidence_span_level": not legacy_evidence,
        "all_severity_assigned": not severity_unassigned,
        "release_partition_has_cases": partitions["release"]["expected_cases"] > 0,
        "adversarial_partition_has_cases": partitions["adversarial"]["expected_cases"] > 0,
        "real_user_partition_has_cases": partitions["real_user"]["expected_cases"] > 0,
    }
    blockers = [name for name, passed in freeze_checks.items() if not passed]
    return {
        "schema_version": "1.0.0",
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "dataset_fingerprint_sha256": manifest["fingerprint_sha256"],
        "case_count": len(cases),
        "counts": {
            "split": dict(sorted(Counter(case["split"] for case in cases).items())),
            "lane": dict(sorted(Counter(case["lane"] for case in cases).items())),
            "answerability": dict(
                sorted(Counter(case["answerability"] for case in cases).items())
            ),
            "review_status": dict(
                sorted(Counter(case["review"]["status"] for case in cases).items())
            ),
            "evidence_granularity": dict(
                sorted(Counter(case["evidence_granularity"] for case in cases).items())
            ),
            "source_document_labels": dict(sorted(document_counts.items())),
        },
        "independent_units": {
            "intent_families": len({case["intent_family_id"] for case in cases}),
            "evidence_clusters": len({case["evidence_cluster_id"] for case in cases}),
        },
        "queues": {
            "pending_review": pending_review,
            "legacy_unatomized_claims": legacy_claims,
            "legacy_heading_evidence": legacy_evidence,
            "severity_unassigned": severity_unassigned,
            "author_independence_unknown": independence_unknown,
        },
        "freeze_readiness": {
            "ready": all(freeze_checks.values()),
            "checks": freeze_checks,
            "blockers": blockers,
        },
    }


def render(inventory: dict) -> bytes:
    return (json.dumps(inventory, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    inventory = build_inventory()
    expected = render(inventory)
    output = args.out.resolve()
    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(expected)
        action = "wrote"
    else:
        if not output.is_file() or output.read_bytes() != expected:
            raise ValueError(
                "review inventory is missing or stale; regenerate with "
                "python scripts/build_review_inventory.py --write"
            )
        action = "verified"
    readiness = inventory["freeze_readiness"]
    print(f"{action} review inventory for {inventory['case_count']} cases")
    print(f"freeze ready: {readiness['ready']}")
    print(f"blockers: {', '.join(readiness['blockers'])}")
    print(f"output sha256: {sha256_bytes(expected)}")


if __name__ == "__main__":
    main()
