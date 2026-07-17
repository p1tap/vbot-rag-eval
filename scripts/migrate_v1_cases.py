"""Losslessly project the visible V1 golden set into the V2 development schema.

The migration never upgrades legacy labels to span-level or approved V2 labels.
Every original V1 record is embedded verbatim so the transformation is
reversible and auditable.

Run:
  python scripts/migrate_v1_cases.py --write
  python scripts/migrate_v1_cases.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "evals" / "golden.jsonl"
DEFAULT_OUT = ROOT / "evals" / "v2" / "dev" / "v1-migrated.jsonl"
SOURCE_SHA256 = "c65ca56a5bf07834d72eed68614f0debd1625a2a12ad6e6ece42e345acd77008"
DATASET_VERSION = "2.0.0-dev.1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_v1() -> tuple[list[dict], list[bytes]]:
    raw = SOURCE.read_bytes()
    actual = sha256_bytes(raw)
    if actual != SOURCE_SHA256:
        raise ValueError(
            f"V1 source hash changed: expected {SOURCE_SHA256}, got {actual}"
        )
    lines = [line for line in raw.splitlines() if line.strip()]
    return [json.loads(line) for line in lines], lines


def evidence_cluster(record: dict) -> str:
    if not record["answerable"]:
        return f"legacy-v1-unanswerable::{record['id']}"
    canonical = json.dumps(
        sorted(record["relevant"]), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"legacy-v1-evidence::{sha256_bytes(canonical)[:16]}"


def migrate(record: dict, raw_line: bytes) -> dict:
    answerable = bool(record["answerable"])
    documents = sorted({evidence.split("::", 1)[0] for evidence in record["relevant"]})
    tags = ["legacy_v1", "human_authored", *documents]
    claims = []
    if answerable:
        claims.append(
            {
                "id": "c1",
                "text": record["answer"],
                "evidence_ids": list(record["relevant"]),
                "status": "legacy_unatomized",
            }
        )
    return {
        "schema_version": "2.0.0-dev",
        "id": f"v1-{record['id']}",
        "dataset_version": DATASET_VERSION,
        "split": "dev",
        "lane": "legacy_v1_answerable" if answerable else "legacy_v1_unanswerable",
        "question": record["question"],
        "language": "en",
        "answerability": "answerable" if answerable else "unanswerable",
        "answer_action": "answer" if answerable else "abstain_absent",
        "reference_answer": record["answer"],
        "required_claims": claims,
        "acceptable_evidence": list(record["relevant"]),
        "distractor_evidence": [],
        "evidence_granularity": "legacy_heading" if answerable else "none",
        "reasoning_hops": 1 if answerable else 0,
        "intent_family_id": f"legacy-v1-intent::{record['id']}",
        "evidence_cluster_id": evidence_cluster(record),
        "conversation_id": None,
        "turn_index": None,
        "as_of": None,
        "severity": "unassigned",
        "traffic_weight": None,
        "tags": tags,
        "authoring": {
            "method": "human",
            "identity_disclosure": "not_recorded",
            "independent_of_corpus_authors": None,
            "model_assistance": "none",
        },
        "review": {
            "status": "migrated_pending_v2_review",
            "reviewers": [],
            "reviewed_at": None,
            "notes": [
                "Visible V1 case; retained in dev only.",
                "Reference answer remains one legacy unatomized claim."
                if answerable
                else "V1 authoring script classifies this as absent from the corpus.",
            ],
        },
        "unanswerable": None
        if answerable
        else {
            "reason": "absent_from_corpus",
            "notes": "Migrated from the V1 in-domain out-of-corpus abstention lane.",
        },
        "legacy": {
            "source_dataset": "evals/golden.jsonl",
            "source_dataset_sha256": SOURCE_SHA256,
            "source_record_sha256": sha256_bytes(raw_line),
            "v1_record": record,
        },
    }


def build_cases() -> list[dict]:
    records, raw_lines = load_v1()
    cases = [migrate(record, raw) for record, raw in zip(records, raw_lines)]
    if [case["legacy"]["v1_record"] for case in cases] != records:
        raise AssertionError("V1 round-trip check failed")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Migrated case IDs are not unique")
    return cases


def render(cases: list[dict]) -> bytes:
    text = "\n".join(
        json.dumps(case, ensure_ascii=False, separators=(",", ":"))
        for case in cases
    ) + "\n"
    return text.encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cases = build_cases()
    expected = render(cases)
    output = args.out.resolve()
    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(expected)
        action = "wrote"
    else:
        if not output.is_file():
            raise FileNotFoundError(f"Missing migrated dataset: {output}")
        if output.read_bytes() != expected:
            raise ValueError(
                "Migrated dataset is stale; regenerate with "
                "python scripts/migrate_v1_cases.py --write"
            )
        action = "verified"

    answerable = sum(case["answerability"] == "answerable" for case in cases)
    print(
        f"{action} {len(cases)} V2 dev cases "
        f"({answerable} answerable, {len(cases) - answerable} unanswerable)"
    )
    print(f"source sha256: {SOURCE_SHA256}")
    print(f"output sha256: {sha256_bytes(expected)}")
    print("round-trip: OK — every embedded V1 record matches the source")


if __name__ == "__main__":
    main()
