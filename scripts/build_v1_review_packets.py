"""Build source-backed V1 review material without approving any annotation.

The committed source map is deterministic and safe for CI. The private inbox
is a human workspace beside the repository. Generated source files may be
refreshed; decision templates and HUMAN-NOTES.md are created once and never
overwritten.

Run:
  python scripts/build_v1_review_packets.py --write
  python scripts/build_v1_review_packets.py --check
  python scripts/build_v1_review_packets.py --write-inbox
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_MANIFEST = ROOT / "evals" / "v2" / "dataset-manifest.json"
CORPUS_MANIFEST = ROOT / "corpus" / "manifest.json"
DEFAULT_OUT = ROOT / "evals" / "v2" / "v1-review-source-map.json"
DEFAULT_INBOX = ROOT.parent / "rag-eval-human-review"
BATCH_SIZE = 7


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


def build_source_map() -> dict:
    dataset = load_json(DATASET_MANIFEST)
    corpus = load_json(CORPUS_MANIFEST)
    cases = []
    for partition in dataset["partitions"]:
        for entry in partition["files"]:
            cases.extend(load_jsonl(ROOT / entry["path"]))
    catalog = load_jsonl(ROOT / corpus["evidence_catalog"]["path"])
    by_heading: dict[str, list[dict]] = {}
    for record in catalog:
        by_heading.setdefault(record["heading_key"], []).append(record)

    mapped_cases = []
    for case in sorted(cases, key=lambda item: item["id"]):
        legacy = case.get("legacy")
        if not legacy:
            continue
        # This artifact reconstructs the original V1 navigation packet even
        # after canonical cases have been upgraded to exact span evidence.
        headings = list(legacy["v1_record"]["relevant"])
        missing = [heading for heading in headings if heading not in by_heading]
        if missing:
            raise ValueError(f"unknown legacy headings for {case['id']}: {missing}")
        candidate_evidence = []
        for heading in headings:
            candidate_evidence.append(
                {
                    "heading_key": heading,
                    "candidate_blocks": [
                        {
                            "evidence_id": record["evidence_id"],
                            "block_type": record["block_type"],
                            "start_line": record["start_line"],
                            "end_line": record["end_line"],
                            "text": record["text"],
                        }
                        for record in by_heading[heading]
                    ],
                }
            )
        mapped_cases.append(
            {
                "case_id": case["id"],
                "legacy_id": legacy["v1_record"]["id"],
                "question": case["question"],
                "answerability": case["answerability"],
                "current_answer_action": case["answer_action"],
                "legacy_reference_answer": case["reference_answer"],
                "legacy_relevant_headings": headings,
                "candidate_evidence": candidate_evidence,
                # The source map is an immutable reconstruction of the input
                # packet, not the canonical post-review status artifact.
                "current_review_status": "migrated_pending_v2_review",
                "human_decision_required": True,
                "model_proposal_status": "source_navigation_only",
            }
        )

    return {
        "schema_version": "1.0.0",
        "purpose": "source-backed human review navigation; not approved annotation",
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["dataset_version"],
        "dataset_fingerprint_sha256": dataset["fingerprint_sha256"],
        "corpus_id": corpus["corpus_id"],
        "corpus_version": corpus["corpus_version"],
        "corpus_fingerprint_sha256": corpus["fingerprint_sha256"],
        "evidence_catalog_sha256": corpus["evidence_catalog"]["sha256"],
        "case_count": len(mapped_cases),
        "answerable_cases": sum(
            case["answerability"] == "answerable" for case in mapped_cases
        ),
        "unanswerable_cases": sum(
            case["answerability"] == "unanswerable" for case in mapped_cases
        ),
        "cases": mapped_cases,
        "warnings": [
            "Candidate blocks are navigation aids, not accepted evidence labels.",
            "Legacy reference answers may contain multiple atomic claims.",
            "Unanswerable cases require a human corpus-wide absence review.",
            "No decision in this artifact is human-approved.",
        ],
    }


def render_source_map(source_map: dict) -> bytes:
    return (json.dumps(source_map, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def batches(cases: list[dict]) -> list[list[dict]]:
    return [cases[index : index + BATCH_SIZE] for index in range(0, len(cases), BATCH_SIZE)]


def decision_template(case: dict) -> dict:
    return {
        "schema_version": "1.0.0",
        "case_id": case["case_id"],
        "reviewer_id": None,
        "decision": "pending",
        "answerability_confirmed": None,
        "answer_action": None,
        "lane": None,
        "severity": None,
        "atomic_claims": [],
        "acceptable_evidence_ids": [],
        "distractor_evidence_ids": [],
        "intent_family_id": None,
        "evidence_cluster_id": None,
        "notes": [],
        "reviewed_at": None,
    }


def markdown_quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def render_batch(batch_number: int, cases: list[dict]) -> str:
    lines = [
        f"# V1 review batch {batch_number:02d}",
        "",
        "Generated source packet. Human decisions belong in the matching file",
        "under `decisions/` or can be discussed with Codex in real time.",
        "Candidate blocks are navigation aids and are not approved labels.",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"Question: {case['question']}",
                "",
                f"Current answerability: `{case['answerability']}`",
                "",
                f"Current action: `{case['current_answer_action']}`",
                "",
                "Legacy reference answer:",
                "",
                markdown_quote(case["legacy_reference_answer"]),
                "",
            ]
        )
        if case["candidate_evidence"]:
            lines.extend(["Candidate source blocks:", ""])
            for group in case["candidate_evidence"]:
                lines.extend([f"### `{group['heading_key']}`", ""])
                for block in group["candidate_blocks"]:
                    lines.extend(
                        [
                            f"`{block['evidence_id']}` — {block['block_type']}, "
                            f"lines {block['start_line']}-{block['end_line']}",
                            "",
                            markdown_quote(block["text"]),
                            "",
                        ]
                    )
        else:
            lines.extend(
                [
                    "No legacy relevant heading exists. Human must verify that the",
                    "answer is absent from the complete approved corpus.",
                    "",
                ]
            )
        lines.extend(
            [
                "Human decisions required:",
                "",
                "- Confirm or correct answerability and action.",
                "- Assign the real V2 lane and severity.",
                "- For answerable cases, split the reference into atomic claims.",
                "- Select sufficient span IDs and any true distractors.",
                "- Confirm or merge intent/evidence cluster identities.",
                "- Record uncertainty or requested corpus changes.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_if_missing(path: Path, content: bytes) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True


def write_inbox(source_map: dict, inbox: Path) -> dict:
    source_dir = inbox / "source-batches"
    decisions_dir = inbox / "decisions"
    source_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir.mkdir(parents=True, exist_ok=True)
    packet_batches = batches(source_map["cases"])

    created_decisions = []
    generated_sources = []
    for number, batch in enumerate(packet_batches, 1):
        source_path = source_dir / f"batch-{number:02d}.md"
        source_path.write_text(render_batch(number, batch), encoding="utf-8")
        generated_sources.append(source_path.name)

        decision_path = decisions_dir / f"batch-{number:02d}-decisions.jsonl"
        decision_bytes = (
            "\n".join(
                json.dumps(decision_template(case), ensure_ascii=False, separators=(",", ":"))
                for case in batch
            )
            + "\n"
        ).encode("utf-8")
        if write_if_missing(decision_path, decision_bytes):
            created_decisions.append(decision_path.name)

    start_here = """# Vbot RAG human-review inbox

This folder is private working material. Start with `source-batches/batch-01.md`.
Record decisions in the matching JSONL file under `decisions/`, or discuss each
case with Codex in real time and ask Codex to transcribe your decision.

Use a pseudonymous reviewer ID such as `owner-reviewer-01`. Change `decision`
from `pending` only after checking the question, action, claims, and cited source
blocks. `revise` is the right choice whenever a claim or source label needs
work. These V1 development cases need one genuine human reviewer; future sealed
release/adversarial cases require independent second review and adjudication.

Codex-generated source navigation and wording proposals are not human approval.
Decision files and `HUMAN-NOTES.md` are never overwritten by the generator.
"""
    (inbox / "START-HERE.md").write_text(start_here, encoding="utf-8")
    write_if_missing(
        inbox / "HUMAN-NOTES.md",
        b"# Human review notes\n\nAdd cross-case decisions, questions, and policy changes here.\n",
    )
    manifest = {
        "schema_version": "1.0.0",
        "dataset_version": source_map["dataset_version"],
        "dataset_fingerprint_sha256": source_map["dataset_fingerprint_sha256"],
        "source_map_sha256": sha256_bytes(render_source_map(source_map)),
        "batch_size": BATCH_SIZE,
        "batches": len(packet_batches),
        "cases": source_map["case_count"],
        "generated_source_files": generated_sources,
        "new_decision_files_created": created_decisions,
        "decision_files_are_never_overwritten": True,
    }
    (inbox / "inbox-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write-inbox", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    args = parser.parse_args()

    source_map = build_source_map()
    expected = render_source_map(source_map)
    output = args.out.resolve()
    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(expected)
        action = "wrote"
    elif args.check:
        if not output.is_file() or output.read_bytes() != expected:
            raise ValueError(
                "V1 review source map is missing or stale; regenerate with "
                "python scripts/build_v1_review_packets.py --write"
            )
        action = "verified"
    else:
        manifest = write_inbox(source_map, args.inbox.resolve())
        print(
            f"wrote private inbox: {manifest['batches']} batches, "
            f"{manifest['cases']} cases"
        )
        print(f"inbox: {args.inbox.resolve()}")
        return

    print(
        f"{action} V1 review source map: {source_map['case_count']} cases "
        f"({source_map['answerable_cases']} answerable, "
        f"{source_map['unanswerable_cases']} unanswerable)"
    )
    print(f"output sha256: {sha256_bytes(expected)}")


if __name__ == "__main__":
    main()
