"""Audit the AI-reviewed release set without upgrading its provenance."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts import build_ai_release_seed, validate_data  # noqa: E402

DEFAULT_CASES = (
    ROOT / "evals" / "v2" / "release" / "vbot-ai-reviewed-release-50.jsonl"
)
DEFAULT_OUT = ROOT / "reports" / "v2" / "ai-release-seed-audit.json"
DEV_CASES = ROOT / "evals" / "v2" / "reviewed" / "vbot-human-reviewed-100.jsonl"
MAX_QUESTION_JACCARD = 0.75


def question_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9_.:/-]*", text.casefold()))


def question_jaccard(left: str, right: str) -> float:
    left_tokens = question_tokens(left)
    right_tokens = question_tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def audit(cases_path: Path) -> dict:
    if cases_path.read_bytes() != build_ai_release_seed.render(
        build_ai_release_seed.build_cases()
    ):
        raise ValueError("release set differs from deterministic builder output")
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    dev_cases = [
        json.loads(line)
        for line in DEV_CASES.read_text(encoding="utf-8").splitlines()
        if line
    ]
    overlap_rows = sorted(
        (
            {
                "release_case_id": release_case["id"],
                "comparison_case_id": comparison_case["id"],
                "comparison_split": comparison_split,
                "question_jaccard": question_jaccard(
                    release_case["question"], comparison_case["question"]
                ),
            }
            for release_index, release_case in enumerate(cases)
            for comparison_split, comparison_cases in (
                ("release", cases[:release_index]),
                ("dev", dev_cases),
            )
            for comparison_case in comparison_cases
        ),
        key=lambda row: (
            -row["question_jaccard"],
            row["release_case_id"],
            row["comparison_case_id"],
        ),
    )
    max_overlap = overlap_rows[0] if overlap_rows else None
    if max_overlap and max_overlap["question_jaccard"] >= MAX_QUESTION_JACCARD:
        raise ValueError(
            "release question is too close to an existing case: "
            f"{max_overlap}"
        )
    schema = json.loads(
        (ROOT / "evals" / "schema" / "case-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    corpus, legacy, spans = validate_data.validate_corpus()
    catalog = {
        row["evidence_id"]: row
        for line in (
            ROOT / corpus["evidence_catalog"]["path"]
        ).read_text(encoding="utf-8").splitlines()
        if line
        for row in [json.loads(line)]
    }
    decisions = []
    for case in cases:
        errors = sorted(validator.iter_errors(case), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"{case['id']} schema error: {errors[0].message}")
        validate_data.validate_case_integrity(case, legacy, spans)
        if (
            case["authoring"]["method"] != "model_proposed_ai_verified"
            or case["review"].get("reviewer_kinds") != ["ai_agent"]
            or "no_new_human_review" not in case["tags"]
        ):
            raise ValueError(f"{case['id']} has ambiguous AI provenance")
        snapshots = [
            {
                "evidence_id": evidence_id,
                "block_type": catalog[evidence_id]["block_type"],
                "lines": [
                    catalog[evidence_id]["start_line"],
                    catalog[evidence_id]["end_line"],
                ],
                "text": catalog[evidence_id]["text"],
                "content_sha256": catalog[evidence_id]["content_sha256"],
            }
            for evidence_id in [
                *case["acceptable_evidence"],
                *case["distractor_evidence"],
            ]
        ]
        decisions.append(
            {
                "case_id": case["id"],
                "decision": "approved_ai_only",
                "answerability": case["answerability"],
                "claim_count": len(case["required_claims"]),
                "evidence_snapshots": snapshots,
                "semantic_review": {
                    "reviewer_kind": "ai_agent",
                    "reviewer_id": case["review"]["reviewers"][0],
                    "same_agent_authored_and_reviewed": True,
                    "human_reviewed": False,
                    "basis": "direct comparison of question, reference answer, atomic claims, and exact catalog spans",
                },
            }
        )
    return {
        "schema_version": "1.0.0",
        "completed_at_utc": build_ai_release_seed.REVIEWED_AT,
        "status": "ai_only_release_seed_audit_passed",
        "release_eligible_for_production_claim": False,
        "source": {
            "path": cases_path.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(cases_path),
            "case_count": len(cases),
        },
        "review_provenance": {
            "locally_human_reviewed_case_count": 0,
            "locally_ai_reviewed_case_count": len(cases),
            "independent_author_reviewer": False,
        },
        "checks": {
            "deterministic_builder_match": True,
            "schema_valid": True,
            "case_integrity_valid": True,
            "exact_evidence_resolved": True,
            "ai_provenance_unambiguous": True,
            "no_high_lexical_overlap_with_dev_or_release": True,
        },
        "composition": {
            "answerability": dict(sorted(Counter(case["answerability"] for case in cases).items())),
            "lanes": dict(sorted(Counter(case["lane"] for case in cases).items())),
            "distinct_intent_family_count": len({case["intent_family_id"] for case in cases}),
            "distinct_acceptable_evidence_count": len(
                {evidence_id for case in cases for evidence_id in case["acceptable_evidence"]}
            ),
        },
        "lexical_overlap_audit": {
            "metric": "set Jaccard over case-folded alphanumeric question tokens",
            "reject_at_or_above": MAX_QUESTION_JACCARD,
            "maximum_observed": max_overlap,
            "top_five": overlap_rows[:5],
            "semantic_independence_claimed": False,
        },
        "limitations": [
            "The same AI agent authored and reviewed every case.",
            "Absence labels rely on an AI review of the small approved corpus, not an independent human annotator.",
            "Fifty cases improve lane coverage but remain a same-agent AI-reviewed release set, not a production-traffic sample.",
        ],
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = audit(args.cases)
    expected = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(expected)
        action = "wrote"
    elif not args.out.is_file() or args.out.read_bytes() != expected:
        raise SystemExit("AI release audit is stale; regenerate with --write")
    else:
        action = "verified"
    print(
        f"{action} AI release audit: {report['status']} "
        f"({report['source']['case_count']} cases, 0 human reviewed)"
    )


if __name__ == "__main__":
    main()
