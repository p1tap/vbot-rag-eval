"""Materialize the cross-model review of GLM-authored candidate cases."""
from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402

OUT = ROOT / "evals" / "v2" / "candidates" / "glm-5.2-multihop-cross-model-review.json"
CATALOG = ROOT / "corpus" / "evidence-catalog-v1.jsonl"
SOURCES = [
    ROOT / "evals" / "v2" / "candidates" / f"glm-5.2-multihop-v1{suffix}.json"
    for suffix in "abcde"
]

APPROVED = {
    "v2g-glm-a-003": "New multi-section chain connects host diversity, the empty-200 failure, and the retry contract.",
    "v2g-glm-d-001": "New comparison joins the overfit probe and the later DPO gate failure.",
    "v2g-glm-d-002": "New operational comparison covers both real gate PR outcomes and reconciliation configuration.",
    "v2g-glm-d-003": "New policy contrast explains why product generation uses fallback while blind evaluation forbids it.",
    "v2g-glm-d-004": "New topology aggregation joins the three hosted RP legs with the later self-hosted leg.",
    "v2g-glm-e-001": "New training-plan aggregation tests epoch choice, conversation construction, and capability-collapse checking.",
    "v2g-glm-e-003": "New data-quality comparison joins DPO pair thresholds, teacher filtering, and the hold-rate probe.",
    "v2g-glm-e-004": "New cost-control comparison spans load testing, dataset generation, and per-key ledger verification.",
}

REJECTED = {
    "v2g-glm-a-001": "Semantic overlap with the existing rank/calibration and Lane C cases.",
    "v2g-glm-a-002": "Semantic overlap with the existing DPO failure-mode case; one training-loss claim also needed an additional citation.",
    "v2g-glm-a-004": "Broad lifecycle case superseded by the tighter two-PR candidate v2g-glm-d-002.",
    "v2g-glm-b-001": "Joins environment isolation and DPO behavior without a coherent user task.",
    "v2g-glm-b-002": "Semantic duplicate of the existing optimization-metrics promotion-gate case.",
    "v2g-glm-b-003": "Superseded by the more explicit fallback-policy contrast in v2g-glm-d-003.",
    "v2g-glm-b-004": "Semantic overlap with the existing fine-tune-to-production lifecycle case.",
    "v2g-glm-c-001": "Semantic duplicate of the existing optimization-metrics promotion-gate case.",
    "v2g-glm-c-002": "Semantic overlap with existing gateway-budget cases and v2g-glm-d-003.",
    "v2g-glm-c-003": "Semantic duplicate of the existing DPO failure-mode case.",
    "v2g-glm-c-004": "Semantic overlap with the existing lifecycle case and v2g-glm-d-002.",
    "v2g-glm-e-002": "Aggregates three security facts without a coherent end-user task.",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    catalog = {row["evidence_id"]: row for row in load_jsonl(CATALOG)}
    candidates = {}
    source_artifacts = []
    for path in SOURCES:
        body = json.loads(path.read_text(encoding="utf-8"))
        source_artifacts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "provider": body["generation"]["response_provider"],
                "case_count": body["case_count"],
            }
        )
        for case in body["cases"]:
            if case["id"] in candidates:
                raise SystemExit(f"duplicate candidate id: {case['id']}")
            candidates[case["id"]] = case

    expected = set(APPROVED) | set(REJECTED)
    if set(candidates) != expected or set(APPROVED) & set(REJECTED):
        raise SystemExit("review disposition does not cover the candidate universe exactly")

    now = datetime.now(timezone.utc).isoformat()
    approved_cases = []
    for case_id, rationale in APPROVED.items():
        case = copy.deepcopy(candidates[case_id])
        acceptable = set(case["acceptable_evidence"])
        distractors = set(case["distractor_evidence"])
        if acceptable & distractors:
            raise SystemExit(f"support/distractor overlap: {case_id}")
        for claim in case["required_claims"]:
            if not set(claim["evidence_ids"]) <= acceptable:
                raise SystemExit(f"claim evidence missing from acceptable set: {case_id}")
            for evidence_id in claim["evidence_ids"]:
                if evidence_id not in catalog:
                    raise SystemExit(f"unknown evidence id: {case_id} {evidence_id}")
            claim["status"] = "atomic_verified_by_cross_model_ai"
        case["tags"] = [
            "ai_authored",
            "ai_reviewed",
            "glm_5_2",
            "cross_model_reviewed",
            "no_new_human_review",
        ]
        case["review"] = {
            "status": "approved",
            "reviewers": ["codex-gpt-5-cross-model-reviewer"],
            "reviewer_kinds": ["ai_agent"],
            "reviewed_at": now,
            "notes": [
                rationale,
                "Exact claim evidence IDs were checked against the frozen evidence catalog.",
                "GLM authored the proposal; a non-GLM Codex model performed this review.",
                "No human review occurred; this does not increase the human-reviewed count.",
                "Approved for future AI-reviewed adversarial integration, not as an independent holdout.",
            ],
        }
        approved_cases.append(case)

    report = {
        "schema_version": "1.0.0",
        "kind": "cross_model_ai_review_of_glm_multihop_candidates",
        "completed_at_utc": now,
        "reviewer": {
            "id": "codex-gpt-5-cross-model-reviewer",
            "kind": "ai_agent",
            "independent_model_family_from_author": True,
        },
        "status": "approved_candidates_pending_dataset_integration",
        "human_reviewed_case_count_added": 0,
        "candidate_count": len(candidates),
        "approved_count": len(APPROVED),
        "rejected_count": len(REJECTED),
        "source_artifacts": source_artifacts,
        "evidence_catalog": {
            "path": CATALOG.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(CATALOG),
        },
        "approved_cases": approved_cases,
        "rejections": [
            {"case_id": case_id, "reason": reason}
            for case_id, reason in REJECTED.items()
        ],
        "limitations": [
            "These cases are AI-authored and AI-reviewed; they are not human-reviewed.",
            "They are visible derivative adversarial candidates, not an independent sealed holdout.",
            "Dataset integration requires a new manifest version and must not rewrite frozen partitions.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"approved {len(APPROVED)}; rejected {len(REJECTED)} -> {OUT}")


if __name__ == "__main__":
    main()
