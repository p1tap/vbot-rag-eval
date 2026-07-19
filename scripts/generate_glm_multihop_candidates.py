"""Generate a disjoint, AI-authored multi-hop candidate queue with GLM 5.2."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag.llm import chat_with_metadata  # noqa: E402
from rag.provenance import sha256_file  # noqa: E402

CATALOG = ROOT / "corpus" / "evidence-catalog-v1.jsonl"
DEFAULT_OUT = ROOT / "evals" / "v2" / "candidates" / "glm-5.2-multihop-v1.json"
ALLOWED_LANES = (
    "multi_section",
    "multi_document",
    "comparison_aggregation",
    "global_corpus",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def existing_intent_ids() -> list[str]:
    result: set[str] = set()
    for path in (ROOT / "evals" / "v2").rglob("*.jsonl"):
        for row in jsonl(path):
            if row.get("intent_family_id"):
                result.add(row["intent_family_id"])
    for path in (ROOT / "evals" / "v2" / "candidates").glob("*.json"):
        body = json.loads(path.read_text(encoding="utf-8"))
        for row in body.get("cases", []):
            if row.get("intent_family_id"):
                result.add(row["intent_family_id"])
    return sorted(result)


def existing_case_summaries() -> list[dict]:
    summaries: dict[str, dict] = {}
    for path in (ROOT / "evals" / "v2").rglob("*.jsonl"):
        for row in jsonl(path):
            if row.get("id") and row.get("question"):
                summaries[row["id"]] = {
                    "id": row["id"],
                    "question": row["question"],
                    "reference_answer": row.get("reference_answer"),
                    "acceptable_evidence": row.get("acceptable_evidence", []),
                }
    for path in (ROOT / "evals" / "v2" / "candidates").glob("*.json"):
        body = json.loads(path.read_text(encoding="utf-8"))
        for row in body.get("cases", []):
            if row.get("id") and row.get("question"):
                summaries[row["id"]] = {
                    "id": row["id"],
                    "question": row["question"],
                    "reference_answer": row.get("reference_answer"),
                    "acceptable_evidence": row.get("acceptable_evidence", []),
                }
    return [summaries[key] for key in sorted(summaries)]


def response_format(count: int) -> dict:
    claim = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "evidence_ids"],
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    case = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "lane",
            "question",
            "reference_answer",
            "required_claims",
            "acceptable_evidence",
            "distractor_evidence",
            "reasoning_hops",
            "intent_family_id",
            "evidence_cluster_id",
            "severity",
        ],
        "properties": {
            "lane": {"type": "string", "enum": list(ALLOWED_LANES)},
            "question": {"type": "string", "minLength": 1},
            "reference_answer": {"type": "string", "minLength": 1},
            "required_claims": {
                "type": "array",
                "minItems": 2,
                "items": claim,
            },
            "acceptable_evidence": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "string", "minLength": 1},
            },
            "distractor_evidence": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "reasoning_hops": {"type": "integer", "minimum": 2, "maximum": 6},
            "intent_family_id": {"type": "string", "minLength": 1},
            "evidence_cluster_id": {"type": "string", "minLength": 1},
            "severity": {"type": "string", "enum": ["standard", "high", "critical"]},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "glm_multihop_case_candidates",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cases"],
                "properties": {
                    "cases": {
                        "type": "array",
                        "minItems": count,
                        "maxItems": count,
                        "items": case,
                    }
                },
            },
        },
    }


def validate(cases: list[dict], catalog: list[dict], intents: set[str], count: int) -> None:
    if len(cases) != count:
        raise SystemExit(f"expected {count} cases, received {len(cases)}")
    evidence_ids = {row["evidence_id"] for row in catalog}
    proposed_intents: set[str] = set()
    for position, case in enumerate(cases, start=1):
        if case["lane"] not in ALLOWED_LANES or case["reasoning_hops"] < 2:
            raise SystemExit(f"candidate {position} is not multi-hop")
        intent = case["intent_family_id"]
        if intent in intents or intent in proposed_intents:
            raise SystemExit(f"candidate {position} repeats intent {intent}")
        proposed_intents.add(intent)
        acceptable = set(case["acceptable_evidence"])
        cited = {
            evidence_id
            for claim in case["required_claims"]
            for evidence_id in claim["evidence_ids"]
        }
        all_ids = acceptable | cited | set(case["distractor_evidence"])
        unknown = all_ids - evidence_ids
        if unknown:
            raise SystemExit(f"candidate {position} cites unknown evidence: {sorted(unknown)}")
        if not cited <= acceptable:
            raise SystemExit(f"candidate {position} omits claim evidence from acceptable_evidence")
        if len(cited) < 2:
            raise SystemExit(f"candidate {position} uses fewer than two evidence spans")


def normalize_redundant_evidence(cases: list[dict]) -> list[dict]:
    """Make acceptable_evidence the ordered union of explicit claim citations."""
    changes = []
    for position, case in enumerate(cases, start=1):
        acceptable = list(dict.fromkeys(case["acceptable_evidence"]))
        for claim in case["required_claims"]:
            for evidence_id in claim["evidence_ids"]:
                if evidence_id not in acceptable:
                    acceptable.append(evidence_id)
                    changes.append(
                        {
                            "case_position": position,
                            "field": "acceptable_evidence",
                            "added_evidence_id": evidence_id,
                            "reason": "deterministic union of required-claim citations",
                        }
                    )
        case["acceptable_evidence"] = acceptable
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--id-prefix", default="v2g-glm")
    args = parser.parse_args()
    if not 4 <= args.count <= 20:
        raise SystemExit("count must be between 4 and 20")

    catalog = [
        row
        for row in jsonl(CATALOG)
        if row.get("block_type") != "heading" and row.get("word_count", 0) >= 8
    ]
    intents = existing_intent_ids()
    existing_cases = existing_case_summaries()
    profile = config.OPERATIONAL_CASE_AUTHORING_PROFILES[0]
    system = (
        "You author difficult evidence-grounded RAG evaluation candidates. "
        "Use only the supplied evidence catalog. Create genuinely disjoint multi-hop "
        "questions, not paraphrases of listed existing intent IDs. Every atomic claim "
        "must cite exact evidence IDs, and every question must require combining at "
        "least two distinct evidence spans. Distractors must be plausible but must not "
        "support a required claim. Return only the requested JSON."
    )
    user = json.dumps(
        {
            "requested_case_count": args.count,
            "allowed_lanes": list(ALLOWED_LANES),
            "existing_intent_ids_do_not_repeat": intents,
            "existing_cases_do_not_semantically_duplicate": existing_cases,
            "evidence_catalog": catalog,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    call = chat_with_metadata(
        profile["model"],
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=10000,
        temperature=None,
        retries=4,
        response_format=response_format(args.count),
        request_options=profile["request_options"],
        timeout_seconds=900,
    )
    parsed = json.loads(call.content)
    cases = parsed["cases"]
    deterministic_normalizations = normalize_redundant_evidence(cases)
    validate(cases, catalog, set(intents), args.count)
    now = datetime.now(timezone.utc).isoformat()
    output_cases = []
    for position, source in enumerate(cases, start=1):
        claims = [
            {
                "id": f"c{claim_position}",
                "text": claim["text"],
                "evidence_ids": claim["evidence_ids"],
                "status": "model_proposed_pending_review",
            }
            for claim_position, claim in enumerate(source["required_claims"], start=1)
        ]
        output_cases.append(
            {
                "schema_version": "2.0.0-dev",
                "id": f"{args.id_prefix}-{position:03d}",
                "dataset_version": "2.0.0-candidate.glm-5.2-v1",
                "split": "candidate",
                "lane": source["lane"],
                "question": source["question"],
                "language": "en",
                "answerability": "answerable",
                "answer_action": "answer",
                "reference_answer": source["reference_answer"],
                "required_claims": claims,
                "acceptable_evidence": source["acceptable_evidence"],
                "distractor_evidence": source["distractor_evidence"],
                "evidence_granularity": "span",
                "reasoning_hops": source["reasoning_hops"],
                "intent_family_id": source["intent_family_id"],
                "evidence_cluster_id": source["evidence_cluster_id"],
                "conversation_id": None,
                "turn_index": None,
                "as_of": None,
                "severity": source["severity"],
                "traffic_weight": None,
                "tags": ["ai_authored", "glm_5_2", "pending_cross_model_review"],
                "authoring": {
                    "method": "model_proposed_pending_review",
                    "model": profile["model"],
                    "provider": call.response_provider,
                    "model_assistance": "candidate_generation",
                },
                "review": {
                    "status": "pending",
                    "reviewers": [],
                    "reviewer_kinds": [],
                    "reviewed_at": None,
                    "notes": [
                        "AI-authored proposal only; no human review occurred.",
                        "Requires exact-source review by a model family other than GLM.",
                    ],
                },
                "unanswerable": None,
            }
        )
    report = {
        "schema_version": "1.0.0",
        "kind": "ai_authored_multihop_review_candidates",
        "created_at_utc": now,
        "promotion_status": "proposals_only_pending_cross_model_review",
        "human_reviewed_case_count_added": 0,
        "case_count": len(output_cases),
        "deterministic_normalizations": deterministic_normalizations,
        "generation": {
            "profile_id": profile["id"],
            "profile_sha256": canonical_sha256(profile),
            "requested_model": call.requested_model,
            "response_model": call.response_model,
            "response_provider": call.response_provider,
            "input_sha256": call.input_sha256,
            "response_sha256": hashlib.sha256(call.content.encode("utf-8")).hexdigest(),
            "request_options": call.request_options,
            "latency_ms": call.latency_ms,
            "usage": call.usage,
        },
        "inputs": {
            "evidence_catalog_path": CATALOG.relative_to(ROOT).as_posix(),
            "evidence_catalog_sha256": sha256_file(CATALOG),
            "existing_intent_count": len(intents),
            "existing_intent_ids_sha256": canonical_sha256(intents),
            "existing_case_count": len(existing_cases),
            "existing_cases_sha256": canonical_sha256(existing_cases),
            "system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        },
        "cases": output_cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(output_cases)} candidates -> {args.out}")
    print(f"provider: {call.response_provider}")


if __name__ == "__main__":
    main()
