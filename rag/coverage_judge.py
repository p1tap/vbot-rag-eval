"""Required-claim coverage judging against the generated response only."""
from __future__ import annotations

import json
from dataclasses import dataclass

import config
from rag.judge_contract import (
    missing_exact_anchors,
    missing_explicit_negation,
    namespaced_ids,
)
from rag.llm import chat_with_metadata

COVERAGE_STATUSES = {"covered", "missing", "contradicted"}

COVERAGE_JUDGE_SYSTEM = """You are a required-claim coverage judge, not an assistant.
Compare each required reference claim only with the generated claims. You will
not receive source documents. Do not use outside knowledge and do not fill in a
fact merely because the question, a rationale, or a related generated claim
makes it plausible.

Generated claims use IDs g1, g2, and so on. Required claims use IDs r1, r2,
and so on. A covered or contradicted verdict must link every generated claim
needed for that verdict. A missing verdict must use an empty generated_claim_ids
array. Generic wording does not cover a more specific required fact.

Important near misses:
- "Public Grafana is available" does not cover "anonymous writes return 403."
- "The model goes through a gate and Argo" does not cover a required PR number,
  model role, or merge-to-deploy transition that the answer omitted.
- Reporting the post-retry failure rate does not cover the rule that the client
  retries an empty response exactly once.

Return exactly one JSON object with no Markdown:
{
  "schema_version": "1.0.0",
  "required_claim_verdicts": [
    {
      "required_claim_id": "r1",
      "status": "covered",
      "generated_claim_ids": ["g1"],
      "rationale": "brief comparison reason"
    }
  ]
}

Status must be covered, missing, or contradicted. Emit exactly one verdict for
every supplied required claim, in the supplied order, and do not add IDs."""

COVERAGE_JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "required_claim_coverage",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "required_claim_verdicts"],
            "properties": {
                "schema_version": {"type": "string", "const": "1.0.0"},
                "required_claim_verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "required_claim_id",
                            "status",
                            "generated_claim_ids",
                            "rationale",
                        ],
                        "properties": {
                            "required_claim_id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["covered", "missing", "contradicted"],
                            },
                            "generated_claim_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "rationale": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True)
class CoverageJudgeResult:
    valid: bool
    judgment: dict | None
    legacy_verdicts: tuple[dict, ...]
    errors: tuple[dict[str, str], ...]
    metrics: dict[str, float | int | bool]
    call: dict
    calls: tuple[dict, ...]
    raw_output: str
    raw_outputs: tuple[str, ...]

    def to_record(self) -> dict:
        return {
            "valid": self.valid,
            "judgment": self.judgment,
            "legacy_verdicts": list(self.legacy_verdicts),
            "errors": list(self.errors),
            "metrics": self.metrics,
            "call": self.call,
            "calls": list(self.calls),
            "raw_output": self.raw_output,
            "raw_outputs": list(self.raw_outputs),
        }


def coverage_payload(question: str, required_claims: list[dict], answer: dict) -> str:
    generated_ids = namespaced_ids("g", len(answer.get("claims", [])))
    required_ids = namespaced_ids("r", len(required_claims))
    payload = {
        "question": question,
        "generated_response": " ".join(
            claim["text"].strip() for claim in answer.get("claims", [])
        ),
        "generated_claims": [
            {"id": generated_id, "text": claim["text"]}
            for generated_id, claim in zip(generated_ids, answer.get("claims", []))
        ],
        "required_claims": [
            {"id": required_id, "text": claim["text"]}
            for required_id, claim in zip(required_ids, required_claims)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def validate_coverage_judgment(
    judgment: object, required_claims: list[dict], answer: dict
) -> tuple[tuple[dict[str, str], ...], dict[str, float | int | bool]]:
    errors: list[dict[str, str]] = []
    generated_claims = answer.get("claims", [])
    generated_ids = namespaced_ids("g", len(generated_claims))
    required_ids = namespaced_ids("r", len(required_claims))
    generated_by_id = dict(zip(generated_ids, generated_claims))

    def error(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    if not isinstance(judgment, dict):
        error("root_type", "coverage judgment must be an object")
        return tuple(errors), _metrics(len(required_claims), [])
    if set(judgment) != {"schema_version", "required_claim_verdicts"}:
        error("fields", "coverage judgment has missing or extra fields")
    if judgment.get("schema_version") != "1.0.0":
        error("schema_version", "schema_version must be 1.0.0")
    verdicts = judgment.get("required_claim_verdicts")
    if not isinstance(verdicts, list):
        error("verdicts_type", "required_claim_verdicts must be an array")
        verdicts = []

    actual_ids: list[object] = []
    for index, verdict in enumerate(verdicts):
        if not isinstance(verdict, dict):
            error("verdict_type", f"coverage verdict {index + 1} must be an object")
            continue
        expected_fields = {
            "required_claim_id",
            "status",
            "generated_claim_ids",
            "rationale",
        }
        if set(verdict) != expected_fields:
            error("verdict_fields", f"coverage verdict {index + 1} has wrong fields")
        required_id = verdict.get("required_claim_id")
        actual_ids.append(required_id)
        status = verdict.get("status")
        if status not in COVERAGE_STATUSES:
            error("status", f"coverage verdict {index + 1} has invalid status")
        if not isinstance(verdict.get("rationale"), str) or not verdict.get(
            "rationale", ""
        ).strip():
            error("rationale", f"coverage verdict {index + 1} needs a rationale")
        linked = verdict.get("generated_claim_ids")
        if not isinstance(linked, list):
            error("generated_claim_ids_type", f"{required_id} links must be an array")
            linked = []
        elif len(linked) != len(set(linked)):
            error("duplicate_generated_claim_id", f"{required_id} has duplicate links")
        unknown = set(linked) - set(generated_ids)
        if unknown:
            error("unknown_generated_claim_id", f"{required_id} links unknown IDs {sorted(unknown)}")
        if status == "missing" and linked:
            error("missing_with_links", f"missing verdict {required_id} cannot link claims")
        if status in {"covered", "contradicted"} and not linked:
            error("verdict_without_links", f"{status} verdict {required_id} must link claims")

        if status == "covered" and index < len(required_claims) and not unknown:
            linked_text = " ".join(
                generated_by_id[item]["text"] for item in linked if item in generated_by_id
            )
            missing = missing_exact_anchors(required_claims[index]["text"], linked_text)
            if missing:
                error(
                    "coverage_exact_anchor_missing",
                    f"{required_id} exact values are absent from linked claims: {list(missing)}",
                )
            if missing_explicit_negation(required_claims[index]["text"], linked_text):
                error(
                    "coverage_explicit_negation_missing",
                    f"{required_id} is explicitly negative but its linked claims are not",
                )

    if actual_ids != required_ids:
        error("id_order", "coverage verdict IDs must exactly match required claim order")
    valid = not errors
    return tuple(errors), _metrics(len(required_claims), verdicts if valid else [])


def _metrics(expected_count: int, verdicts: list[dict]) -> dict[str, float | int | bool]:
    covered = sum(
        isinstance(verdict, dict) and verdict.get("status") == "covered"
        for verdict in verdicts
    )
    return {
        "required_claim_count": expected_count,
        "covered_required_claim_count": covered,
        "required_claim_coverage_rate": covered / expected_count if expected_count else 0.0,
        "required_claim_coverage_pass": bool(expected_count) and covered == expected_count,
    }


def _legacy_verdicts(
    judgment: dict | None, required_claims: list[dict], answer: dict
) -> tuple[dict, ...]:
    if not judgment:
        return ()
    generated_ids = namespaced_ids("g", len(answer.get("claims", [])))
    generated_to_legacy = {
        generated_id: claim["id"]
        for generated_id, claim in zip(generated_ids, answer.get("claims", []))
    }
    mapped = []
    for required, verdict in zip(required_claims, judgment["required_claim_verdicts"]):
        mapped.append(
            {
                "claim_id": required["id"],
                "status": verdict["status"],
                "generated_claim_ids": [
                    generated_to_legacy[item] for item in verdict["generated_claim_ids"]
                ],
                "rationale": verdict["rationale"],
            }
        )
    return tuple(mapped)


def judge_coverage(
    question: str, required_claims: list[dict], answer: dict
) -> CoverageJudgeResult:
    messages = [
        {"role": "system", "content": COVERAGE_JUDGE_SYSTEM},
        {"role": "user", "content": coverage_payload(question, required_claims, answer)},
    ]
    calls, raw_outputs = [], []
    judgment = None
    errors: tuple[dict[str, str], ...] = ()
    metrics = _metrics(len(required_claims), [])
    for _ in range(2):
        call = chat_with_metadata(
            config.JUDGE_MODEL,
            messages,
            max_tokens=900,
            temperature=0.0,
            response_format=COVERAGE_JUDGE_RESPONSE_FORMAT,
            request_options=config.JUDGE_REQUEST_OPTIONS,
        )
        calls.append(call.to_record())
        raw_outputs.append(call.content)
        try:
            judgment = json.loads(call.content)
        except json.JSONDecodeError as exc:
            judgment = None
            errors = ({"code": "invalid_json", "message": str(exc)},)
            metrics = _metrics(len(required_claims), [])
            continue
        errors, metrics = validate_coverage_judgment(judgment, required_claims, answer)
        if not errors:
            break
    valid = not errors
    return CoverageJudgeResult(
        valid,
        judgment,
        _legacy_verdicts(judgment if valid else None, required_claims, answer),
        errors,
        metrics,
        calls[-1],
        tuple(calls),
        raw_outputs[-1],
        tuple(raw_outputs),
    )


__all__ = [
    "COVERAGE_JUDGE_RESPONSE_FORMAT",
    "COVERAGE_JUDGE_SYSTEM",
    "CoverageJudgeResult",
    "coverage_payload",
    "judge_coverage",
    "validate_coverage_judgment",
]
