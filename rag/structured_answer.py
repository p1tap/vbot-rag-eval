"""Fail-closed structured answers and deterministic citation checks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

ANSWER_ACTIONS = {
    "answer",
    "abstain_absent",
    "clarify_ambiguous",
    "abstain_conflict",
    "abstain_obsolete",
    "abstain_unauthorized",
    "reject_injection",
}
CLAIM_ID = re.compile(r"^c[1-9][0-9]*$")
SOURCE_ID = re.compile(r"^S[1-9][0-9]*$")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[dict[str, str], ...]
    metrics: dict[str, float | int | bool]

    def to_record(self) -> dict:
        return {"valid": self.valid, "errors": list(self.errors), "metrics": self.metrics}


def context_sources(chunks: Iterable[dict]) -> list[dict]:
    """Assign prompt-local citation IDs while retaining stable chunk identity."""

    return [
        {
            "citation_id": f"S{index}",
            "chunk_id": chunk.get("id") or chunk.get("heading_key"),
            "heading_key": chunk.get("heading_key"),
            "heading": chunk.get("heading", ""),
            "text": chunk.get("text", ""),
            "score": chunk.get("score"),
        }
        for index, chunk in enumerate(chunks, 1)
    ]


def format_sources(sources: Iterable[dict]) -> str:
    return "\n\n".join(
        f"[{source['citation_id']}] ({source['heading']})\n{source['text']}"
        for source in sources
    )


def structured_answer_response_format(
    valid_source_ids: Iterable[str], allowed_actions: Iterable[str]
) -> dict:
    """Build a decoder schema; semantic cross-field checks still run afterward."""

    source_ids = sorted(set(valid_source_ids))
    citation_item = (
        {"type": "string", "enum": source_ids}
        if source_ids
        else {"type": "string", "pattern": r"^S[1-9][0-9]*$"}
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "action", "response", "claims"],
        "properties": {
            "schema_version": {"type": "string", "const": "1.0.0"},
            "action": {"type": "string", "enum": sorted(set(allowed_actions))},
            "response": {"type": ["string", "null"]},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "text", "citation_ids"],
                    "properties": {
                        "id": {"type": "string", "pattern": r"^c[1-9][0-9]*$"},
                        "text": {"type": "string", "minLength": 1},
                        "citation_ids": {
                            "type": "array",
                            "items": citation_item,
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                    },
                },
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "vbot_structured_answer",
            "strict": True,
            "schema": schema,
        },
    }


def parse_structured_answer(raw: str) -> dict:
    """Parse exact JSON; prose and fenced JSON fail instead of being repaired silently."""

    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"model output is not exact JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


def validate_structured_answer(
    answer: object,
    valid_source_ids: Iterable[str],
    allowed_actions: Iterable[str] | None = None,
) -> ValidationResult:
    """Validate semantic constraints JSON Schema cannot check across prompt context."""

    errors: list[dict[str, str]] = []

    def error(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    if not isinstance(answer, dict):
        error("root_type", "answer must be an object")
        return ValidationResult(False, tuple(errors), _citation_metrics([], set()))

    expected_keys = {"schema_version", "action", "response", "claims"}
    missing = expected_keys - set(answer)
    extra = set(answer) - expected_keys
    if missing:
        error("missing_fields", f"missing fields: {sorted(missing)}")
    if extra:
        error("extra_fields", f"unexpected fields: {sorted(extra)}")
    if answer.get("schema_version") != "1.0.0":
        error("schema_version", "schema_version must be 1.0.0")

    action = answer.get("action")
    if action not in ANSWER_ACTIONS:
        error("action", f"unknown action: {action!r}")
    allowed = set(allowed_actions) if allowed_actions is not None else ANSWER_ACTIONS
    if action in ANSWER_ACTIONS and action not in allowed:
        error("unsupported_action", f"action {action!r} is unavailable in this runtime")
    if action == "answer" and answer.get("response") is not None:
        error("answer_response", "answer action requires response to be null")
    if action in ANSWER_ACTIONS - {"answer"} and (
        not isinstance(answer.get("response"), str)
        or not answer.get("response", "").strip()
    ):
        error("nonanswer_response", "non-answer action requires a non-empty response")

    claims = answer.get("claims")
    if not isinstance(claims, list):
        error("claims_type", "claims must be an array")
        claims = []
    if action == "answer" and not claims:
        error("answer_without_claims", "answer action requires at least one atomic claim")
    if action in ANSWER_ACTIONS - {"answer"} and claims:
        error("nonanswer_with_claims", "non-answer actions must not contain factual claims")

    seen_claims: set[str] = set()
    for index, claim in enumerate(claims, 1):
        if not isinstance(claim, dict):
            error("claim_type", f"claim {index} must be an object")
            continue
        claim_keys = {"id", "text", "citation_ids"}
        if set(claim) != claim_keys:
            error("claim_fields", f"claim {index} must contain exactly {sorted(claim_keys)}")
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not CLAIM_ID.fullmatch(claim_id):
            error("claim_id", f"claim {index} has invalid ID")
        elif claim_id in seen_claims:
            error("duplicate_claim_id", f"duplicate claim ID: {claim_id}")
        else:
            seen_claims.add(claim_id)
        if claim_id != f"c{index}":
            error("claim_order", f"claim {index} must use ID c{index}")
        if not isinstance(claim.get("text"), str) or not claim.get("text", "").strip():
            error("claim_text", f"claim {index} must have non-empty text")
        citations = claim.get("citation_ids")
        if not isinstance(citations, list) or not citations:
            error("uncited_claim", f"claim {claim_id or index} needs at least one citation")
            continue
        if any(not isinstance(item, str) or not SOURCE_ID.fullmatch(item) for item in citations):
            error("citation_format", f"claim {claim_id or index} has malformed citation IDs")
        if len(citations) != len(set(item for item in citations if isinstance(item, str))):
            error("duplicate_citation", f"claim {claim_id or index} repeats a citation")

    valid_sources = set(valid_source_ids)
    metrics = _citation_metrics(claims, valid_sources)
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("citation_ids"), list):
            continue
        for citation_id in claim["citation_ids"]:
            if citation_id not in valid_sources:
                error("nonexistent_citation", f"citation {citation_id!r} was not supplied to the model")

    return ValidationResult(not errors, tuple(errors), metrics)


def render_response(answer: dict) -> str:
    """Construct answer prose from the only factual surface: validated claims."""

    if answer.get("action") == "answer":
        return " ".join(claim["text"].strip() for claim in answer.get("claims", []))
    response = answer.get("response")
    return response.strip() if isinstance(response, str) else ""


def _citation_metrics(claims: list, valid_sources: set[str]) -> dict[str, float | int | bool]:
    citation_lists = [
        claim.get("citation_ids", [])
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("citation_ids", []), list)
    ]
    citations = [item for items in citation_lists for item in items if isinstance(item, str)]
    valid_count = sum(item in valid_sources for item in citations)
    claims_with_any = sum(bool(items) for items in citation_lists)
    claims_with_valid = sum(any(item in valid_sources for item in items) for items in citation_lists)
    claim_count = len(claims)
    return {
        "claim_count": claim_count,
        "citation_count": len(citations),
        "valid_citation_count": valid_count,
        "citation_existence_precision": valid_count / len(citations) if citations else 0.0,
        "claims_with_citations": claims_with_any,
        "claims_with_valid_citations": claims_with_valid,
        "claim_citation_coverage": claims_with_any / claim_count if claim_count else 0.0,
        "claim_valid_citation_coverage": claims_with_valid / claim_count if claim_count else 0.0,
    }


__all__ = [
    "ANSWER_ACTIONS",
    "ValidationResult",
    "context_sources",
    "format_sources",
    "parse_structured_answer",
    "render_response",
    "structured_answer_response_format",
    "validate_structured_answer",
]
