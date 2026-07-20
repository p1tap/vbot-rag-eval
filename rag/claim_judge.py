"""Orchestrate independent support and required-claim coverage judges."""
from __future__ import annotations

from dataclasses import dataclass

from rag.claim_verifier import verify_claims
from rag.coverage_judge import judge_coverage

SUPPORT_STATUSES = {"supported", "unsupported", "contradicted"}
COVERAGE_STATUSES = {"covered", "missing", "contradicted"}


@dataclass(frozen=True)
class ClaimJudgeResult:
    """Compatibility record backed by two independently validated judge lanes."""

    valid: bool
    judgment: dict | None
    errors: tuple[dict[str, str], ...]
    metrics: dict[str, float | int | bool]
    support_judge: dict
    coverage_judge: dict
    call: dict
    calls: tuple[dict, ...]
    raw_output: str
    raw_outputs: tuple[str, ...]

    def to_record(self) -> dict:
        return {
            "valid": self.valid,
            "judgment": self.judgment,
            "errors": list(self.errors),
            "metrics": self.metrics,
            "support_judge": self.support_judge,
            "coverage_judge": self.coverage_judge,
            # These are coverage calls only. The runner already accounts for the
            # support calls under claim_verifier, so including them here would
            # double-count latency/cost.
            "call": self.call,
            "calls": list(self.calls),
            "raw_output": self.raw_output,
            "raw_outputs": list(self.raw_outputs),
        }


def _combined_metrics(support: dict, coverage: dict, valid: bool) -> dict:
    support_metrics = support.get("metrics", {})
    coverage_metrics = coverage.get("metrics", {})
    return {
        "generated_claim_count": support_metrics.get("generated_claim_count", 0),
        "supported_claim_count": support_metrics.get("supported_claim_count", 0),
        "generated_claim_support_rate": support_metrics.get(
            "generated_claim_support_rate", 0.0
        ),
        "required_claim_count": coverage_metrics.get("required_claim_count", 0),
        "covered_required_claim_count": coverage_metrics.get(
            "covered_required_claim_count", 0
        ),
        "required_claim_coverage_rate": coverage_metrics.get(
            "required_claim_coverage_rate", 0.0
        ),
        "claim_level_pass": bool(
            valid
            and support_metrics.get("runtime_claim_support_pass")
            and coverage_metrics.get("required_claim_coverage_pass")
        ),
    }


def _legacy_judgment(support: dict, coverage: dict) -> dict:
    return {
        "schema_version": "2.0.0",
        "claim_verdicts": [
            {
                "claim_id": verdict["claim_id"],
                "status": verdict["status"],
                "rationale": verdict["rationale"],
            }
            for verdict in support.get("legacy_verdicts", [])
        ],
        "required_claim_verdicts": list(coverage.get("legacy_verdicts", [])),
    }


def _lane_errors(lane: str, record: dict) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "lane": lane,
            "code": str(item.get("code", "unknown")),
            "message": str(item.get("message", "")),
        }
        for item in record.get("errors", [])
        if isinstance(item, dict)
    )


def validate_claim_judgment(
    judgment: object,
    generated_claim_ids: list[str],
    required_claim_ids: list[str],
) -> tuple[tuple[dict[str, str], ...], dict[str, float | int | bool]]:
    """Validate the legacy combined view used by agreement/report tooling."""
    errors: list[dict[str, str]] = []

    def error(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    if not isinstance(judgment, dict):
        error("root_type", "judgment must be an object")
        return tuple(errors), _legacy_metrics([], [])
    expected = {"schema_version", "claim_verdicts", "required_claim_verdicts"}
    if set(judgment) != expected:
        error("fields", f"judgment must contain exactly {sorted(expected)}")
    if judgment.get("schema_version") not in {"1.0.0", "2.0.0"}:
        error("schema_version", "schema_version must be 1.0.0 or 2.0.0")

    generated = judgment.get("claim_verdicts")
    required = judgment.get("required_claim_verdicts")
    if not isinstance(generated, list):
        error("claim_verdicts_type", "claim_verdicts must be an array")
        generated = []
    if not isinstance(required, list):
        error("required_verdicts_type", "required_claim_verdicts must be an array")
        required = []

    _validate_verdicts(
        generated,
        generated_claim_ids,
        SUPPORT_STATUSES,
        {"claim_id", "status", "rationale"},
        "generated",
        error,
    )
    _validate_verdicts(
        required,
        required_claim_ids,
        COVERAGE_STATUSES,
        {"claim_id", "status", "generated_claim_ids", "rationale"},
        "required",
        error,
    )
    allowed_generated = set(generated_claim_ids)
    for verdict in required:
        if not isinstance(verdict, dict):
            continue
        linked = verdict.get("generated_claim_ids")
        if not isinstance(linked, list):
            error("generated_claim_ids_type", "required verdict links must be an array")
            continue
        if len(linked) != len(set(linked)) or not set(linked) <= allowed_generated:
            error(
                "generated_claim_ids",
                "required verdict links contain duplicate or unknown IDs",
            )
        if verdict.get("status") == "missing" and linked:
            error("missing_with_links", "a missing required claim cannot link claims")
        if verdict.get("status") in {"covered", "contradicted"} and not linked:
            error(
                "verdict_without_links",
                "a covered or contradicted required claim must link generated claims",
            )

    return tuple(errors), _legacy_metrics(generated, required)


def _validate_verdicts(verdicts, expected_ids, statuses, keys, label, error):
    actual_ids = []
    for index, verdict in enumerate(verdicts, 1):
        if not isinstance(verdict, dict):
            error(f"{label}_verdict_type", f"{label} verdict {index} must be an object")
            continue
        if set(verdict) != keys:
            error(f"{label}_verdict_fields", f"{label} verdict {index} has wrong fields")
        actual_ids.append(verdict.get("claim_id"))
        if verdict.get("status") not in statuses:
            error(f"{label}_status", f"{label} verdict {index} has unknown status")
        if not isinstance(verdict.get("rationale"), str) or not verdict.get(
            "rationale", ""
        ).strip():
            error(f"{label}_rationale", f"{label} verdict {index} needs a rationale")
    if actual_ids != expected_ids:
        error(
            f"{label}_id_order",
            f"{label} verdict IDs must exactly match expected order",
        )


def _legacy_metrics(generated: list, required: list) -> dict[str, float | int | bool]:
    supported = sum(
        isinstance(item, dict) and item.get("status") == "supported" for item in generated
    )
    covered = sum(
        isinstance(item, dict) and item.get("status") == "covered" for item in required
    )
    return {
        "generated_claim_count": len(generated),
        "supported_claim_count": supported,
        "generated_claim_support_rate": supported / len(generated) if generated else 0.0,
        "required_claim_count": len(required),
        "covered_required_claim_count": covered,
        "required_claim_coverage_rate": covered / len(required) if required else 0.0,
        "claim_level_pass": bool(generated)
        and bool(required)
        and supported == len(generated)
        and covered == len(required),
    }


def judge_claims(
    question: str,
    required_claims: list[dict],
    answer: dict,
    sources: list[dict],
    *,
    support_record: dict | None = None,
) -> ClaimJudgeResult:
    """Run the two judge lanes and expose a backwards-compatible combined view."""
    support = support_record or verify_claims(question, answer, sources).to_record()
    coverage = judge_coverage(question, required_claims, answer).to_record()
    valid = bool(support.get("valid") and coverage.get("valid"))
    judgment = _legacy_judgment(support, coverage) if valid else None
    errors = _lane_errors("support", support) + _lane_errors("coverage", coverage)
    calls = tuple(coverage.get("calls", ()))
    raw_outputs = tuple(coverage.get("raw_outputs", ()))
    call = coverage.get("call", {})
    raw_output = coverage.get("raw_output", "")
    return ClaimJudgeResult(
        valid=valid,
        judgment=judgment,
        errors=errors,
        metrics=_combined_metrics(support, coverage, valid),
        support_judge=support,
        coverage_judge=coverage,
        call=call,
        calls=calls,
        raw_output=raw_output,
        raw_outputs=raw_outputs,
    )


__all__ = ["ClaimJudgeResult", "judge_claims", "validate_claim_judgment"]
