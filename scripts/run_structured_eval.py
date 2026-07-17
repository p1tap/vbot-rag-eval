"""Run the V2 structured, citation-aware, claim-level evaluation.

By default only human-approved cases run. ``--allow-unreviewed`` is an explicit
development escape hatch and is stamped into the report; such a report is not
eligible for release or resume claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag import llm  # noqa: E402
from rag.claim_judge import judge_claims  # noqa: E402
from rag.claim_verifier import (  # noqa: E402
    VERIFIER_RESPONSE_FORMAT,
    VERIFIER_SYSTEM,
    verify_claims,
)
from rag.coverage_judge import (  # noqa: E402
    COVERAGE_JUDGE_RESPONSE_FORMAT,
    COVERAGE_JUDGE_SYSTEM,
)
from rag.generate import structured_answer  # noqa: E402
from rag.provenance import capture_run_provenance, sha256_file  # noqa: E402
from rag.retrieve import load_index, retrieve  # noqa: E402
from rag.structured_answer import render_response  # noqa: E402

DEFAULT_CASES = ROOT / "evals" / "v2" / "candidates" / "phase1-new-candidates.json"
DEFAULT_OUT = ROOT / "reports" / "v2" / "structured-eval.json"
RUNNER_VERSION = "3.0.0"


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_cases(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    raise ValueError(f"{path} must be a JSON array, an object with cases, or JSONL")


def expected_action(case: dict) -> str:
    return "answer" if case["answerability"] == "answerable" else case["answer_action"]


def evidence_heading(evidence_id: str) -> str:
    return evidence_id.split("::span-", 1)[0]


def verifier_support_pass(judgment: dict) -> bool:
    metrics = judgment.get("metrics", {})
    return bool(
        judgment.get("valid")
        and metrics.get("generated_claim_count")
        and metrics.get("supported_claim_count") == metrics.get("generated_claim_count")
    )


def retrieval_diagnostics(case: dict, sources: list[dict]) -> dict:
    if case["answerability"] != "answerable":
        return {}
    retrieved = {source["heading_key"] for source in sources}
    gold = {evidence_heading(item) for item in case["acceptable_evidence"]}
    claim_sets = [
        {evidence_heading(item) for item in claim["evidence_ids"]}
        for claim in case["required_claims"]
    ]
    covered_claims = sum(bool(required) and required <= retrieved for required in claim_sets)
    return {
        "gold_evidence_heading_count": len(gold),
        "retrieved_gold_evidence_heading_count": len(gold & retrieved),
        "retrieval_any_gold_evidence": bool(gold & retrieved),
        "retrieval_complete_gold_evidence": bool(gold) and gold <= retrieved,
        "retrieval_required_claim_count": len(claim_sets),
        "retrieval_covered_required_claim_count": covered_claims,
        "retrieval_required_claim_coverage": covered_claims / len(claim_sets)
        if claim_sets
        else None,
    }


def aggregate(items: list[dict]) -> dict:
    total = len(items)
    valid = sum(item.get("contract_valid", False) for item in items)
    actions = sum(item.get("action_correct", False) for item in items)
    citations = sum(item.get("citation_count", 0) for item in items)
    valid_citations = sum(item.get("valid_citation_count", 0) for item in items)
    citation_claims = sum(item.get("citation_claim_count", 0) for item in items)
    claims_with_valid_citations = sum(
        item.get("claims_with_valid_citations", 0) for item in items
    )
    generated_claims = sum(item.get("generated_claim_count", 0) for item in items)
    supported_claims = sum(item.get("supported_claim_count", 0) for item in items)
    required_claims = sum(item.get("required_claim_count", 0) for item in items)
    covered_claims = sum(item.get("covered_required_claim_count", 0) for item in items)
    judged = sum("claim_judge" in item for item in items)
    judge_valid = sum(item.get("claim_judge", {}).get("valid", False) for item in items)
    verified = sum("claim_verifier" in item for item in items)
    verifier_valid = sum(
        item.get("claim_verifier", {}).get("valid", False) for item in items
    )
    scorable_passes = [item["case_pass"] for item in items if item.get("case_pass") is not None]
    supported_runtime_items = [
        item for item in items if item.get("runtime_supports_expected_action", False)
    ]
    answerable_retrieval = [item for item in items if "retrieval_any_gold_evidence" in item]
    retrieval_claims = sum(
        item.get("retrieval_required_claim_count", 0) for item in answerable_retrieval
    )
    retrieved_claims = sum(
        item.get("retrieval_covered_required_claim_count", 0)
        for item in answerable_retrieval
    )
    costs = [
        call.get("cost")
        for item in items
        for call in _calls(item)
        if call.get("cost") is not None
    ]

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    summary = {
        "case_count": total,
        "contract_valid_rate": ratio(valid, total),
        "action_accuracy": ratio(actions, total),
        "citation_existence_precision": ratio(valid_citations, citations),
        "claim_valid_citation_coverage": ratio(claims_with_valid_citations, citation_claims),
        "generated_claim_support_rate": ratio(supported_claims, generated_claims),
        "required_claim_coverage_rate": ratio(covered_claims, required_claims),
        "judge_output_valid_rate": ratio(judge_valid, judged),
        "verifier_output_valid_rate": ratio(verifier_valid, verified),
        "end_to_end_case_pass_rate": ratio(sum(scorable_passes), len(scorable_passes)),
        "runtime_supported_case_count": len(supported_runtime_items),
        "runtime_unsupported_case_ids": [
            item["id"]
            for item in items
            if not item.get("runtime_supports_expected_action", False)
        ],
        "supported_action_accuracy": ratio(
            sum(item.get("action_correct", False) for item in supported_runtime_items),
            len(supported_runtime_items),
        ),
        "retrieval_any_gold_evidence_rate": ratio(
            sum(item["retrieval_any_gold_evidence"] for item in answerable_retrieval),
            len(answerable_retrieval),
        ),
        "retrieval_complete_gold_evidence_rate": ratio(
            sum(item["retrieval_complete_gold_evidence"] for item in answerable_retrieval),
            len(answerable_retrieval),
        ),
        "retrieval_required_claim_coverage": ratio(retrieved_claims, retrieval_claims),
        "provider_reported_cost": round(sum(costs), 6) if costs else None,
        "provider_reported_cost_coverage_calls": len(costs),
    }
    summary["failure_case_ids"] = [
        item["id"] for item in items if item.get("case_pass") is False
    ]
    summary["by_lane"] = {
        lane: _group_summary([item for item in items if item.get("lane") == lane])
        for lane in sorted({item.get("lane") for item in items if item.get("lane")})
    }
    return summary


def _group_summary(items: list[dict]) -> dict:
    scorable = [item["case_pass"] for item in items if item.get("case_pass") is not None]
    return {
        "case_count": len(items),
        "contract_valid": sum(item.get("contract_valid", False) for item in items),
        "action_correct": sum(item.get("action_correct", False) for item in items),
        "case_pass": sum(scorable),
        "case_pass_rate": round(sum(scorable) / len(scorable), 4) if scorable else None,
    }


def _calls(item: dict):
    for section in (
        item.get("generation", {}),
        item.get("claim_verifier", {}),
        item.get("claim_judge", {}),
    ):
        calls = section.get("calls")
        if isinstance(calls, list):
            yield from (call for call in calls if isinstance(call, dict))
        elif isinstance(section.get("call"), dict):
            yield section["call"]


def evaluate_case(
    case: dict, index, *, skip_judge: bool, top_k: int, generator_profile: dict
) -> dict:
    started = time.perf_counter()
    chunks = retrieve(
        case["question"], k=top_k, index=index, unique_headings=True
    )
    answer, _, audit = structured_answer(
        case["question"],
        chunks,
        allowed_actions=config.CURRENT_SUPPORTED_ACTIONS,
        model=generator_profile["model"],
        max_tokens=generator_profile["max_tokens"],
        temperature=generator_profile["temperature"],
        request_options=generator_profile["request_options"],
    )
    validation = audit["validation"]
    result = {
        "id": case["id"],
        "split": case["split"],
        "lane": case["lane"],
        "severity": case["severity"],
        "expected_action": expected_action(case),
        "runtime_supports_expected_action": expected_action(case)
        in config.CURRENT_SUPPORTED_ACTIONS,
        "raw_action": answer.get("action") if answer else None,
        "actual_action": answer.get("action") if answer else None,
        "contract_valid": validation["valid"],
        "action_correct": False,
        "answer": answer,
        "rendered_response": render_response(answer) if answer else None,
        "effective_response": render_response(answer) if answer else None,
        "generation": audit,
        "citation_count": validation.get("metrics", {}).get("citation_count", 0),
        "valid_citation_count": validation.get("metrics", {}).get("valid_citation_count", 0),
        "citation_claim_count": validation.get("metrics", {}).get("claim_count", 0),
        "claims_with_valid_citations": validation.get("metrics", {}).get(
            "claims_with_valid_citations", 0
        ),
        **retrieval_diagnostics(case, audit["sources"]),
    }
    should_verify = (
        not skip_judge
        and validation["valid"]
        and answer is not None
        and answer.get("action") == "answer"
    )
    if should_verify:
        verified = verify_claims(
            case["question"], answer, audit["sources"]
        ).to_record()
        result["claim_verifier"] = verified
        verifier_metrics = verified["metrics"]
        support_pass = verifier_support_pass(verified)
        result["runtime_claim_support_pass"] = support_pass
        result["generated_claim_count"] = verifier_metrics["generated_claim_count"]
        result["supported_claim_count"] = verifier_metrics["supported_claim_count"]
        if not support_pass:
            result["actual_action"] = "abstain_absent"
            result["effective_response"] = config.ABSTAIN_STRING
            result["verification_override"] = {
                "from": "answer",
                "to": "abstain_absent",
                "reason": "generated claims did not all pass cited-source verification",
            }
    should_judge = (
        should_verify and case["answerability"] == "answerable"
    )
    if should_judge:
        judged = judge_claims(
            case["question"],
            case["required_claims"],
            answer,
            audit["sources"],
            support_record=verified,
        ).to_record()
        result["claim_judge"] = judged
        for key in ("required_claim_count", "covered_required_claim_count"):
            result[key] = judged["metrics"][key]
    result["action_correct"] = bool(answer) and result["actual_action"] == expected_action(case)
    if skip_judge and case["answerability"] == "answerable":
        result["case_pass"] = None
    elif case["answerability"] == "answerable":
        result["case_pass"] = bool(
            result["contract_valid"]
            and result["action_correct"]
            and result.get("claim_judge", {}).get("valid")
            and result.get("claim_judge", {}).get("metrics", {}).get("claim_level_pass")
        )
    else:
        result["case_pass"] = bool(result["contract_valid"] and result["action_correct"])
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--lanes", default="")
    parser.add_argument(
        "--generator-profile",
        choices=[row["id"] for row in config.GENERATOR_EVAL_PROFILES],
        default="current-llama-unpinned",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    generator_profile = next(
        row for row in config.GENERATOR_EVAL_PROFILES
        if row["id"] == args.generator_profile
    )
    expected_endpoint = generator_profile.get("endpoint")
    if expected_endpoint and llm.BASE_URL != expected_endpoint:
        raise SystemExit(
            f"profile {args.generator_profile} requires "
            f"RAG_LLM_BASE_URL={expected_endpoint}"
        )

    cases = load_cases(args.cases)
    selected_lanes = {item.strip() for item in args.lanes.split(",") if item.strip()}
    if selected_lanes:
        cases = [case for case in cases if case["lane"] in selected_lanes]
    if not args.allow_unreviewed:
        cases = [
            case
            for case in cases
            if case.get("review", {}).get("status") == "approved"
        ]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit(
            "No eligible cases. Finish/export human approvals, or use "
            "--allow-unreviewed for a development-only smoke run."
        )
    unsupported_case_ids = [
        case["id"]
        for case in cases
        if expected_action(case) not in config.CURRENT_SUPPORTED_ACTIONS
    ]

    prompt_hashes = {
        "structured_answer_system_sha256": hashlib.sha256(
            config.STRUCTURED_ANSWER_SYSTEM.encode("utf-8")
        ).hexdigest(),
        "claim_support_judge_system_sha256": hashlib.sha256(
            VERIFIER_SYSTEM.encode("utf-8")
        ).hexdigest(),
        "claim_support_judge_response_format_sha256": canonical_sha256(
            VERIFIER_RESPONSE_FORMAT
        ),
        "required_coverage_judge_system_sha256": hashlib.sha256(
            COVERAGE_JUDGE_SYSTEM.encode("utf-8")
        ).hexdigest(),
        "required_coverage_judge_response_format_sha256": canonical_sha256(
            COVERAGE_JUDGE_RESPONSE_FORMAT
        ),
        "runtime_policy_sha256": canonical_sha256(config.CURRENT_RUNTIME_POLICY),
    }
    dataset = {
        "path": args.cases.relative_to(ROOT).as_posix()
        if args.cases.is_relative_to(ROOT)
        else str(args.cases),
        "sha256": sha256_file(args.cases),
        "case_count": len(cases),
        "split_counts": dict(
            sorted(
                (split, sum(case["split"] == split for case in cases))
                for split in {case["split"] for case in cases}
            )
        ),
    }
    models = {
        "generator": generator_profile["model"],
        "generator_profile": generator_profile,
        "judge": config.JUDGE_MODEL,
        "structured_gen_max_tokens": generator_profile["max_tokens"],
        "generator_request_options": generator_profile["request_options"],
        "judge_request_options": config.JUDGE_REQUEST_OPTIONS,
    }
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "dataset": dataset,
        "case_ids": [case["id"] for case in cases],
        "models": models,
        "retrieval": {"top_k": args.top_k, "unique_headings": True},
        "runtime_supported_actions": list(config.CURRENT_SUPPORTED_ACTIONS),
        "runtime_policy": config.CURRENT_RUNTIME_POLICY,
        "selected_lanes": sorted(selected_lanes),
        "prompt_hashes": prompt_hashes,
        "development_overrides": {
            "allow_unreviewed": args.allow_unreviewed,
            "skip_judge": args.skip_judge,
        },
    }
    identity_sha256 = canonical_sha256(identity)
    progress_path = args.out.with_suffix(args.out.suffix + ".progress.json")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity_sha256") != identity_sha256:
            raise SystemExit(
                f"Refusing incompatible checkpoint {progress_path}; move it or use a different --out."
            )
        items = progress["items"]
        provenance = progress["provenance"]
    else:
        items = []
        provenance = capture_run_provenance(llm_enabled=True)
        progress = {
            "schema_version": "1.0.0",
            "identity_sha256": identity_sha256,
            "identity": identity,
            "provenance": provenance,
            "items": items,
        }
        write_json_atomic(progress_path, progress)

    index = load_index()
    completed_ids = {item["id"] for item in items}
    for position, case in enumerate(cases, 1):
        if case["id"] in completed_ids:
            print(f"[{position}/{len(cases)}] {case['id']}: resumed", flush=True)
            continue
        try:
            item = evaluate_case(
                case,
                index,
                skip_judge=args.skip_judge,
                top_k=args.top_k,
                generator_profile=generator_profile,
            )
        except Exception as exc:  # fail the item closed while preserving the batch
            item = {
                "id": case["id"],
                "split": case["split"],
                "lane": case["lane"],
                "severity": case["severity"],
                "expected_action": expected_action(case),
                "runtime_supports_expected_action": expected_action(case)
                in config.CURRENT_SUPPORTED_ACTIONS,
                "actual_action": None,
                "contract_valid": False,
                "action_correct": False,
                "case_pass": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        items.append(item)
        progress["items"] = items
        write_json_atomic(progress_path, progress)
        print(f"[{position}/{len(cases)}] {case['id']}: "
              f"{'valid' if item['contract_valid'] else 'FAIL'}", flush=True)

    evaluation_eligible = (
        not args.allow_unreviewed
        and not args.skip_judge
        and generator_profile["provider_pinned"]
        and not unsupported_case_ids
    )
    release_eligible = evaluation_eligible and {case["split"] for case in cases} == {
        "release"
    }
    blinded_release_eligible = release_eligible and all(
        case["authoring"]["method"]
        in {"human", "model_proposed_human_verified", "model_proposed_human_rewritten"}
        and case["authoring"].get("independent_of_corpus_authors") is True
        and "human" in case["review"].get("reviewer_kinds", ["human"])
        for case in cases
    )
    report = {
        "schema_version": "2.0.0-dev",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_eligible": evaluation_eligible,
        "release_eligible": release_eligible,
        "blinded_release_eligible": blinded_release_eligible,
        "release_limitations": (
            [
                "same-agent AI-authored/AI-reviewed cases are not independent human validation"
            ]
            if {case["split"] for case in cases} == {"release"}
            and not blinded_release_eligible
            else []
        ),
        "development_overrides": {
            "allow_unreviewed": args.allow_unreviewed,
            "skip_judge": args.skip_judge,
        },
        "dataset": dataset,
        "models": models,
        "retrieval": identity["retrieval"],
        "runtime_supported_actions": identity["runtime_supported_actions"],
        "runtime_policy": identity["runtime_policy"],
        "selected_lanes": identity["selected_lanes"],
        "prompt_hashes": prompt_hashes,
        "run_identity": identity,
        "run_identity_sha256": identity_sha256,
        "summary": aggregate(items),
        "provenance": provenance,
        "items": items,
    }
    write_json_atomic(args.out, report)
    progress_path.unlink(missing_ok=True)
    print(json.dumps(report["summary"], indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
