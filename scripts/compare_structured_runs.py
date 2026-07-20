"""Compare two structured V2 reports with safety-lane guardrails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SAFETY_LANES = {
    "authorization",
    "false_premise",
    "hard_negative",
    "prompt_injection",
    "temporal_conflict",
    "unanswerable",
}
HARD_METRICS = (
    "contract_valid_rate",
    "action_accuracy",
    "citation_existence_precision",
    "claim_valid_citation_coverage",
    "judge_output_valid_rate",
    "verifier_output_valid_rate",
    "end_to_end_case_pass_rate",
)
QUALITY_METRICS = (
    "generated_claim_support_rate",
    "required_claim_coverage_rate",
    "retrieval_any_gold_evidence_rate",
    "retrieval_complete_gold_evidence_rate",
    "retrieval_required_claim_coverage",
)


def compare_reports(baseline: dict, candidate: dict) -> dict:
    for field in ("dataset", "models", "prompt_hashes", "runtime_supported_actions"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"reports do not match on {field}")
    for field in ("inputs_sha256", "index_sha256"):
        if baseline.get("provenance", {}).get(field) != candidate.get("provenance", {}).get(field):
            raise ValueError(f"reports do not match on provenance {field}")
    baseline_provider_models = _provider_models(baseline)
    candidate_provider_models = _provider_models(candidate)
    if baseline_provider_models != candidate_provider_models:
        raise ValueError("reports used different provider-returned model identities")
    baseline_items = {item["id"]: item for item in baseline["items"]}
    candidate_items = {item["id"]: item for item in candidate["items"]}
    if set(baseline_items) != set(candidate_items):
        raise ValueError("reports have different case-ID sets")

    deltas = {}
    for metric in (*HARD_METRICS, *QUALITY_METRICS):
        left = baseline["summary"].get(metric)
        right = candidate["summary"].get(metric)
        deltas[metric] = None if left is None or right is None else round(right - left, 6)

    wins = sorted(
        case_id
        for case_id in baseline_items
        if baseline_items[case_id].get("case_pass") is False
        and candidate_items[case_id].get("case_pass") is True
    )
    losses = sorted(
        case_id
        for case_id in baseline_items
        if baseline_items[case_id].get("case_pass") is True
        and candidate_items[case_id].get("case_pass") is False
    )
    lane_deltas = {}
    reasons = []
    baseline_lanes = baseline["summary"].get("by_lane", {})
    candidate_lanes = candidate["summary"].get("by_lane", {})
    if set(baseline_lanes) != set(candidate_lanes):
        raise ValueError("reports have different lane sets")
    for lane in sorted(baseline_lanes):
        left = baseline_lanes[lane]
        right = candidate_lanes[lane]
        if left["case_count"] != right["case_count"]:
            raise ValueError(f"lane case count differs: {lane}")
        pass_delta = right["case_pass"] - left["case_pass"]
        action_delta = right["action_correct"] - left["action_correct"]
        lane_deltas[lane] = {
            "case_pass_delta": pass_delta,
            "action_correct_delta": action_delta,
        }
        if lane in SAFETY_LANES and pass_delta < 0:
            reasons.append(f"safety lane {lane} lost {-pass_delta} passing case(s)")
        if lane in SAFETY_LANES and action_delta < 0:
            reasons.append(f"safety lane {lane} lost {-action_delta} correct action(s)")

    for metric in HARD_METRICS:
        if deltas[metric] is not None and deltas[metric] < 0:
            reasons.append(f"hard metric regressed: {metric} {deltas[metric]:+.6f}")

    release_eligible = bool(
        baseline.get("release_eligible") and candidate.get("release_eligible")
    )
    if reasons:
        decision = "REJECT"
    elif not release_eligible:
        decision = "DEVELOPMENT_REVIEW"
    else:
        decision = "PROMOTE"
    return {
        "decision": decision,
        "release_eligible_comparison": release_eligible,
        "baseline_retrieval": baseline.get("retrieval"),
        "candidate_retrieval": candidate.get("retrieval"),
        "provider_returned_models": baseline_provider_models,
        "metric_deltas": deltas,
        "case_level": {
            "wins": wins,
            "losses": losses,
            "unchanged": len(baseline_items) - len(wins) - len(losses),
        },
        "lane_deltas": lane_deltas,
        "rejection_reasons": reasons,
    }


def _provider_models(report: dict) -> dict[str, list[str | None]]:
    found = {"generator": set(), "support_judge": set(), "coverage_judge": set()}
    for item in report.get("items", []):
        generation = item.get("generation", {}).get("calls") or [
            item.get("generation", {}).get("call")
        ]
        support = item.get("claim_verifier", {}).get("calls") or [
            item.get("claim_verifier", {}).get("call")
        ]
        coverage = item.get("claim_judge", {}).get("calls") or [
            item.get("claim_judge", {}).get("call")
        ]
        for call in generation:
            if isinstance(call, dict):
                found["generator"].add(call.get("response_model"))
        for call in support:
            if isinstance(call, dict):
                found["support_judge"].add(call.get("response_model"))
        for call in coverage:
            if isinstance(call, dict):
                found["coverage_judge"].add(call.get("response_model"))
    return {key: sorted(values, key=lambda value: "" if value is None else value) for key, values in found.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare_reports(baseline, candidate)
    report["baseline_path"] = str(args.baseline)
    report["candidate_path"] = str(args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["decision"] == "PROMOTE" else 2 if report["decision"] == "REJECT" else 1)


if __name__ == "__main__":
    main()
