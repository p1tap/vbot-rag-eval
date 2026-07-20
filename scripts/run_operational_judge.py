"""Run the selected no-human judge policy over a frozen generation report."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag.provenance import sha256_file  # noqa: E402
from scripts.build_judge_calibration_queue import build_queue  # noqa: E402
from scripts.run_judge_bakeoff import (  # noqa: E402
    canonical_sha256,
    group_tasks,
    judge_group,
    load_jsonl,
    select_profiles,
    write_atomic,
)
from scripts.simulate_judge_cascade import simulate_policy  # noqa: E402
from scripts.run_structured_eval import load_cases  # noqa: E402

DEFAULT_CASES = ROOT / "evals" / "v2" / "reviewed" / "vbot-human-reviewed-100.jsonl"
DEFAULT_GENERATION = ROOT / "reports" / "v2" / "structured-vbot-human-100-gpt54-generation.json"
DEFAULT_POLICY = ROOT / "reports" / "v2" / "judge-cascade-simulation.json"
DEFAULT_OUT = ROOT / "reports" / "v2" / "structured-vbot-human-100-final.json"


def write_json(path: Path, value: object) -> None:
    write_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def records_path(out: Path, profile_id: str) -> Path:
    return out.parent / f"{out.stem}.judge-{profile_id}.jsonl"


def profile_records(
    profile: dict,
    groups: list[list[dict]],
    out: Path,
    *,
    retry_invalid: bool,
) -> list[dict]:
    path = records_path(out, profile["id"])
    existing = load_jsonl(path)
    by_group = {row["group_id"]: row for row in existing}
    for position, group in enumerate(groups, start=1):
        group_id = f"{group[0]['case_id']}:{'support' if group[0]['task_type'] == 'generated_claim_support' else 'coverage'}"
        old = by_group.get(group_id)
        if old and (old.get("valid") or not retry_invalid):
            continue
        result = judge_group(group, profile, retry_budget=retry_invalid)
        if old:
            prior = {key: value for key, value in old.items() if key != "retry_history"}
            result["retry_history"] = [*old.get("retry_history", []), prior]
        by_group[group_id] = result
        ordered = [
            by_group[f"{item[0]['case_id']}:{'support' if item[0]['task_type'] == 'generated_claim_support' else 'coverage'}"]
            for item in groups
            if f"{item[0]['case_id']}:{'support' if item[0]['task_type'] == 'generated_claim_support' else 'coverage'}" in by_group
        ]
        write_atomic(
            path,
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in ordered
            ),
        )
        print(
            f"{profile['id']} [{position}/{len(groups)}] {group_id}: "
            f"{'valid' if result['valid'] else 'INVALID'}",
            flush=True,
        )
    missing = [
        f"{group[0]['case_id']}:{'support' if group[0]['task_type'] == 'generated_claim_support' else 'coverage'}"
        for group in groups
        if f"{group[0]['case_id']}:{'support' if group[0]['task_type'] == 'generated_claim_support' else 'coverage'}" not in by_group
    ]
    if missing:
        raise SystemExit(f"profile {profile['id']} is incomplete")
    return [
        by_group[f"{group[0]['case_id']}:{'support' if group[0]['task_type'] == 'generated_claim_support' else 'coverage'}"]
        for group in groups
    ]


def apply_case_metrics(
    generation: dict, predictions: dict[str, dict]
) -> tuple[list[dict], dict]:
    items = []
    for source in generation["items"]:
        answer = source.get("answer") or {}
        support_ids = [
            f"{source['id']}:support:{claim['id']}"
            for claim in answer.get("claims", [])
        ]
        coverage_prefix = f"{source['id']}:coverage:"
        coverage_ids = [task_id for task_id in predictions if task_id.startswith(coverage_prefix)]
        support_pass = bool(support_ids) and all(
            predictions.get(task_id, {}).get("decision") == "supported"
            for task_id in support_ids
        )
        coverage_pass = bool(coverage_ids) and all(
            predictions[task_id]["decision"] == "covered" for task_id in coverage_ids
        )
        if source["expected_action"] == "answer":
            case_pass = bool(
                source.get("contract_valid")
                and source.get("action_correct")
                and support_pass
                and coverage_pass
            )
        else:
            case_pass = bool(source.get("contract_valid") and source.get("action_correct"))
        items.append(
            {
                "id": source["id"],
                "split": source.get("split"),
                "lane": source["lane"],
                "severity": source["severity"],
                "expected_action": source["expected_action"],
                "actual_action": source.get("actual_action"),
                "contract_valid": source.get("contract_valid", False),
                "action_correct": source.get("action_correct", False),
                "support_task_count": len(support_ids),
                "support_pass": support_pass if support_ids else None,
                "coverage_task_count": len(coverage_ids),
                "coverage_pass": coverage_pass if coverage_ids else None,
                "case_pass": case_pass,
                "generation_item_sha256": canonical_sha256(source),
            }
        )
    total = len(items)
    summary = {
        "case_count": total,
        "contract_valid_rate": sum(row["contract_valid"] for row in items) / total,
        "action_accuracy": sum(row["action_correct"] for row in items) / total,
        "end_to_end_case_pass_rate": sum(row["case_pass"] for row in items) / total,
        "answerable_case_count": sum(row["expected_action"] == "answer" for row in items),
        "support_pass_rate": (
            sum(row["support_pass"] is True for row in items)
            / sum(row["support_pass"] is not None for row in items)
        ),
        "coverage_pass_rate": (
            sum(row["coverage_pass"] is True for row in items)
            / sum(row["coverage_pass"] is not None for row in items)
        ),
        "failure_case_ids": [row["id"] for row in items if not row["case_pass"]],
        "by_lane": {},
    }
    for lane in sorted({row["lane"] for row in items}):
        selected = [row for row in items if row["lane"] == lane]
        summary["by_lane"][lane] = {
            "case_count": len(selected),
            "case_pass_count": sum(row["case_pass"] for row in selected),
            "case_pass_rate": sum(row["case_pass"] for row in selected) / len(selected),
        }
    return items, summary


def direct_profile_predictions(
    grouped: dict[str, list[dict]], profile_records_map: dict[str, dict]
) -> dict[str, dict]:
    """Expand one diagnostic profile, failing every invalid group closed."""
    predictions = {}
    for group_id, tasks in grouped.items():
        record = profile_records_map[group_id]
        verdicts = {
            verdict["task_id"]: verdict for verdict in record.get("verdicts", [])
        }
        for task in tasks:
            verdict = verdicts.get(task["task_id"])
            predictions[task["task_id"]] = (
                verdict
                if record.get("valid") and verdict is not None
                else {
                    "task_id": task["task_id"],
                    "decision": "fail_closed",
                    "evidence": [],
                    "linked_claim_ids": [],
                    "rationale": "local diagnostic judge group was invalid or incomplete",
                }
            )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=DEFAULT_GENERATION)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--profiles", default="")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="finalize one explicitly configured local diagnostic; never release-eligible",
    )
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    generation = json.loads(args.generation.read_text(encoding="utf-8"))
    if sha256_file(args.cases) != generation["dataset"]["sha256"]:
        raise SystemExit("generation report and reviewed dataset hash differ")
    cases = load_cases(args.cases)
    tasks, metadata = build_queue(generation, cases)
    groups = group_tasks(tasks)
    policy_report = json.loads(args.policy.read_text(encoding="utf-8"))
    selected_policy_id = policy_report["selection"]["selected_policy_id"]
    selected_definition = policy_report["policies"][selected_policy_id]["definition"]
    profiles = select_profiles(args.profiles)
    if args.profiles:
        required_ids = {profile["id"] for profile in profiles}
    else:
        required_ids = {profile["id"] for profile in config.JUDGE_BAKEOFF_PROFILES}
    records = {}
    artifact_records = {}
    for profile in profiles:
        rows = profile_records(
            profile, groups, args.out, retry_invalid=args.retry_invalid
        )
        records[profile["id"]] = {row["group_id"]: row for row in rows}
        path = records_path(args.out, profile["id"])
        artifact_records[profile["id"]] = {
            "path": path.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "valid_group_count": sum(row["valid"] for row in rows),
            "group_count": len(rows),
        }
    if required_ids != set(records):
        raise SystemExit("selected profile records are incomplete")
    production_ids = {profile["id"] for profile in config.JUDGE_BAKEOFF_PROFILES}
    diagnostic_ids = {
        profile["id"]
        for profile in getattr(config, "LOCAL_JUDGE_DIAGNOSTIC_PROFILES", ())
    }
    is_diagnostic = set(records) <= diagnostic_ids and len(records) == 1
    # The cascade simulator expects the full cross-family roster. An explicitly
    # requested local profile may instead produce a non-promotable diagnostic.
    if set(records) != production_ids and not (args.diagnostic_only and is_diagnostic):
        print("profile checkpoint complete; rerun without --profiles to finalize")
        return
    grouped = {
        f"{group[0]['case_id']}:{'support' if group[0]['task_type'] == 'generated_claim_support' else 'coverage'}": group
        for group in groups
    }
    if is_diagnostic:
        profile_id = next(iter(records))
        predictions = direct_profile_predictions(grouped, records[profile_id])
        simulation = {
            "operation": {
                "mode": "single_same_family_local_diagnostic",
                "profile_id": profile_id,
                "task_count": len(predictions),
                "fail_closed_task_count": sum(
                    row["decision"] == "fail_closed" for row in predictions.values()
                ),
            }
        }
    else:
        simulation = simulate_policy(selected_definition, grouped, records)
        predictions = simulation.pop("predictions")
    items, summary = apply_case_metrics(generation, predictions)
    calls = [
        call
        for profile_records_map in records.values()
        for record in profile_records_map.values()
        for call in record.get("calls", [])
    ]
    costs = [float(call["cost"]) for call in calls if call.get("cost") is not None]
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_eligible": not is_diagnostic,
        "blinded_release_eligible": False,
        "blinded_release_blocker": "all 100 Vbot cases are visible development cases",
        "inputs": {
            "generation_path": args.generation.resolve().relative_to(ROOT).as_posix(),
            "generation_sha256": sha256_file(args.generation),
            "dataset_path": args.cases.resolve().relative_to(ROOT).as_posix(),
            "dataset_sha256": sha256_file(args.cases),
            "policy_path": args.policy.resolve().relative_to(ROOT).as_posix(),
            "policy_sha256": sha256_file(args.policy),
        },
        "judge_policy": (
            {
                "selected_policy_id": next(iter(records)),
                "definition": {"mode": "single_same_family_local_diagnostic"},
                "review_mode": "ai_only_same_family_diagnostic",
            }
            if is_diagnostic
            else {
                "selected_policy_id": selected_policy_id,
                "definition": selected_definition,
                "review_mode": policy_report["review_mode"],
            }
        ),
        "task_workload": metadata["summary"],
        "summary": summary,
        "operation": {
            **simulation["operation"],
            "experiment_call_count": len(calls),
            "experiment_provider_reported_cost_total": sum(costs) if costs else None,
        },
        "artifacts": {"profile_records": artifact_records},
        "task_predictions_sha256": canonical_sha256(predictions),
        "items": items,
        "limitations": [
            "The domain suite is owner-reviewed but entirely visible development data.",
            "The automated policy is calibrated to 175 frozen human verdicts plus any separately declared AI overrides.",
            "AI-only adjudication is not represented as independent human review.",
            *(
                [
                    "The local judge is the same model family as the generator and is not independent validation.",
                    "This diagnostic cannot promote a release or replace the selected cross-family cascade.",
                ]
                if is_diagnostic
                else []
            ),
        ],
    }
    write_json(args.out, report)
    print(json.dumps(summary, indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
