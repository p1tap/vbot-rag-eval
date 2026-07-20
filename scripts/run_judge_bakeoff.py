"""Run frozen support/coverage tasks through the pinned judge bakeoff roster."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag.claim_verifier import (  # noqa: E402
    VERIFIER_RESPONSE_FORMAT,
    VERIFIER_SYSTEM,
    support_payload,
    validate_verification,
)
from rag.coverage_judge import (  # noqa: E402
    COVERAGE_JUDGE_RESPONSE_FORMAT,
    COVERAGE_JUDGE_SYSTEM,
    coverage_payload,
    validate_coverage_judgment,
)
from rag.judge_agreement import cohen_kappa  # noqa: E402
from rag.llm import chat_with_metadata  # noqa: E402
from rag.provenance import sha256_file  # noqa: E402

DEFAULT_MANIFEST = ROOT / "evals" / "v2" / "judge-calibration" / "manifest.json"
DEFAULT_LABELS = ROOT / "evals" / "v2" / "judge-calibration" / "human-labels.jsonl"
DEFAULT_ROOT = ROOT / "reports" / "v2" / "judge-bakeoff"
DEFAULT_SUMMARY = ROOT / "reports" / "v2" / "judge-bakeoff-summary.json"
RUNNER_VERSION = "3"
SUPPORT_MAX_TOKENS = 1600
COVERAGE_MAX_TOKENS = 1200
RETRY_SUPPORT_MAX_TOKENS = 5000
RETRY_COVERAGE_MAX_TOKENS = 4000
HTTP_RETRIES = 6
SEMANTIC_ATTEMPTS = 2


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def labels_by_task(labels: list[dict]) -> dict[str, dict]:
    result = {}
    for case in labels:
        for row in case["claim_labels"]:
            task_id = f"{case['case_id']}:support:{row['claim_id']}"
            result[task_id] = {"decision": row["status"], "rationale": row["rationale"]}
        for row in case["required_claim_labels"]:
            task_id = f"{case['case_id']}:coverage:{row['claim_id']}"
            result[task_id] = {"decision": row["status"], "rationale": row["rationale"]}
    return result


def select_profiles(value: str) -> list[dict]:
    production = list(config.JUDGE_BAKEOFF_PROFILES)
    if not value:
        return production
    diagnostic = list(getattr(config, "LOCAL_JUDGE_DIAGNOSTIC_PROFILES", ()))
    experimental = list(getattr(config, "EXPERIMENTAL_JUDGE_PROFILES", ()))
    available = [*production, *experimental, *diagnostic]
    wanted = {item.strip() for item in value.split(",") if item.strip()}
    profiles = [item for item in available if item["id"] in wanted]
    unknown = wanted - {item["id"] for item in profiles}
    if unknown:
        raise ValueError(f"unknown profiles: {sorted(unknown)}")
    if not profiles:
        raise ValueError("no judge profiles selected")
    return profiles


def group_tasks(tasks: list[dict]) -> list[list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for task in tasks:
        groups.setdefault((task["case_id"], task["task_type"]), []).append(task)
    return list(groups.values())


def group_fingerprint(tasks: list[dict], profile: dict) -> str:
    task_type = tasks[0]["task_type"]
    contract = (
        {
            "system": VERIFIER_SYSTEM,
            "response_format": VERIFIER_RESPONSE_FORMAT,
        }
        if task_type == "generated_claim_support"
        else {
            "system": COVERAGE_JUDGE_SYSTEM,
            "response_format": COVERAGE_JUDGE_RESPONSE_FORMAT,
        }
    )
    return canonical_sha256(
        {
            "runner_version": RUNNER_VERSION,
            "tasks": tasks,
            "profile": profile,
            "contract": contract,
            "call_policy": {
                "support_max_tokens": SUPPORT_MAX_TOKENS,
                "coverage_max_tokens": COVERAGE_MAX_TOKENS,
                "http_retries": HTTP_RETRIES,
                "semantic_attempts": SEMANTIC_ATTEMPTS,
                "temperature": None,
            },
        }
    )


def _support_inputs(tasks: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    first = tasks[0]
    answer = {
        "claims": [
            {
                "id": task["generated_claim"]["id"],
                "text": task["generated_claim"]["text"],
                "citation_ids": task["generated_claim"]["citation_ids"],
            }
            for task in tasks
        ]
    }
    source_by_id = {}
    for task in tasks:
        for source in task["cited_sources"]:
            previous = source_by_id.setdefault(source["citation_id"], source)
            if previous != source:
                raise ValueError(f"inconsistent source {source['citation_id']} in support group")
    sources = list(source_by_id.values())
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": support_payload(first["question"], answer, sources)},
    ]
    return messages, answer, sources


def _coverage_inputs(tasks: list[dict]) -> tuple[list[dict], list[dict], dict]:
    first = tasks[0]
    required = [task["required_claim"] for task in tasks]
    answer = {
        "claims": [
            {"id": claim["id"], "text": claim["text"]}
            for claim in first["generated_claims"]
        ]
    }
    messages = [
        {"role": "system", "content": COVERAGE_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": coverage_payload(first["question"], required, answer),
        },
    ]
    return messages, required, answer


def judge_group(tasks: list[dict], profile: dict, *, retry_budget: bool = False) -> dict:
    first = tasks[0]
    task_type = first["task_type"]
    if any(
        task["case_id"] != first["case_id"] or task["task_type"] != task_type
        for task in tasks
    ):
        raise ValueError("judge groups must contain one case and one task type")
    calls: list[dict] = []
    raw_outputs: list[str] = []
    errors: list[dict | str] = []
    final = None
    for semantic_attempt in range(1, SEMANTIC_ATTEMPTS + 1):
        try:
            if task_type == "generated_claim_support":
                messages, answer, sources = _support_inputs(tasks)
                response_format = VERIFIER_RESPONSE_FORMAT
                max_tokens = (
                    RETRY_SUPPORT_MAX_TOKENS if retry_budget else SUPPORT_MAX_TOKENS
                )
            else:
                messages, required, answer = _coverage_inputs(tasks)
                response_format = COVERAGE_JUDGE_RESPONSE_FORMAT
                max_tokens = (
                    RETRY_COVERAGE_MAX_TOKENS if retry_budget else COVERAGE_MAX_TOKENS
                )
            call = chat_with_metadata(
                profile["model"],
                messages,
                max_tokens=max_tokens,
                temperature=None,
                retries=HTTP_RETRIES,
                response_format=response_format,
                request_options=profile["request_options"],
            )
            calls.append(call.to_record())
            raw_outputs.append(call.content)
            try:
                judgment = json.loads(call.content)
            except json.JSONDecodeError as exc:
                errors = [{"code": "invalid_json", "message": str(exc)}]
                continue
            if task_type == "generated_claim_support":
                validation_errors, _ = validate_verification(judgment, answer, sources)
                if validation_errors:
                    errors = list(validation_errors)
                    continue
                final = [
                    {
                        "task_id": task["task_id"],
                        "decision": verdict["status"],
                        "evidence": verdict["evidence"],
                        "linked_claim_ids": [],
                        "rationale": verdict["rationale"],
                    }
                    for task, verdict in zip(tasks, judgment["claim_verdicts"])
                ]
            else:
                validation_errors, _ = validate_coverage_judgment(
                    judgment, required, answer
                )
                if validation_errors:
                    errors = list(validation_errors)
                    continue
                generated_ids = [f"g{index}" for index in range(1, len(answer["claims"]) + 1)]
                generated_to_original = {
                    generated_id: claim["id"]
                    for generated_id, claim in zip(generated_ids, answer["claims"])
                }
                final = [
                    {
                        "task_id": task["task_id"],
                        "decision": verdict["status"],
                        "evidence": [],
                        "linked_claim_ids": [
                            generated_to_original[item]
                            for item in verdict["generated_claim_ids"]
                        ],
                        "rationale": verdict["rationale"],
                    }
                    for task, verdict in zip(tasks, judgment["required_claim_verdicts"])
                ]
            errors = []
            break
        except Exception as exc:  # noqa: BLE001 - retain group failure and continue
            errors = [f"{type(exc).__name__}: {exc}"]

    return {
        "schema_version": "1.0.0",
        "group_id": f"{first['case_id']}:{'support' if task_type == 'generated_claim_support' else 'coverage'}",
        "task_type": task_type,
        "case_id": first["case_id"],
        "lane": first["lane"],
        "severity": first["severity"],
        "answer_sha256": first["answer_sha256"],
        "tasks": [
            {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "case_id": task["case_id"],
                "lane": task["lane"],
                "severity": task["severity"],
            }
            for task in tasks
        ],
        "profile_id": profile["id"],
        "profile_sha256": canonical_sha256(profile),
        "group_fingerprint_sha256": group_fingerprint(tasks, profile),
        "valid": final is not None,
        "verdicts": final or [],
        "errors": errors,
        "semantic_attempts": len(raw_outputs),
        "execution_policy": {
            "retry_budget": retry_budget,
            "max_tokens": max_tokens,
            "semantic_attempt_limit": SEMANTIC_ATTEMPTS,
            "http_retries": HTTP_RETRIES,
        },
        "calls": calls,
        "raw_outputs": raw_outputs,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def score_records(
    records: list[dict], human: dict[str, dict], partitions: dict[str, str]
) -> dict:
    expanded = []
    for record in records:
        verdict_by_task = {
            row["task_id"]: row for row in record.get("verdicts", [])
        }
        for task in record["tasks"]:
            expanded.append(
                {
                    **task,
                    "valid": record["valid"] and task["task_id"] in verdict_by_task,
                    "verdict": verdict_by_task.get(task["task_id"]),
                }
            )
    valid = [row for row in expanded if row["valid"]]
    pairs = [
        (human[row["task_id"]]["decision"], row["verdict"]["decision"])
        for row in valid
    ]
    confusion = Counter(f"human={left}|judge={right}" for left, right in pairs)

    def is_accept(task_type: str, decision: str) -> bool:
        return decision == ("supported" if task_type == "generated_claim_support" else "covered")

    false_accepts = []
    false_rejects = []
    disagreements = []
    for row in valid:
        expected = human[row["task_id"]]["decision"]
        actual = row["verdict"]["decision"]
        if expected != actual:
            disagreement = {
                "task_id": row["task_id"],
                "case_id": row["case_id"],
                "task_type": row["task_type"],
                "lane": row["lane"],
                "severity": row["severity"],
                "partition": partitions[row["case_id"]],
                "human": expected,
                "judge": actual,
                "human_rationale": human[row["task_id"]]["rationale"],
                "judge_rationale": row["verdict"]["rationale"],
            }
            disagreements.append(disagreement)
            if is_accept(row["task_type"], actual) and not is_accept(row["task_type"], expected):
                false_accepts.append(disagreement)
            if is_accept(row["task_type"], expected) and not is_accept(row["task_type"], actual):
                false_rejects.append(disagreement)

    calls = [call for row in records for call in row.get("calls", [])]
    historical_calls = [
        call
        for row in records
        for attempt in row.get("retry_history", [])
        for call in attempt.get("calls", [])
    ]
    latencies = [float(call["latency_ms"]) for call in calls if call.get("latency_ms") is not None]
    costs = [float(call["cost"]) for call in calls if call.get("cost") is not None]
    usage = Counter()
    for call in calls:
        for key, value in call.get("usage", {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
    total = len(expanded)
    first_pass_valid_tasks = 0
    retried_group_count = 0
    for record in records:
        retry_history = record.get("retry_history", [])
        if retry_history:
            retried_group_count += 1
            first_result = retry_history[0]
        else:
            first_result = record
        if first_result.get("valid"):
            first_pass_valid_tasks += len(record["tasks"])
    agreement = sum(left == right for left, right in pairs) / len(pairs) if pairs else None
    false_accept_rate = len(false_accepts) / len(pairs) if pairs else None
    high_critical_false_accepts = sum(
        row["severity"] in {"high", "critical"} for row in false_accepts
    )
    return {
        "task_count": total,
        "valid_task_count": len(valid),
        "validity_rate": len(valid) / total if total else None,
        "first_pass_valid_task_count": first_pass_valid_tasks,
        "first_pass_validity_rate": first_pass_valid_tasks / total if total else None,
        "retried_group_count": retried_group_count,
        "exact_agreement": agreement,
        "cohen_kappa": cohen_kappa(pairs),
        "false_accept_count": len(false_accepts),
        "false_accept_rate": false_accept_rate,
        "false_reject_count": len(false_rejects),
        "false_reject_rate": len(false_rejects) / len(pairs) if pairs else None,
        "high_critical_false_accept_count": high_critical_false_accepts,
        "confusion": dict(sorted(confusion.items())),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "call_count": len(calls),
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
        "reported_cost_total": sum(costs) if costs else None,
        "experiment_call_count_including_retries": len(calls) + len(historical_calls),
        "experiment_reported_cost_total_including_retries": (
            sum(
                float(call["cost"])
                for call in [*historical_calls, *calls]
                if call.get("cost") is not None
            )
            if any(call.get("cost") is not None for call in [*historical_calls, *calls])
            else None
        ),
        "usage_totals": dict(sorted(usage.items())),
        "eligibility_targets": {
            "validity_100_percent": len(valid) == total,
            "zero_high_critical_false_accepts": high_critical_false_accepts == 0,
            "false_accept_rate_at_most_2_percent": (
                false_accept_rate is not None and false_accept_rate <= 0.02
            ),
            "exact_agreement_at_least_90_percent": (
                agreement is not None and agreement >= 0.90
            ),
            "cohen_kappa_at_least_0_80": (
                cohen_kappa(pairs) is not None and cohen_kappa(pairs) >= 0.80
            ),
        },
    }


def sliced_summary(records: list[dict], human: dict, partitions: dict) -> dict:
    result = {"overall": score_records(records, human, partitions)}
    for task_type in ("generated_claim_support", "required_claim_coverage"):
        selected = [row for row in records if row["task_type"] == task_type]
        result[task_type] = score_records(selected, human, partitions)
    for partition in ("development", "confirmation"):
        selected = [row for row in records if partitions[row["case_id"]] == partition]
        result[partition] = score_records(selected, human, partitions)
    return result


def _checkpoint_text(records: list[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in records
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--human-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--profiles", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--pace-seconds", type=float, default=0.0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    queue_path = ROOT / manifest["queue"]["path"]
    if sha256_file(queue_path) != manifest["queue"]["sha256"]:
        raise SystemExit("frozen calibration queue hash mismatch")
    tasks = load_jsonl(queue_path)
    if args.limit:
        tasks = tasks[: args.limit]
    groups = group_tasks(tasks)
    human = labels_by_task(load_jsonl(args.human_labels))
    missing = {task["task_id"] for task in tasks} - set(human)
    if missing:
        raise SystemExit(f"human labels are missing tasks: {sorted(missing)[:5]}")
    profiles = select_profiles(args.profiles)
    if args.validate_only:
        print(json.dumps({
            "tasks": len(tasks),
            "judge_calls_per_profile": len(groups),
            "profiles": [row["id"] for row in profiles],
        }))
        return

    if args.pace_seconds < 0:
        raise SystemExit("--pace-seconds cannot be negative")
    run_id = f"judge-bakeoff-{manifest['queue']['sha256'][:12]}-r{RUNNER_VERSION}"
    profile_summaries = {}
    for profile in profiles:
        checkpoint = args.out_root / run_id / f"{profile['id']}.jsonl"
        existing = load_jsonl(checkpoint) if checkpoint.exists() else []
        by_group = {row["group_id"]: row for row in existing}
        for group in groups:
            group_id = f"{group[0]['case_id']}:{'support' if group[0]['task_type'] == 'generated_claim_support' else 'coverage'}"
            old = by_group.get(group_id)
            expected_fingerprint = group_fingerprint(group, profile)
            if old and old.get("group_fingerprint_sha256") != expected_fingerprint:
                raise SystemExit(f"stale checkpoint for {profile['id']} {group_id}")
            if old and (old.get("valid") or not args.retry_invalid):
                continue
            result = judge_group(group, profile, retry_budget=args.retry_invalid)
            if old:
                prior = {
                    key: value
                    for key, value in old.items()
                    if key != "retry_history"
                }
                result["retry_history"] = [*old.get("retry_history", []), prior]
            by_group[group_id] = result
            ordered = [
                by_group[f"{item[0]['case_id']}:{'support' if item[0]['task_type'] == 'generated_claim_support' else 'coverage'}"]
                for item in groups
                if f"{item[0]['case_id']}:{'support' if item[0]['task_type'] == 'generated_claim_support' else 'coverage'}" in by_group
            ]
            write_atomic(checkpoint, _checkpoint_text(ordered))
            print(
                f"{profile['id']} {len(ordered)}/{len(groups)} {group_id} "
                f"({len(group)} tasks) "
                f"{'valid' if result['valid'] else 'INVALID'}",
                flush=True,
            )
            if args.pace_seconds:
                import time

                time.sleep(args.pace_seconds)
        records = [
            by_group[f"{group[0]['case_id']}:{'support' if group[0]['task_type'] == 'generated_claim_support' else 'coverage'}"]
            for group in groups
        ]
        profile_summaries[profile["id"]] = {
            "profile": profile,
            "profile_sha256": canonical_sha256(profile),
            "records_path": checkpoint.relative_to(ROOT).as_posix(),
            "records_sha256": sha256_file(checkpoint),
            "metrics": sliced_summary(records, human, manifest["case_partitions"]),
        }

    summary = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_roster": {
            "configured": [row["id"] for row in config.JUDGE_BAKEOFF_PROFILES],
            "included": [row["id"] for row in profiles],
            "omitted": [
                row["id"]
                for row in config.JUDGE_BAKEOFF_PROFILES
                if row["id"] not in {profile["id"] for profile in profiles}
            ],
            "complete": {row["id"] for row in profiles}
            == {row["id"] for row in config.JUDGE_BAKEOFF_PROFILES},
            "diagnostic_only": [
                row["id"]
                for row in profiles
                if row["id"]
                in {
                    item["id"]
                    for item in getattr(config, "LOCAL_JUDGE_DIAGNOSTIC_PROFILES", ())
                }
            ],
        },
        "frozen_inputs": {
            "manifest_path": args.manifest.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(args.manifest),
            "queue_sha256": manifest["queue"]["sha256"],
            "human_labels_path": args.human_labels.relative_to(ROOT).as_posix(),
            "human_labels_sha256": sha256_file(args.human_labels),
            "task_count": len(tasks),
            "judge_call_group_count": len(groups),
        },
        "profiles": profile_summaries,
        "bounded_invalid_retry_policy": {
            "support_max_tokens": RETRY_SUPPORT_MAX_TOKENS,
            "coverage_max_tokens": RETRY_COVERAGE_MAX_TOKENS,
            "semantic_attempts": SEMANTIC_ATTEMPTS,
            "only_initially_invalid_groups": True,
        },
    }
    write_atomic(args.summary, summary)
    print(json.dumps({key: value["metrics"]["overall"] for key, value in profile_summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
