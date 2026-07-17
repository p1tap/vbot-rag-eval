"""Simulate no-human judge cascades against the frozen calibration labels.

Policy selection uses only the development partition. The confirmation
partition is reported after selection and never participates in ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.judge_agreement import cohen_kappa  # noqa: E402
from rag.provenance import sha256_file  # noqa: E402
from scripts.run_judge_bakeoff import labels_by_task, load_jsonl  # noqa: E402

DEFAULT_BAKEOFF = ROOT / "reports" / "v2" / "judge-bakeoff-summary.json"
DEFAULT_LABELS = ROOT / "evals" / "v2" / "judge-calibration" / "human-labels.jsonl"
DEFAULT_MANIFEST = ROOT / "evals" / "v2" / "judge-calibration" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "reports" / "v2" / "judge-cascade-simulation.json"
DEFAULT_DECISION = ROOT / "reports" / "v2" / "judge-bakeoff-decision.md"

PRIMARY = "v4-flash-high"
SECONDARY = "gemini-3.5-flash-high"
ADJUDICATOR = "gpt-5.4-high"
MAX_PROFILE = "v4-flash-max"
REQUIRED_PROFILES = {PRIMARY, SECONDARY, ADJUDICATOR}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_records(summary: dict) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for profile_id, profile in summary["profiles"].items():
        path = ROOT / profile["records_path"]
        if sha256_file(path) != profile["records_sha256"]:
            raise SystemExit(f"bakeoff record hash mismatch: {profile_id}")
        result[profile_id] = {row["group_id"]: row for row in load_jsonl(path)}
    missing = REQUIRED_PROFILES - set(result)
    if missing:
        raise SystemExit(f"bakeoff is missing required profiles: {sorted(missing)}")
    return result


def verdicts(record: dict | None) -> dict[str, str]:
    if not record or not record.get("valid"):
        return {}
    return {row["task_id"]: row["decision"] for row in record["verdicts"]}


def fallback_decision(task_type: str) -> str:
    if task_type == "generated_claim_support":
        return "unsupported"
    return "missing"


def accept_decision(task_type: str) -> str:
    if task_type == "generated_claim_support":
        return "supported"
    return "covered"


def stable_sample(group_id: str, rate: float) -> bool:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16)
    return bucket / 0xFFFFFFFF < rate


def call_stats(record: dict) -> tuple[float, float]:
    cost = 0.0
    latency = 0.0
    for call in record.get("calls", []):
        if call.get("cost") is not None:
            cost += float(call["cost"])
        if call.get("latency_ms") is not None:
            latency += float(call["latency_ms"])
    return cost, latency


def _use(
    consulted: list[str],
    profile_id: str,
    group_id: str,
    records: dict[str, dict[str, dict]],
) -> dict | None:
    if profile_id not in records:
        return None
    consulted.append(profile_id)
    return records[profile_id].get(group_id)


def resolve_with_adjudicator(
    tasks: list[dict],
    primary: dict | None,
    secondary: dict | None,
    adjudicator: dict | None,
    maximum: dict | None = None,
) -> tuple[dict[str, str], set[str]]:
    """Resolve a group conservatively; an automated accept needs two votes."""
    sources = [verdicts(primary), verdicts(secondary)]
    adjudicated = verdicts(adjudicator)
    maxed = verdicts(maximum)
    output: dict[str, str] = {}
    fail_closed: set[str] = set()
    for task in tasks:
        task_id = task["task_id"]
        task_type = task["task_type"]
        available = [source[task_id] for source in sources if task_id in source]
        if len(available) == 2 and available[0] == available[1]:
            output[task_id] = available[0]
            continue
        candidate = adjudicated.get(task_id)
        if candidate is None:
            candidate = maxed.get(task_id)
        if candidate is None:
            output[task_id] = fallback_decision(task_type)
            fail_closed.add(task_id)
            continue
        if candidate == accept_decision(task_type):
            supporting_accept = any(value == candidate for value in available)
            if not supporting_accept:
                output[task_id] = fallback_decision(task_type)
                fail_closed.add(task_id)
                continue
        output[task_id] = candidate
    return output, fail_closed


def simulate_policy(
    policy: dict,
    groups: dict[str, list[dict]],
    records: dict[str, dict[str, dict]],
) -> dict:
    predictions: dict[str, dict] = {}
    consulted_calls: set[tuple[str, str]] = set()
    group_latencies: list[float] = []
    for group_id, tasks in groups.items():
        consulted: list[str] = []
        fail_closed: set[str] = set()
        mode = policy["mode"]
        if mode == "single":
            profile_id = policy["profile"]
            record = _use(consulted, profile_id, group_id, records)
            chosen = verdicts(record)
            decisions = {}
            for task in tasks:
                task_id = task["task_id"]
                if task_id in chosen:
                    decisions[task_id] = chosen[task_id]
                else:
                    decisions[task_id] = fallback_decision(task["task_type"])
                    fail_closed.add(task_id)
        elif mode == "full_panel":
            primary = _use(consulted, PRIMARY, group_id, records)
            secondary = _use(consulted, SECONDARY, group_id, records)
            adjudicator = _use(consulted, ADJUDICATOR, group_id, records)
            maximum = _use(consulted, MAX_PROFILE, group_id, records)
            decisions, fail_closed = resolve_with_adjudicator(
                tasks, primary, secondary, adjudicator, maximum
            )
        else:
            primary = _use(consulted, PRIMARY, group_id, records)
            primary_verdicts = verdicts(primary)
            high_critical = any(
                task["severity"] in {"high", "critical"} for task in tasks
            )
            primary_failure = (
                not primary_verdicts
                or any(
                    primary_verdicts.get(task["task_id"])
                    != accept_decision(task["task_type"])
                    for task in tasks
                )
            )
            if mode == "severity_sample":
                consult_secondary = (
                    high_critical
                    or stable_sample(group_id, policy["routine_sample_rate"])
                    or not primary_verdicts
                )
            elif mode == "primary_failure":
                consult_secondary = primary_failure
            elif mode == "disagreement":
                consult_secondary = True
            else:
                raise ValueError(f"unknown policy mode: {mode}")

            if not consult_secondary:
                decisions = {}
                for task in tasks:
                    task_id = task["task_id"]
                    if task_id in primary_verdicts:
                        decisions[task_id] = primary_verdicts[task_id]
                    else:
                        decisions[task_id] = fallback_decision(task["task_type"])
                        fail_closed.add(task_id)
            else:
                secondary = _use(consulted, SECONDARY, group_id, records)
                secondary_verdicts = verdicts(secondary)
                needs_adjudication = (
                    not primary_verdicts
                    or not secondary_verdicts
                    or any(
                        primary_verdicts.get(task["task_id"])
                        != secondary_verdicts.get(task["task_id"])
                        for task in tasks
                    )
                )
                adjudicator = None
                maximum = None
                if needs_adjudication:
                    adjudicator = _use(consulted, ADJUDICATOR, group_id, records)
                    if not verdicts(adjudicator):
                        maximum = _use(consulted, MAX_PROFILE, group_id, records)
                decisions, fail_closed = resolve_with_adjudicator(
                    tasks, primary, secondary, adjudicator, maximum
                )

        path_latency = 0.0
        for profile_id in consulted:
            consulted_calls.add((profile_id, group_id))
            record = records[profile_id].get(group_id)
            if record:
                _, latency = call_stats(record)
                path_latency += latency
        group_latencies.append(path_latency)
        for task in tasks:
            predictions[task["task_id"]] = {
                "decision": decisions[task["task_id"]],
                "fail_closed": task["task_id"] in fail_closed,
                "profiles_consulted": list(consulted),
                "group_id": group_id,
                "case_id": task["case_id"],
                "task_type": task["task_type"],
                "lane": task["lane"],
                "severity": task["severity"],
            }

    total_cost = 0.0
    for profile_id, group_id in consulted_calls:
        record = records[profile_id][group_id]
        cost, _ = call_stats(record)
        total_cost += cost
    ordered_latencies = sorted(group_latencies)
    return {
        "predictions": predictions,
        "operation": {
            "unique_group_call_count": len(consulted_calls),
            "reported_cost_total": total_cost,
            "group_path_latency_ms": {
                "mean": statistics.mean(group_latencies) if group_latencies else None,
                "p50": percentile(ordered_latencies, 0.50),
                "p95": percentile(ordered_latencies, 0.95),
            },
            "profile_group_calls": dict(
                sorted(Counter(profile for profile, _ in consulted_calls).items())
            ),
        },
    }


def percentile(ordered: list[float], fraction: float) -> float | None:
    if not ordered:
        return None
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def wilson_upper(failures: int, total: int, z: float = 1.6448536269514722) -> float | None:
    """One-sided 95% Wilson upper bound for a binomial rate."""
    if total == 0:
        return None
    probability = failures / total
    denominator = 1 + z * z / total
    center = probability + z * z / (2 * total)
    radius = z * math.sqrt(
        probability * (1 - probability) / total + z * z / (4 * total * total)
    )
    return min(1.0, (center + radius) / denominator)


def score(
    predictions: dict[str, dict],
    human: dict[str, dict],
    partitions: dict[str, str],
    partition: str | None,
) -> dict:
    comparisons = []
    for task_id, row in predictions.items():
        if partition is not None and partitions[row["case_id"]] != partition:
            continue
        expected = human[task_id]["decision"]
        actual = row["decision"]
        comparisons.append((task_id, row, expected, actual))
    pairs = [(expected, actual) for _, _, expected, actual in comparisons]
    errors = []
    false_accepts = []
    false_rejects = []
    for task_id, row, expected, actual in comparisons:
        if expected == actual:
            continue
        detail = {
            "task_id": task_id,
            "case_id": row["case_id"],
            "task_type": row["task_type"],
            "lane": row["lane"],
            "severity": row["severity"],
            "human": expected,
            "cascade": actual,
            "fail_closed": row["fail_closed"],
        }
        errors.append(detail)
        accept = accept_decision(row["task_type"])
        if actual == accept and expected != accept:
            false_accepts.append(detail)
        if expected == accept and actual != accept:
            false_rejects.append(detail)
    total = len(comparisons)
    agreement = sum(left == right for left, right in pairs) / total if total else None
    return {
        "task_count": total,
        "exact_agreement": agreement,
        "cohen_kappa": cohen_kappa(pairs),
        "false_accept_count": len(false_accepts),
        "false_accept_rate": len(false_accepts) / total if total else None,
        "false_accept_rate_one_sided_95pct_upper": wilson_upper(len(false_accepts), total),
        "high_critical_false_accept_count": sum(
            row["severity"] in {"high", "critical"} for row in false_accepts
        ),
        "false_reject_count": len(false_rejects),
        "false_reject_rate": len(false_rejects) / total if total else None,
        "fail_closed_count": sum(row["fail_closed"] for _, row, _, _ in comparisons),
        "disagreement_count": len(errors),
        "confusion": dict(sorted(Counter(f"human={left}|cascade={right}" for left, right in pairs).items())),
        "errors": errors,
    }


def rank_key(policy: dict) -> tuple:
    metrics = policy["metrics"]["development"]
    operation = policy["operation"]
    return (
        metrics["high_critical_false_accept_count"],
        metrics["false_accept_count"],
        metrics["false_reject_count"],
        metrics["disagreement_count"],
        metrics["fail_closed_count"],
        operation["reported_cost_total"],
        operation["group_path_latency_ms"]["mean"] or float("inf"),
    )


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def decision_markdown(report: dict) -> str:
    selected_id = report["selection"]["selected_policy_id"]
    selected = report["policies"][selected_id]
    reference_note = (
        "The calibration reference includes separately recorded unanimous "
        "three-family AI overrides; the original human labels remain frozen."
        if report["inputs"].get("ai_override_sha256")
        else "The calibration reference is the original frozen human-label artifact."
    )
    lines = [
        "# Judge bakeoff and cascade decision",
        "",
        reference_note,
        "",
        f"Decision date: {report['completed_at_utc'][:10]}",
        "",
        "## Decision",
        "",
        f"Selected policy: `{selected_id}`.",
        "",
        "The policy was selected using only the frozen development partition. "
        "Ranking minimizes high/critical false accepts, total false accepts, "
        "false rejects, categorical disagreements, fail-closed decisions, cost, "
        "and latency—in that order. The confirmation partition was opened only "
        "after the policy identity was fixed.",
        "",
        "No future human-review queue is part of this policy. Unresolved or "
        "invalid automated adjudication fails closed and is recorded as AI "
        "review, never relabeled as human review.",
        "",
        "## Policy comparison",
        "",
        "| Policy | Dev agreement | Dev false accepts | Confirmation agreement | Confirmation false accepts | Calls | Reported cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_id, policy in report["policies"].items():
        dev = policy["metrics"]["development"]
        confirmation = policy["metrics"]["confirmation"]
        operation = policy["operation"]
        lines.append(
            f"| `{policy_id}` | {format_percent(dev['exact_agreement'])} | "
            f"{dev['false_accept_count']} | {format_percent(confirmation['exact_agreement'])} | "
            f"{confirmation['false_accept_count']} | {operation['unique_group_call_count']} | "
            f"${operation['reported_cost_total']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Selected-policy result",
            "",
            f"- Development: {format_percent(selected['metrics']['development']['exact_agreement'])} "
            f"agreement; {selected['metrics']['development']['false_accept_count']} false accepts.",
            f"- Hidden confirmation: {format_percent(selected['metrics']['confirmation']['exact_agreement'])} "
            f"agreement; {selected['metrics']['confirmation']['false_accept_count']} false accepts.",
            f"- Overall: {format_percent(selected['metrics']['overall']['exact_agreement'])} agreement; "
            f"{selected['metrics']['overall']['false_accept_count']} false accepts and "
            f"{selected['metrics']['overall']['false_reject_count']} false rejects.",
            f"- One-sided 95% upper bound on the observed overall false-accept rate: "
            f"{format_percent(selected['metrics']['overall']['false_accept_rate_one_sided_95pct_upper'])}.",
            "",
            "This calibration set is deliberately failure-rich and small. Its "
            "point metrics select an operating policy; they do not prove the "
            "same error rate on unseen production traffic.",
            "",
            "## Frozen evidence",
            "",
            f"- Bakeoff summary SHA-256: `{report['inputs']['bakeoff_sha256']}`",
            f"- Human labels SHA-256: `{report['inputs']['human_labels_sha256']}`",
            f"- Calibration manifest SHA-256: `{report['inputs']['manifest_sha256']}`",
        ]
    )
    if report["inputs"].get("ai_override_sha256"):
        lines.append(
            f"- AI adjudication overrides SHA-256: `{report['inputs']['ai_override_sha256']}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bakeoff", type=Path, default=DEFAULT_BAKEOFF)
    parser.add_argument("--human-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    args = parser.parse_args()

    bakeoff = json.loads(args.bakeoff.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    human = labels_by_task(load_jsonl(args.human_labels))
    if args.overrides:
        for override in load_jsonl(args.overrides):
            task_id = override["task_id"]
            if task_id not in human:
                raise SystemExit(f"override references unknown task: {task_id}")
            if human[task_id]["decision"] != override["original_human_decision"]:
                raise SystemExit(f"override human decision drifted: {task_id}")
            human[task_id] = {
                "decision": override["ai_adjudicated_decision"],
                "rationale": "Three-family AI consensus override; see adjudication artifact.",
            }
    records = load_records(bakeoff)
    groups: dict[str, list[dict]] = {}
    reference = records[PRIMARY]
    for group_id, record in reference.items():
        groups[group_id] = record["tasks"]
    expected_tasks = {task["task_id"] for tasks in groups.values() for task in tasks}
    if expected_tasks != set(human):
        raise SystemExit("bakeoff and human-label task universes differ")
    for profile_id, profile_records in records.items():
        if set(profile_records) != set(groups):
            raise SystemExit(f"incomplete group universe for {profile_id}")

    definitions = [
        {"id": f"single-{profile_id}", "mode": "single", "profile": profile_id}
        for profile_id in sorted(records)
    ] + [
        {
            "id": "v4-primary-severity-sample",
            "mode": "severity_sample",
            "routine_sample_rate": 0.10,
        },
        {"id": "v4-primary-failure-cascade", "mode": "primary_failure"},
        {"id": "v4-gemini-disagreement-gpt", "mode": "disagreement"},
        {"id": "conservative-full-panel", "mode": "full_panel"},
    ]

    policies = {}
    for definition in definitions:
        simulation = simulate_policy(definition, groups, records)
        predictions = simulation.pop("predictions")
        policies[definition["id"]] = {
            "definition": definition,
            **simulation,
            "metrics": {
                "development": score(predictions, human, manifest["case_partitions"], "development"),
                "confirmation": score(predictions, human, manifest["case_partitions"], "confirmation"),
                "overall": score(predictions, human, manifest["case_partitions"], None),
            },
            "predictions_sha256": canonical_sha256(predictions),
        }

    ranking = sorted(policies, key=lambda policy_id: rank_key(policies[policy_id]))
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_mode": (
            "frozen_human_calibration_with_explicit_ai_consensus_overrides"
            if args.overrides
            else "frozen_human_calibration_then_ai_only_operation"
        ),
        "inputs": {
            "bakeoff_path": args.bakeoff.relative_to(ROOT).as_posix(),
            "bakeoff_sha256": sha256_file(args.bakeoff),
            "human_labels_path": args.human_labels.relative_to(ROOT).as_posix(),
            "human_labels_sha256": sha256_file(args.human_labels),
            "manifest_path": args.manifest.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(args.manifest),
            "ai_override_path": (
                args.overrides.resolve().relative_to(ROOT).as_posix()
                if args.overrides
                else None
            ),
            "ai_override_sha256": sha256_file(args.overrides) if args.overrides else None,
        },
        "selection": {
            "partition_used": "development",
            "confirmation_used_for_selection": False,
            "priority": [
                "high_critical_false_accept_count",
                "false_accept_count",
                "false_reject_count",
                "categorical_disagreement_count",
                "fail_closed_count",
                "reported_cost_total",
                "mean_group_path_latency_ms",
            ],
            "ranking": ranking,
            "selected_policy_id": ranking[0],
        },
        "policies": policies,
    }
    write_json(args.output, report)
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    args.decision.write_text(decision_markdown(report), encoding="utf-8")
    selected = policies[ranking[0]]
    print(
        json.dumps(
            {
                "selected_policy": ranking[0],
                "development": selected["metrics"]["development"],
                "confirmation": selected["metrics"]["confirmation"],
                "overall": selected["metrics"]["overall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
