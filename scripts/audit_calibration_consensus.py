"""Audit frozen human calibration labels against collapsed model-family consensus."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.run_judge_bakeoff import labels_by_task, load_jsonl, score_records  # noqa: E402

DEFAULT_BAKEOFF = ROOT / "reports" / "v2" / "judge-bakeoff-summary.json"
DEFAULT_LABELS = ROOT / "evals" / "v2" / "judge-calibration" / "human-labels.jsonl"
DEFAULT_MANIFEST = ROOT / "evals" / "v2" / "judge-calibration" / "manifest.json"
DEFAULT_REPORT = ROOT / "reports" / "v2" / "judge-calibration-consensus-audit.json"
DEFAULT_OVERRIDES = (
    ROOT / "evals" / "v2" / "judge-calibration" / "ai-adjudication-overrides.jsonl"
)

FAMILIES = {
    "deepseek_v4": ("v4-flash-high", "v4-flash-max"),
    "gemini": ("gemini-3.5-flash-high",),
    "gpt": ("gpt-5.4-high",),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_profile_records(bakeoff: dict) -> dict[str, list[dict]]:
    output = {}
    for profile_id, profile in bakeoff["profiles"].items():
        path = ROOT / profile["records_path"]
        if sha256_file(path) != profile["records_sha256"]:
            raise SystemExit(f"record hash mismatch for {profile_id}")
        output[profile_id] = load_jsonl(path)
    return output


def task_decisions(records: list[dict]) -> dict[str, dict]:
    decisions = {}
    for record in records:
        if not record.get("valid"):
            continue
        for verdict in record["verdicts"]:
            decisions[verdict["task_id"]] = {
                "decision": verdict["decision"],
                "rationale": verdict["rationale"],
                "group_id": record["group_id"],
            }
    return decisions


def family_vote(
    task_id: str, profile_maps: dict[str, dict], profiles: tuple[str, ...]
) -> tuple[str | None, dict]:
    available = {
        profile_id: profile_maps[profile_id][task_id]
        for profile_id in profiles
        if task_id in profile_maps.get(profile_id, {})
    }
    labels = {row["decision"] for row in available.values()}
    return (next(iter(labels)) if len(labels) == 1 else None), available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bakeoff", type=Path, default=DEFAULT_BAKEOFF)
    parser.add_argument("--human-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    args = parser.parse_args()

    bakeoff = json.loads(args.bakeoff.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    human = labels_by_task(load_jsonl(args.human_labels))
    records = load_profile_records(bakeoff)
    missing_families = {
        family
        for family, profiles in FAMILIES.items()
        if not any(profile in records for profile in profiles)
    }
    if missing_families:
        raise SystemExit(f"bakeoff is missing model families: {sorted(missing_families)}")
    profile_maps = {profile: task_decisions(rows) for profile, rows in records.items()}

    rows = []
    overrides = []
    adjusted = {task_id: dict(label) for task_id, label in human.items()}
    for task_id, human_label in sorted(human.items()):
        votes = {}
        profile_evidence = {}
        for family, profiles in FAMILIES.items():
            vote, evidence = family_vote(task_id, profile_maps, profiles)
            if vote is not None:
                votes[family] = vote
            profile_evidence[family] = evidence
        unanimous = len(votes) == len(FAMILIES) and len(set(votes.values())) == 1
        consensus = next(iter(votes.values())) if unanimous else None
        conflict = bool(consensus and consensus != human_label["decision"])
        row = {
            "task_id": task_id,
            "human_decision": human_label["decision"],
            "human_rationale": human_label["rationale"],
            "family_votes": votes,
            "profile_evidence": profile_evidence,
            "three_family_unanimous": unanimous,
            "consensus_decision": consensus,
            "human_consensus_conflict": conflict,
        }
        rows.append(row)
        if conflict:
            override = {
                "schema_version": "1.0.0",
                "task_id": task_id,
                "original_human_decision": human_label["decision"],
                "ai_adjudicated_decision": consensus,
                "adjudication_basis": "three_distinct_model_families_unanimous",
                "family_votes": votes,
                "review_provenance": {
                    "reviewer_kind": "multi_model_ai_consensus",
                    "locally_human_reviewed": False,
                },
            }
            overrides.append(override)
            adjusted[task_id] = {
                "decision": consensus,
                "rationale": "Three collapsed model families unanimously disagreed with the frozen human label.",
            }

    original_metrics = {
        profile: score_records(profile_records, human, manifest["case_partitions"])
        for profile, profile_records in records.items()
    }
    adjusted_metrics = {
        profile: score_records(profile_records, adjusted, manifest["case_partitions"])
        for profile, profile_records in records.items()
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ai_consensus_audit_complete",
        "inputs": {
            "bakeoff_sha256": sha256_file(args.bakeoff),
            "human_labels_sha256": sha256_file(args.human_labels),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "provenance": {
            "frozen_human_labels_preserved": True,
            "audit_reviewer_kind": "multi_model_ai_consensus",
            "locally_human_reviewed": False,
            "deepseek_profiles_collapsed_to_one_family_vote": True,
            "deepseek_profiles_available": [
                profile for profile in FAMILIES["deepseek_v4"] if profile in records
            ],
            "configured_bakeoff_roster_complete": bakeoff.get(
                "profile_roster", {}
            ).get("complete", True),
        },
        "summary": {
            "task_count": len(rows),
            "three_family_unanimous_count": sum(row["three_family_unanimous"] for row in rows),
            "human_consensus_conflict_count": len(overrides),
            "override_transitions": dict(
                sorted(
                    Counter(
                        f"{row['original_human_decision']}->{row['ai_adjudicated_decision']}"
                        for row in overrides
                    ).items()
                )
            ),
        },
        "metrics_against_original_human_labels": original_metrics,
        "metrics_against_ai_adjudicated_reference": adjusted_metrics,
        "tasks": rows,
        "limitations": [
            "The frozen human labels are preserved; overrides are a separate AI artifact.",
            "The same model families appear in the bakeoff and the consensus, so adjusted metrics are diagnostic and partially circular.",
            "Consensus cannot establish human-equivalent ground truth or a population error rate.",
            "Only unanimous agreement across DeepSeek, Gemini, and GPT overrides a human decision.",
            "DeepSeek V4 high is sufficient for the DeepSeek family vote when V4 max is unavailable; profile availability is recorded explicitly.",
        ],
    }
    write_jsonl(args.overrides, overrides)
    report["override_artifact"] = {
        "path": args.overrides.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(args.overrides),
        "count": len(overrides),
    }
    write_json(args.report, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
