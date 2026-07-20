"""Audit a structured generator's answer/refuse actions on frozen cases."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.abstention import wilson_interval  # noqa: E402
from rag.provenance import sha256_file  # noqa: E402

DEFAULT_GENERATION = (
    ROOT / "reports" / "v2" / "structured-vbot-human-100-qwen35-generation.json"
)
DEFAULT_OUT = ROOT / "reports" / "v2" / "vbot-human-100-qwen35-action-policy.json"


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def analyze(
    generation: dict,
    *,
    max_false_answer_rate: float,
    max_false_refusal_rate: float,
) -> dict:
    items = generation.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("generation report contains no items")

    rows = []
    for item in items:
        expected = item.get("expected_action")
        expected_answer = expected == "answer"
        delivered = (
            item.get("actual_action")
            if item.get("contract_valid")
            else "invalid_failclosed"
        )
        answered = delivered == "answer"
        rows.append(
            {
                "id": item["id"],
                "lane": item.get("lane"),
                "expected_action": expected,
                "delivered_action": delivered,
                "contract_valid": bool(item.get("contract_valid")),
                "action_correct": delivered == expected,
                "false_answer": not expected_answer and answered,
                "false_refusal": expected_answer and not answered,
            }
        )

    answerable = sum(row["expected_action"] == "answer" for row in rows)
    unanswerable = len(rows) - answerable
    false_answers = sum(row["false_answer"] for row in rows)
    false_refusals = sum(row["false_refusal"] for row in rows)
    action_correct = sum(row["action_correct"] for row in rows)
    contract_valid = sum(row["contract_valid"] for row in rows)
    false_answer_rate = _rate(false_answers, unanswerable)
    false_refusal_rate = _rate(false_refusals, answerable)
    passes = bool(
        false_answer_rate is not None
        and false_refusal_rate is not None
        and false_answer_rate <= max_false_answer_rate
        and false_refusal_rate <= max_false_refusal_rate
    )

    by_lane = {}
    for lane in sorted({row["lane"] for row in rows if row["lane"]}):
        selected = [row for row in rows if row["lane"] == lane]
        lane_answerable = sum(row["expected_action"] == "answer" for row in selected)
        lane_unanswerable = len(selected) - lane_answerable
        lane_false_answers = sum(row["false_answer"] for row in selected)
        lane_false_refusals = sum(row["false_refusal"] for row in selected)
        by_lane[lane] = {
            "case_count": len(selected),
            "answerable_count": lane_answerable,
            "unanswerable_count": lane_unanswerable,
            "action_correct_count": sum(row["action_correct"] for row in selected),
            "false_answer_count": lane_false_answers,
            "false_answer_rate": _rate(lane_false_answers, lane_unanswerable),
            "false_refusal_count": lane_false_refusals,
            "false_refusal_rate": _rate(lane_false_refusals, lane_answerable),
        }

    return {
        "sample": {
            "case_count": len(rows),
            "answerable_count": answerable,
            "unanswerable_count": unanswerable,
        },
        "metrics": {
            "contract_valid_count": contract_valid,
            "contract_valid_rate": _rate(contract_valid, len(rows)),
            "action_correct_count": action_correct,
            "action_accuracy": _rate(action_correct, len(rows)),
            "false_answer_count": false_answers,
            "false_answer_rate": false_answer_rate,
            "false_answer_rate_95ci": wilson_interval(false_answers, unanswerable),
            "false_refusal_count": false_refusals,
            "false_refusal_rate": false_refusal_rate,
            "false_refusal_rate_95ci": wilson_interval(false_refusals, answerable),
        },
        "guardrails": {
            "max_false_answer_rate": max_false_answer_rate,
            "max_false_refusal_rate": max_false_refusal_rate,
            "decision_basis": "observed development-set rates; confidence intervals reported separately",
            "passes": passes,
        },
        "failure_case_ids": [row["id"] for row in rows if not row["action_correct"]],
        "by_lane": by_lane,
        "items": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=DEFAULT_GENERATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-false-answer-rate", type=float, default=0.05)
    parser.add_argument("--max-false-refusal-rate", type=float, default=0.15)
    args = parser.parse_args()
    for value in (args.max_false_answer_rate, args.max_false_refusal_rate):
        if value < 0 or value > 1:
            raise SystemExit("guardrails must be between zero and one")

    generation = json.loads(args.generation.read_text(encoding="utf-8"))
    analysis = analyze(
        generation,
        max_false_answer_rate=args.max_false_answer_rate,
        max_false_refusal_rate=args.max_false_refusal_rate,
    )
    report = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_policy_accepted"
        if analysis["guardrails"]["passes"]
        else "development_policy_rejected",
        "release_eligible": False,
        "release_blockers": (
            [
                "the AI-reviewed release set is not a production-traffic sample",
                "the release set lacks independent human author/reviewer validation",
                "the action report does not independently judge answer claim support or required-claim coverage",
            ]
            if set((generation.get("dataset") or {}).get("split_counts", {}))
            == {"release"}
            else [
                "the 100 owner-reviewed cases are a visible development partition, not a blinded release set",
                "the generator action policy has not been validated on adversarial/real-user partitions",
            ]
        ),
        "policy": {
            "kind": "structured_generator_action_with_invalid_output_failed_closed",
            "threshold": None,
            "retrieval_similarity_threshold": "rejected; see vbot-human-100-abstention-calibration-dev.json",
        },
        "source": {
            "path": args.generation.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(args.generation),
            "dataset": generation.get("dataset"),
            "generator_profile": generation.get("generator_profile"),
        },
        **analysis,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], **analysis["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
