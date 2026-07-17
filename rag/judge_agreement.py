"""Deterministic human-versus-judge agreement reporting by error type."""
from __future__ import annotations

from collections import Counter
import hashlib
import json


def answer_sha256(answer: object) -> str:
    encoded = json.dumps(
        answer, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left = Counter(left for left, _ in pairs)
    right = Counter(right for _, right in pairs)
    labels = set(left) | set(right)
    expected = sum((left[label] / len(pairs)) * (right[label] / len(pairs)) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def compare_judgments(human_records: list[dict], eval_items: list[dict]) -> dict:
    by_case = {item["id"]: item for item in eval_items}
    generated_pairs: list[tuple[str, str]] = []
    required_pairs: list[tuple[str, str]] = []
    disagreements: list[dict] = []
    missing_cases: list[str] = []
    hash_mismatches: list[str] = []

    for human in human_records:
        case_id = human["case_id"]
        item = by_case.get(case_id)
        judgment = (item or {}).get("claim_judge", {}).get("judgment")
        if not judgment:
            missing_cases.append(case_id)
            continue
        if human.get("answer_sha256") != answer_sha256(item.get("answer")):
            hash_mismatches.append(case_id)
            continue
        judge_generated = {
            row["claim_id"]: row["status"] for row in judgment["claim_verdicts"]
        }
        judge_required = {
            row["claim_id"]: row["status"]
            for row in judgment["required_claim_verdicts"]
        }
        _collect(
            case_id,
            "generated_support",
            human["claim_labels"],
            judge_generated,
            generated_pairs,
            disagreements,
        )
        _collect(
            case_id,
            "required_coverage",
            human["required_claim_labels"],
            judge_required,
            required_pairs,
            disagreements,
        )

    generated_false_accepts = sum(
        row["lane"] == "generated_support"
        and row["judge"] == "supported"
        and row["human"] != "supported"
        for row in disagreements
    )
    required_false_accepts = sum(
        row["lane"] == "required_coverage"
        and row["judge"] == "covered"
        and row["human"] != "covered"
        for row in disagreements
    )
    return {
        "human_record_count": len(human_records),
        "matched_human_record_count": len(human_records) - len(missing_cases) - len(hash_mismatches),
        "missing_judgment_case_ids": sorted(missing_cases),
        "answer_hash_mismatch_case_ids": sorted(hash_mismatches),
        "generated_support": _summary(generated_pairs, generated_false_accepts),
        "required_coverage": _summary(required_pairs, required_false_accepts),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }


def _collect(case_id, lane, human_rows, judge_by_id, pairs, disagreements):
    for row in human_rows:
        claim_id = row["claim_id"]
        human_status = row["status"]
        judge_status = judge_by_id.get(claim_id)
        if judge_status is None:
            disagreements.append(
                {
                    "case_id": case_id,
                    "lane": lane,
                    "claim_id": claim_id,
                    "human": human_status,
                    "judge": None,
                    "kind": "missing_judge_verdict",
                }
            )
            continue
        pairs.append((human_status, judge_status))
        if human_status != judge_status:
            disagreements.append(
                {
                    "case_id": case_id,
                    "lane": lane,
                    "claim_id": claim_id,
                    "human": human_status,
                    "judge": judge_status,
                    "kind": "label_disagreement",
                }
            )


def _summary(pairs, false_accepts):
    total = len(pairs)
    agreements = sum(left == right for left, right in pairs)
    return {
        "label_count": total,
        "exact_agreement": agreements / total if total else None,
        "cohen_kappa": cohen_kappa(pairs),
        "false_accept_count": false_accepts,
        "false_accept_rate": false_accepts / total if total else None,
        "confusion": dict(sorted(Counter(f"human={h}|judge={j}" for h, j in pairs).items())),
    }


__all__ = ["answer_sha256", "cohen_kappa", "compare_judgments"]
