"""Selective-answering curves and fail-closed threshold selection."""
from __future__ import annotations

import math


def binary_auc(rows: list[dict], feature: str) -> float | None:
    positives = [row[feature] for row in rows if row["answerability"] == "answerable"]
    negatives = [row[feature] for row in rows if row["answerability"] == "unanswerable"]
    if not positives or not negatives:
        return None
    wins = sum(left > right for left in positives for right in negatives)
    ties = sum(left == right for left in positives for right in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def risk_coverage_curve(rows: list[dict], feature: str) -> list[dict]:
    if not rows:
        raise ValueError("cannot calibrate abstention on an empty dataset")
    if any(feature not in row for row in rows):
        raise ValueError(f"missing feature: {feature}")
    answerable_total = sum(row["answerability"] == "answerable" for row in rows)
    unanswerable_total = len(rows) - answerable_total
    if not answerable_total or not unanswerable_total:
        raise ValueError("calibration requires answerable and unanswerable cases")

    thresholds = [math.inf, *sorted({float(row[feature]) for row in rows}, reverse=True)]
    curve = []
    for threshold in thresholds:
        answered = [row for row in rows if float(row[feature]) >= threshold]
        false_answers = sum(
            row["answerability"] == "unanswerable" for row in answered
        )
        answered_answerable = sum(
            row["answerability"] == "answerable" for row in answered
        )
        false_refusals = answerable_total - answered_answerable
        curve.append(
            {
                "threshold": None if math.isinf(threshold) else threshold,
                "answered_cases": len(answered),
                "coverage": len(answered) / len(rows),
                "selective_risk": false_answers / len(answered) if answered else 0.0,
                "false_answer_count": false_answers,
                "false_answer_rate": false_answers / unanswerable_total,
                "false_answer_rate_95ci": wilson_interval(false_answers, unanswerable_total),
                "false_refusal_count": false_refusals,
                "false_refusal_rate": false_refusals / answerable_total,
                "false_refusal_rate_95ci": wilson_interval(false_refusals, answerable_total),
            }
        )
    return curve


def select_operating_point(
    curve: list[dict],
    *,
    max_false_answer_rate: float,
    max_false_refusal_rate: float,
) -> dict | None:
    feasible = [
        row
        for row in curve
        if row["false_answer_rate"] <= max_false_answer_rate
        and row["false_refusal_rate"] <= max_false_refusal_rate
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (row["coverage"], -row["selective_risk"], row["threshold"] or math.inf),
    )


__all__ = [
    "binary_auc",
    "risk_coverage_curve",
    "select_operating_point",
    "wilson_interval",
]
