from __future__ import annotations

import unittest

from rag.abstention import (
    binary_auc,
    risk_coverage_curve,
    select_operating_point,
    wilson_interval,
)


class AbstentionCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"id": "a1", "answerability": "answerable", "score": 0.9},
            {"id": "a2", "answerability": "answerable", "score": 0.8},
            {"id": "u1", "answerability": "unanswerable", "score": 0.7},
            {"id": "u2", "answerability": "unanswerable", "score": 0.6},
        ]

    def test_auc_is_one_for_separated_scores(self) -> None:
        self.assertEqual(binary_auc(self.rows, "score"), 1.0)

    def test_curve_includes_answer_none_and_answer_all(self) -> None:
        curve = risk_coverage_curve(self.rows, "score")
        self.assertIsNone(curve[0]["threshold"])
        self.assertEqual(curve[0]["coverage"], 0.0)
        self.assertEqual(curve[-1]["coverage"], 1.0)
        self.assertEqual(curve[-1]["false_answer_rate"], 1.0)
        self.assertEqual(curve[-1]["false_refusal_rate"], 0.0)

    def test_selector_maximizes_coverage_within_both_guardrails(self) -> None:
        selected = select_operating_point(
            risk_coverage_curve(self.rows, "score"),
            max_false_answer_rate=0.0,
            max_false_refusal_rate=0.0,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["threshold"], 0.8)
        self.assertEqual(selected["coverage"], 0.5)

    def test_selector_returns_none_when_tradeoff_is_impossible(self) -> None:
        overlapping = [
            {"answerability": "answerable", "score": 0.8},
            {"answerability": "unanswerable", "score": 0.9},
        ]
        selected = select_operating_point(
            risk_coverage_curve(overlapping, "score"),
            max_false_answer_rate=0.0,
            max_false_refusal_rate=0.0,
        )
        self.assertIsNone(selected)

    def test_wilson_interval_is_honest_for_zero_of_ten(self) -> None:
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.27)
        self.assertLess(high, 0.28)


if __name__ == "__main__":
    unittest.main()
