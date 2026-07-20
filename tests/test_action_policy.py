from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_action_policy import analyze  # noqa: E402


class ActionPolicyTests(unittest.TestCase):
    def test_invalid_output_fails_closed_and_counts_as_false_refusal(self):
        generation = {
            "items": [
                {
                    "id": "a1",
                    "lane": "single_hop",
                    "expected_action": "answer",
                    "actual_action": "answer",
                    "contract_valid": False,
                },
                {
                    "id": "u1",
                    "lane": "unanswerable",
                    "expected_action": "abstain_absent",
                    "actual_action": "answer",
                    "contract_valid": True,
                },
            ]
        }
        report = analyze(
            generation,
            max_false_answer_rate=0.05,
            max_false_refusal_rate=0.15,
        )
        self.assertEqual(report["metrics"]["false_refusal_count"], 1)
        self.assertEqual(report["metrics"]["false_answer_count"], 1)
        self.assertFalse(report["guardrails"]["passes"])
        self.assertEqual(report["items"][0]["delivered_action"], "invalid_failclosed")

    def test_correct_actions_pass_observed_rate_guardrails(self):
        generation = {
            "items": [
                {
                    "id": "a1",
                    "lane": "single_hop",
                    "expected_action": "answer",
                    "actual_action": "answer",
                    "contract_valid": True,
                },
                {
                    "id": "u1",
                    "lane": "unanswerable",
                    "expected_action": "abstain_absent",
                    "actual_action": "abstain_absent",
                    "contract_valid": True,
                },
            ]
        }
        report = analyze(
            generation,
            max_false_answer_rate=0.05,
            max_false_refusal_rate=0.15,
        )
        self.assertTrue(report["guardrails"]["passes"])
        self.assertEqual(report["metrics"]["action_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
