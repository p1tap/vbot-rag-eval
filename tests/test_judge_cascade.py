from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.audit_calibration_consensus import family_vote  # noqa: E402
from scripts.simulate_judge_cascade import resolve_with_adjudicator  # noqa: E402
from scripts.run_operational_judge import apply_case_metrics  # noqa: E402


def record(task_id: str, decision: str) -> dict:
    return {
        "valid": True,
        "verdicts": [{"task_id": task_id, "decision": decision}],
    }


class JudgeCascadeTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "task_id": "case:support:c1",
            "task_type": "generated_claim_support",
        }

    def test_adjudicated_accept_requires_a_second_accept_vote(self):
        decisions, failed = resolve_with_adjudicator(
            [self.task],
            record(self.task["task_id"], "supported"),
            record(self.task["task_id"], "unsupported"),
            record(self.task["task_id"], "supported"),
        )
        self.assertEqual(decisions[self.task["task_id"]], "supported")
        self.assertFalse(failed)

        decisions, failed = resolve_with_adjudicator(
            [self.task], None, None, record(self.task["task_id"], "supported")
        )
        self.assertEqual(decisions[self.task["task_id"]], "unsupported")
        self.assertEqual(failed, {self.task["task_id"]})

    def test_same_family_efforts_collapse_to_one_consistent_vote(self):
        task_id = self.task["task_id"]
        maps = {
            "high": {task_id: {"decision": "supported"}},
            "max": {task_id: {"decision": "supported"}},
        }
        vote, evidence = family_vote(task_id, maps, ("high", "max"))
        self.assertEqual(vote, "supported")
        self.assertEqual(set(evidence), {"high", "max"})
        maps["max"][task_id]["decision"] = "unsupported"
        vote, _ = family_vote(task_id, maps, ("high", "max"))
        self.assertIsNone(vote)

    def test_operational_case_requires_support_and_coverage(self):
        generation = {
            "items": [
                {
                    "id": "case",
                    "lane": "single_hop",
                    "severity": "high",
                    "expected_action": "answer",
                    "actual_action": "answer",
                    "contract_valid": True,
                    "action_correct": True,
                    "answer": {"claims": [{"id": "c1"}]},
                }
            ]
        }
        predictions = {
            "case:support:c1": {"decision": "supported"},
            "case:coverage:c1": {"decision": "missing"},
        }
        items, summary = apply_case_metrics(generation, predictions)
        self.assertFalse(items[0]["case_pass"])
        self.assertEqual(summary["end_to_end_case_pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
