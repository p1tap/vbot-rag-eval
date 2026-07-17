from __future__ import annotations

import copy
import unittest

from scripts.compare_structured_runs import compare_reports


def report(top_k: int, *, action: float = 1.0, case_pass: bool = True) -> dict:
    summary = {
        "contract_valid_rate": 1.0,
        "action_accuracy": action,
        "citation_existence_precision": 1.0,
        "claim_valid_citation_coverage": 1.0,
        "judge_output_valid_rate": 1.0,
        "end_to_end_case_pass_rate": float(case_pass),
        "generated_claim_support_rate": 1.0,
        "required_claim_coverage_rate": 1.0,
        "retrieval_any_gold_evidence_rate": 1.0,
        "retrieval_complete_gold_evidence_rate": 1.0,
        "retrieval_required_claim_coverage": 1.0,
        "by_lane": {
            "unanswerable": {
                "case_count": 1,
                "case_pass": int(case_pass),
                "action_correct": int(action == 1.0),
            }
        },
    }
    return {
        "release_eligible": False,
        "dataset": {"sha256": "a" * 64, "case_count": 1},
        "models": {"generator": "g", "judge": "j"},
        "prompt_hashes": {"answer": "b" * 64},
        "retrieval": {"top_k": top_k, "unique_headings": True},
        "provenance": {"inputs_sha256": {"x": "c" * 64}, "index_sha256": {"i": "d" * 64}},
        "summary": summary,
        "items": [
            {
                "id": "u1",
                "case_pass": case_pass,
                "generation": {"calls": [{"response_model": "provider/g"}]},
                "claim_verifier": {"calls": [{"response_model": "provider/s"}]},
                "claim_judge": {"calls": [{"response_model": "provider/j"}]},
            }
        ],
    }


class StructuredGateTests(unittest.TestCase):
    def test_unreviewed_nonregression_can_only_reach_development_review(self) -> None:
        result = compare_reports(report(4), report(6))
        self.assertEqual(result["decision"], "DEVELOPMENT_REVIEW")

    def test_safety_action_regression_is_rejected(self) -> None:
        result = compare_reports(report(4), report(6, action=0.0, case_pass=False))
        self.assertEqual(result["decision"], "REJECT")
        self.assertTrue(any("unanswerable" in reason for reason in result["rejection_reasons"]))

    def test_provider_model_change_refuses_comparison(self) -> None:
        candidate = copy.deepcopy(report(6))
        candidate["items"][0]["generation"]["calls"][0]["response_model"] = "other/g"
        with self.assertRaisesRegex(ValueError, "provider-returned"):
            compare_reports(report(4), candidate)

    def test_support_judge_model_change_refuses_comparison(self) -> None:
        candidate = copy.deepcopy(report(6))
        candidate["items"][0]["claim_verifier"]["calls"][0][
            "response_model"
        ] = "other/support"
        with self.assertRaisesRegex(ValueError, "provider-returned"):
            compare_reports(report(4), candidate)


if __name__ == "__main__":
    unittest.main()
