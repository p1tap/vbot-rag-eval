from __future__ import annotations

import unittest

from scripts.verify_portfolio_evidence import (
    verify_adversarial_rejection,
    verify_specialist_rag,
    verify_squad_confirmation,
)


class PortfolioEvidenceTests(unittest.TestCase):
    def test_specialist_rag_recomputes_from_10000_cases(self):
        result = verify_specialist_rag()
        self.assertEqual(result["case_count"], 10_000)
        self.assertGreaterEqual(result["strict_macro_joint_correct_rate"], 0.60)
        self.assertTrue(result["zero_fail_closed"])

    def test_squad_confirmation_remains_separate_from_retrieval(self):
        result = verify_squad_confirmation()
        self.assertEqual(result["case_count"], 1_000)
        self.assertFalse(result["retrieval_included"])
        self.assertGreaterEqual(result["exact_match"], 0.87)

    def test_adversarial_rejection_cannot_silently_become_a_claim(self):
        result = verify_adversarial_rejection()
        self.assertEqual(result["status"], "rejection_reproduced")
        self.assertEqual(result["false_answer_count"], 0)
        self.assertEqual(result["non_answer_containment_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
