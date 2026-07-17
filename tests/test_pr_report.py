from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.render_pr_report import render  # noqa: E402


class PrReportTests(unittest.TestCase):
    def test_report_keeps_public_and_local_human_provenance_separate(self):
        report = render(
            "PROMOTE",
            {"summary": {"recall_at_k": 0.9, "correctness": 0.8}},
            {
                "dataset_version": "2.0.0",
                "partitions": [{"expected_cases": 100}],
            },
            {
                "status": "end_to_end_evaluated",
                "public_case_count": 10000,
                "resume_claim_blockers": [],
                "end_to_end_evaluation": {
                    "metrics": {"macro": {"joint_correct_rate": 0.5}}
                },
            },
            {"metrics": {"overall_agreement": 0.76}},
        )
        self.assertIn("**Decision:** PROMOTE", report)
        self.assertIn("publisher human annotations", report)
        self.assertIn("100 locally owner-reviewed", report)
        self.assertNotIn("10,000 locally", report)


if __name__ == "__main__":
    unittest.main()

