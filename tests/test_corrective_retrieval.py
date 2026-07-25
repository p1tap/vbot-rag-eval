from __future__ import annotations

import unittest

from scripts.benchmarks.run_corrective_fault_matrix import run_matrix


class CorrectiveRetrievalTests(unittest.TestCase):
    def test_injected_fault_matrix_passes_and_abstentions_return_no_hits(self) -> None:
        rows = run_matrix()
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["passed"] for row in rows))
        abstentions = [
            row for row in rows if row["observed_status"] == "abstain"
        ]
        self.assertTrue(abstentions)
        self.assertTrue(
            all(not row["returned_document_ids"] for row in abstentions)
        )

    def test_fault_trace_never_contains_injected_exception_messages(self) -> None:
        serialized = repr(run_matrix())
        self.assertNotIn("injected", serialized)


if __name__ == "__main__":
    unittest.main()
