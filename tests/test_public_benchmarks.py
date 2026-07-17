from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.benchmarks import FeverAdapter, HotpotQAAdapter, NaturalQuestionsAdapter  # noqa: E402
from rag.benchmarks.base import render_jsonl  # noqa: E402
from scripts.benchmarks import prepare  # noqa: E402


class PublicBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_path = (
            ROOT / "benchmarks" / "fixtures" / "hotpotqa-dev-sample.json"
        )
        cls.source = json.loads(cls.fixture_path.read_text(encoding="utf-8"))
        cls.adapter = HotpotQAAdapter()
        cls.schema = json.loads(
            (ROOT / "benchmarks" / "schemas" / "normalized-case.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_registry_is_valid_unique_and_targets_ten_thousand(self):
        registry = prepare.load_registry()
        schema = json.loads(
            (ROOT / "benchmarks" / "registry.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(registry)
        ids = [entry["id"] for entry in registry["benchmarks"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(entry["target_cases"] for entry in registry["benchmarks"]), 10000)

    def test_hotpotqa_fixture_normalizes_and_resolves_every_fact(self):
        normalized = self.adapter.normalize_many(
            self.source, source_split="dev_distractor"
        )
        validation = prepare.validate_normalized(normalized)
        self.assertEqual(validation["normalized_cases"], 2)
        self.assertEqual(validation["evidence_references_resolved"], 4)
        self.assertFalse(
            [
                error
                for case in normalized
                for error in Draft202012Validator(self.schema).iter_errors(case)
            ]
        )

    def test_selection_and_rendering_are_order_independent(self):
        forward = self.adapter.normalize_many(
            self.source, source_split="dev_distractor", limit=1
        )
        reverse = self.adapter.normalize_many(
            list(reversed(self.source)), source_split="dev_distractor", limit=1
        )
        self.assertEqual(render_jsonl(forward), render_jsonl(reverse))

    def test_missing_supporting_sentence_fails_closed(self):
        bad = copy.deepcopy(self.source[0])
        bad["supporting_facts"][0][1] = 99
        with self.assertRaisesRegex(ValueError, "references missing sentence"):
            self.adapter.normalize(bad, "dev_distractor")

    def test_duplicate_source_id_fails_closed(self):
        duplicate = [copy.deepcopy(self.source[0]), copy.deepcopy(self.source[0])]
        with self.assertRaisesRegex(ValueError, "duplicate source record ID"):
            self.adapter.normalize_many(duplicate, "dev_distractor")

    def test_prepare_writes_auditable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "normalized.jsonl"
            report_path = Path(directory) / "report.json"
            report = prepare.prepare(
                "hotpotqa",
                self.fixture_path,
                output,
                report_path,
                limit=2,
                verify_source_checksum=False,
            )
            self.assertEqual(report["conversion"]["normalized_cases"], 2)
            self.assertEqual(report["conversion"]["excluded_cases"], 0)
            self.assertEqual(report["conversion"]["conversion_loss_rate"], 0.0)
            self.assertEqual(report["manual_audit"]["status"], "pending")
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

    def test_prepare_rejects_unpinned_source_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "raw checksum mismatch"):
                prepare.prepare(
                    "hotpotqa",
                    self.fixture_path,
                    Path(directory) / "normalized.jsonl",
                    Path(directory) / "report.json",
                    limit=2,
                )

    def test_natural_questions_preserves_five_votes_and_null_cases(self):
        source = json.loads(
            (
                ROOT
                / "benchmarks"
                / "fixtures"
                / "natural-questions-dev-sample.json"
            ).read_text(encoding="utf-8")
        )
        normalized = NaturalQuestionsAdapter().normalize_many(source, "dev")
        prepare.validate_normalized(normalized)
        by_source_id = {
            case["provenance"]["source_record_id"]: case for case in normalized
        }
        answerable = by_source_id["nq-fixture-answerable-001"]
        self.assertEqual(answerable["gold"]["answers"], ["Paris"])
        self.assertEqual(answerable["gold"]["answerability"], "answerable")
        self.assertEqual(len(answerable["annotation_votes"]), 5)
        self.assertEqual(len(answerable["supporting_evidence"]), 1)
        null_case = by_source_id["nq-fixture-null-002"]
        self.assertEqual(null_case["gold"]["answerability"], "unanswerable")
        self.assertEqual(null_case["gold"]["answers"], [])
        self.assertEqual(null_case["supporting_evidence"], [])

    def test_fever_preserves_alternative_evidence_sets_and_nei(self):
        source = json.loads(
            (
                ROOT / "benchmarks" / "fixtures" / "fever-dev-sample.json"
            ).read_text(encoding="utf-8")
        )
        normalized = FeverAdapter().normalize_many(source, "dev")
        prepare.validate_normalized(normalized)
        by_source_id = {
            case["provenance"]["source_record_id"]: case for case in normalized
        }
        supports = by_source_id["1001"]
        self.assertEqual(supports["gold"]["label"], "supports")
        self.assertEqual(len(supports["annotation_votes"]), 2)
        self.assertEqual(len(supports["supporting_evidence"]), 2)
        self.assertEqual(
            supports["metadata"]["evidence_set_semantics"], "any_complete_set"
        )
        nei = by_source_id["1002"]
        self.assertEqual(nei["gold"]["label"], "not_enough_info")
        self.assertEqual(nei["gold"]["answerability"], "unanswerable")
        self.assertEqual(nei["documents"], [])


if __name__ == "__main__":
    unittest.main()
