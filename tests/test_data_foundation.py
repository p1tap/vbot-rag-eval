from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    build_evidence_catalog,
    build_ai_release_seed,
    audit_ai_release_seed,
    build_coverage_matrix,
    build_review_inventory,
    build_v1_review_packets,
    check_dataset_governance,
    freeze_v2_contract,
    holdout_ledger,
    migrate_v1_cases,
    validate_data,
)


class DataFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = migrate_v1_cases.build_cases()
        cls.case_schema = json.loads(
            (ROOT / "evals" / "schema" / "case-v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.validator = Draft202012Validator(cls.case_schema)

    def test_v1_migration_is_exact_and_current(self):
        committed = ROOT / "evals" / "v2" / "dev" / "v1-migrated.jsonl"
        self.assertEqual(committed.read_bytes(), migrate_v1_cases.render(self.cases))
        self.assertEqual(len(self.cases), 49)
        self.assertEqual(
            sum(case["answerability"] == "answerable" for case in self.cases), 38
        )
        source, _ = migrate_v1_cases.load_v1()
        self.assertEqual(
            [case["legacy"]["v1_record"] for case in self.cases], source
        )

    def test_migrated_cases_are_visible_pending_dev_only(self):
        self.assertEqual({case["split"] for case in self.cases}, {"dev"})
        self.assertEqual(
            {case["review"]["status"] for case in self.cases},
            {"migrated_pending_v2_review"},
        )
        self.assertEqual(
            {case["evidence_granularity"] for case in self.cases},
            {"legacy_heading", "none"},
        )

    def test_answerable_requires_claims_and_evidence(self):
        bad = copy.deepcopy(next(case for case in self.cases if case["answerability"] == "answerable"))
        bad["required_claims"] = []
        self.assertTrue(list(self.validator.iter_errors(bad)))

    def test_unanswerable_cannot_use_answer_action(self):
        bad = copy.deepcopy(next(case for case in self.cases if case["answerability"] == "unanswerable"))
        bad["answer_action"] = "answer"
        self.assertTrue(list(self.validator.iter_errors(bad)))

    def test_unseeded_sealed_partitions_contain_no_case_files(self):
        for split in ("real_user", "reserve"):
            self.assertEqual(list((ROOT / "evals" / "v2" / split).glob("*.jsonl")), [])

    def test_ai_adversarial_seed_is_current_and_never_human_labeled(self):
        from scripts import build_adversarial_seed

        path = (
            ROOT
            / "evals"
            / "v2"
            / "adversarial"
            / "vbot-ai-observed-failures-12.jsonl"
        )
        cases = build_adversarial_seed.build()
        self.assertEqual(
            path.read_text(encoding="utf-8"), build_adversarial_seed.render(cases)
        )
        self.assertEqual(len(cases), 12)
        self.assertEqual(
            {case["authoring"]["method"] for case in cases},
            {"model_proposed_ai_verified"},
        )
        self.assertEqual(
            {tuple(case["review"]["reviewer_kinds"]) for case in cases},
            {("ai_agent",)},
        )

    def test_ai_release_seed_is_current_and_never_human_labeled(self):
        path = (
            ROOT
            / "evals"
            / "v2"
            / "release"
            / "vbot-ai-reviewed-release-50.jsonl"
        )
        cases = build_ai_release_seed.build_cases()
        self.assertEqual(path.read_bytes(), build_ai_release_seed.render(cases))
        self.assertEqual(len(cases), 50)
        self.assertEqual(
            {case["authoring"]["method"] for case in cases},
            {"model_proposed_ai_verified"},
        )
        self.assertTrue(
            all(case["review"]["reviewer_kinds"] == ["ai_agent"] for case in cases)
        )
        self.assertEqual(
            sum(case["answerability"] == "answerable" for case in cases), 42
        )
        audit = audit_ai_release_seed.audit(path)
        committed_audit = json.loads(
            (ROOT / "reports" / "v2" / "ai-release-seed-audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(audit, committed_audit)
        self.assertEqual(
            audit["review_provenance"]["locally_human_reviewed_case_count"], 0
        )
        self.assertFalse(audit["review_provenance"]["independent_author_reviewer"])

    def test_evidence_catalog_is_current_and_has_unique_ids(self):
        records = build_evidence_catalog.build_records()
        committed = ROOT / "corpus" / "evidence-catalog-v1.jsonl"
        self.assertEqual(committed.read_bytes(), build_evidence_catalog.render(records))
        ids = [record["evidence_id"] for record in records]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(records), 99)

    def test_corpus_fingerprint_covers_authority_metadata(self):
        manifest = json.loads(
            (ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8")
        )
        changed = copy.deepcopy(manifest["documents"])
        changed[0]["authority"]["priority"] += 1
        self.assertNotEqual(
            validate_data.canonical_json_sha256(
                sorted(manifest["documents"], key=lambda item: item["document_id"])
            ),
            validate_data.canonical_json_sha256(
                sorted(changed, key=lambda item: item["document_id"])
            ),
        )

    def test_unknown_distractor_is_rejected(self):
        bad = copy.deepcopy(
            next(case for case in self.cases if case["answerability"] == "answerable")
        )
        bad["distractor_evidence"] = ["gateway-readme::not-a-real-span"]
        _, legacy, spans = validate_data.validate_corpus()
        with self.assertRaisesRegex(ValueError, "unknown distractor evidence"):
            validate_data.validate_case_integrity(bad, legacy, spans)

    def test_ai_reviewed_case_requires_machine_readable_provenance(self):
        reviewed = json.loads(
            (ROOT / "evals" / "v2" / "reviewed" / "vbot-human-reviewed-100.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        reviewed["authoring"]["method"] = "model_proposed_ai_verified"
        self.assertFalse(list(self.validator.iter_errors(reviewed)))
        _, legacy, spans = validate_data.validate_corpus()
        with self.assertRaisesRegex(ValueError, "ambiguous reviewer provenance"):
            validate_data.validate_case_integrity(reviewed, legacy, spans)
        reviewed["review"]["reviewer_kinds"] = ["ai_agent"]
        validate_data.validate_case_integrity(reviewed, legacy, spans)

    def test_full_contract_validation(self):
        summary = validate_data.validate_all()
        self.assertEqual(summary["documents"], 3)
        self.assertEqual(summary["evidence_spans"], 99)
        self.assertEqual(summary["cases"], 162)
        self.assertEqual(
            summary["splits"], {"dev": 100, "release": 50, "adversarial": 12}
        )
        self.assertEqual(summary["review_status"], {"approved": 162})

    def test_review_inventory_is_current_and_not_freeze_ready(self):
        inventory = build_review_inventory.build_inventory()
        committed = ROOT / "evals" / "v2" / "review-inventory.json"
        self.assertEqual(
            committed.read_bytes(), build_review_inventory.render(inventory)
        )
        self.assertFalse(inventory["freeze_readiness"]["ready"])
        self.assertEqual(inventory["queues"]["pending_review"], [])
        self.assertEqual(
            inventory["freeze_readiness"]["blockers"],
            ["real_user_partition_has_cases"],
        )

    def test_coverage_matrix_is_current_and_truthful(self):
        matrix = build_coverage_matrix.build_matrix()
        committed = ROOT / "evals" / "v2" / "coverage-matrix.json"
        self.assertEqual(
            committed.read_bytes(), build_coverage_matrix.render(matrix)
        )
        self.assertEqual(matrix["summary"]["headings"], 19)
        self.assertEqual(matrix["summary"]["headings_with_current_cases"], 19)
        self.assertEqual(matrix["summary"]["approved_cases"], 162)
        self.assertEqual(matrix["summary"]["lanes_with_approved_seed"], 15)
        self.assertEqual(matrix["summary"]["provisional_total_seed_gap"], 0)

    def test_v1_review_source_map_is_current_and_requires_humans(self):
        source_map = build_v1_review_packets.build_source_map()
        committed = ROOT / "evals" / "v2" / "v1-review-source-map.json"
        self.assertEqual(
            committed.read_bytes(),
            build_v1_review_packets.render_source_map(source_map),
        )
        self.assertEqual(source_map["case_count"], 49)
        self.assertEqual(source_map["answerable_cases"], 38)
        self.assertTrue(
            all(case["human_decision_required"] for case in source_map["cases"])
        )

    def test_private_inbox_never_overwrites_human_decisions(self):
        source_map = build_v1_review_packets.build_source_map()
        with tempfile.TemporaryDirectory() as directory:
            inbox = Path(directory)
            build_v1_review_packets.write_inbox(source_map, inbox)
            decision_path = inbox / "decisions" / "batch-01-decisions.jsonl"
            marker = decision_path.read_text(encoding="utf-8") + "human-marker\n"
            decision_path.write_text(marker, encoding="utf-8")
            build_v1_review_packets.write_inbox(source_map, inbox)
            self.assertEqual(decision_path.read_text(encoding="utf-8"), marker)

    def test_pending_human_decision_cannot_be_approved_incompletely(self):
        source_map = build_v1_review_packets.build_source_map()
        decision = build_v1_review_packets.decision_template(source_map["cases"][0])
        decision["decision"] = "approve"
        schema = json.loads(
            (ROOT / "evals" / "schema" / "human-review-decision.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(list(Draft202012Validator(schema).iter_errors(decision)))

    def test_holdout_ledger_chain_detects_tampering(self):
        events = holdout_ledger.validate_events()
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["scope"], "none")
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            tampered = copy.deepcopy(events[0])
            tampered["purpose"] = "tampered"
            ledger.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "event hash mismatch"):
                holdout_ledger.validate_events(ledger)

    def test_frozen_manifest_rejects_same_version_change(self):
        base = json.loads(
            (ROOT / "evals" / "v2" / "dataset-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        base["status"] = "frozen"
        base["dataset_version"] = "2.0.0"
        changed = copy.deepcopy(base)
        changed["limitations"].append("same-version mutation")
        with self.assertRaisesRegex(ValueError, "without an explicit dataset-version"):
            check_dataset_governance.compare_against_base(base, changed)

    def test_development_manifest_can_freeze_a_nonempty_release_seed(self):
        manifest = json.loads(
            (ROOT / "evals" / "v2" / "dataset-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        inventory = build_review_inventory.build_inventory()
        check_dataset_governance.validate_current(manifest, inventory)
        partitions = {row["split"]: row for row in manifest["partitions"]}
        self.assertTrue(partitions["release"]["frozen"])
        self.assertEqual(partitions["release"]["expected_cases"], 50)
        self.assertEqual(
            manifest["case_record_versions"],
            ["2.0.0-dev.2", "2.0.0-dev.4", "2.0.0-dev.5"],
        )

        bad = copy.deepcopy(manifest)
        real_user = next(
            row for row in bad["partitions"] if row["split"] == "real_user"
        )
        real_user["frozen"] = True
        with self.assertRaisesRegex(ValueError, "empty sealed partition"):
            check_dataset_governance.validate_current(bad, inventory)

    def test_v2_contract_freeze_is_current(self):
        committed = ROOT / "evals" / "v2" / "v2-contract-freeze.json"
        self.assertEqual(
            committed.read_bytes(),
            freeze_v2_contract.render(freeze_v2_contract.build()),
        )


if __name__ == "__main__":
    unittest.main()
