from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    PROFILES,
    SENTINEL,
    answer_scores,
    contract_batch,
    prompt,
    parse_provider_output,
    one_edit_apart,
    response_format,
    run_batch,
    validate_output,
    work_item,
)
import scripts.benchmarks.audit_suite as audit_module  # noqa: E402


class PublicEndToEndTests(unittest.TestCase):
    def test_deepseek_generator_is_high_effort_and_provider_pinned(self):
        profile = PROFILES["deepseek-v4-flash-high"]
        self.assertEqual(profile["model"], "deepseek-v4-flash")
        self.assertEqual(
            profile["request_options"],
            {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        )
        self.assertEqual(profile["endpoint"], "https://api.deepseek.com")
        self.assertEqual(profile["deployment_kind"], "direct_deepseek_api")
        self.assertTrue(profile["strict_schema"])
        self.assertEqual(profile["max_tokens"], 2000)

    def test_gpt_oss_generator_is_local_high_reasoning_and_pinned(self):
        profile = PROFILES["gpt-oss-20b-local-high"]
        self.assertEqual(profile["model"], "gpt-oss-rag-16k:latest")
        self.assertEqual(profile["request_options"], {"think": "high"})
        self.assertEqual(profile["endpoint"], "http://localhost:11434")
        self.assertEqual(profile["deployment_kind"], "local_ollama")
        self.assertEqual(profile["ollama_model_id"], "6f9b0942e9f4")
        self.assertEqual(
            profile["weights_blob_sha256"],
            "e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb",
        )
        self.assertTrue(profile["strict_schema"])

    def test_deepseek_openrouter_baidu_generator_is_provider_pinned(self):
        profile = PROFILES["deepseek-v4-flash-high-openrouter-baidu"]
        self.assertEqual(profile["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(
            profile["request_options"]["reasoning"],
            {"effort": "high", "exclude": True},
        )
        self.assertEqual(
            profile["request_options"]["provider"],
            {
                "only": ["baidu/fp8"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        )
        self.assertEqual(profile["endpoint"], "https://openrouter.ai/api/v1")
        self.assertEqual(
            profile["deployment_kind"], "openrouter_provider_pinned_baidu_fp8"
        )
        self.assertTrue(profile["strict_schema"])

    def test_invalid_batch_slot_is_retried_without_semantic_repair(self):
        class FakeCall:
            def __init__(self, content):
                self.content = content

            def to_record(self):
                return {"content": self.content, "latency_ms": 1.0, "usage": {}}

        malformed = (
            '{"case_id":"C1","prediction":"A","citation_ids":["D1"]},'
            '{"case_id":"C2","prediction":"long uncertain answer"}'
        )
        recovered = (
            '{"case_id":"C2","prediction":"__UNANSWERABLE__",'
            '"citation_ids":[]}'
        )
        batch = [
            {
                "case_id": "q1",
                "benchmark_id": "hotpotqa",
                "query": "first?",
                "contexts": [{"citation_id": "D1", "title": "A", "text": "A"}],
            },
            {
                "case_id": "q2",
                "benchmark_id": "hotpotqa",
                "query": "second?",
                "contexts": [{"citation_id": "D1", "title": "B", "text": "B"}],
            },
        ]
        with patch(
            "scripts.benchmarks.run_end_to_end.chat_with_metadata",
            side_effect=[
                FakeCall(malformed),
                FakeCall(malformed),
                FakeCall(malformed),
                FakeCall(recovered),
            ],
        ):
            record = run_batch(batch, "qwen3.5-9b-local", "hotpotqa:test")
        self.assertTrue(record["valid"])
        self.assertEqual(record["execution_events"], ["retried_invalid_case_slots"])
        self.assertFalse(record["raw_contract_valid"])
        self.assertEqual(record["normalization_events"], [])
        self.assertEqual(len(record["case_slot_retry_records"]), 1)
        self.assertEqual(record["case_slot_retry_records"][0]["wire_case_id"], "C2")
        self.assertEqual(
            record["results"],
            [
                {"case_id": "q1", "prediction": "A", "citation_ids": ["D1"]},
                {
                    "case_id": "q2",
                    "prediction": "__UNANSWERABLE__",
                    "citation_ids": [],
                },
            ],
        )

    def test_claim_audit_replays_local_slot_batch_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_ids = ["h1", "n1", "f1", "f2"]
            mapping = [
                {"wire_case_id": f"C{index}", "case_id": case_id}
                for index, case_id in enumerate(case_ids, 1)
            ]
            batch = {
                "schema_version": "1.1.0",
                "valid": True,
                "errors": [],
                "case_ids": case_ids,
                "wire_case_id_mapping": mapping,
                "results": [
                    {"case_id": case_id, "prediction": "x", "citation_ids": []}
                    for case_id in case_ids
                ],
                "raw_contract_valid": True,
                "normalization_events": [],
            }
            batch_path = root / "batches.jsonl"
            batch_path.write_text(json.dumps(batch) + "\n", encoding="utf-8")
            case_path = root / "cases.jsonl"
            case_path.write_text(
                "\n".join(
                    json.dumps({"case_id": case_id, "fail_closed": False})
                    for case_id in case_ids
                )
                + "\n",
                encoding="utf-8",
            )
            source_hashes = {
                "hotpotqa": "a" * 64,
                "natural_questions": "b" * 64,
                "fever": "c" * 64,
            }
            report = {
                "status": "complete",
                "resume_claim_ready": True,
                "case_count": 4,
                "batch_count": 1,
                "valid_batch_count": 1,
                "raw_contract_valid_batch_count": 1,
                "deterministically_normalized_batch_count": 0,
                "case_slot_retry_batch_count": 0,
                "fail_closed_case_count": 0,
                "contract_guardrails": {
                    "max_deterministic_normalization_rate": 0.05,
                    "normalization_rate_pass": True,
                    "max_case_slot_retry_batch_rate": 0.01,
                    "case_slot_retry_rate_pass": True,
                },
                "identity": {
                    "benchmarks": list(source_hashes),
                    "sample_per_benchmark": 0,
                    "source_sha256": source_hashes,
                    "response_contract_version": audit_module.PUBLIC_E2E_RESPONSE_CONTRACT,
                    "runner_source_sha256": audit_module.file_sha256(
                        ROOT / "scripts" / "benchmarks" / "run_end_to_end.py"
                    ),
                    "profile_id": "local",
                },
                "artifacts": {
                    "case_records_path": "cases.jsonl",
                    "case_records_sha256": audit_module.file_sha256(case_path),
                    "batch_records_path": "batches.jsonl",
                    "batch_records_sha256": audit_module.file_sha256(batch_path),
                },
                "provenance": {"locally_human_reviewed_public_cases": 0},
                "metrics": {},
            }
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(audit_module, "ROOT", root):
                result = audit_module.validate_end_to_end(
                    report_path, source_hashes, set(case_ids)
                )
            self.assertEqual(result["raw_contract_valid_batch_count"], 1)

            bad_slot_batch = json.loads(json.dumps(batch))
            bad_slot_batch["raw_contract_valid"] = False
            bad_slot_batch["execution_events"] = ["retried_invalid_case_slots"]
            bad_slot_batch["case_slot_retry_records"] = [
                {
                    "valid": True,
                    "normalization_events": ["unknown_slot_repair"],
                }
            ]
            batch_path.write_text(
                json.dumps(bad_slot_batch) + "\n", encoding="utf-8"
            )
            report["raw_contract_valid_batch_count"] = 0
            report["case_slot_retry_batch_count"] = 1
            report["artifacts"]["batch_records_sha256"] = audit_module.file_sha256(
                batch_path
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(audit_module, "ROOT", root):
                with self.assertRaisesRegex(
                    ValueError, "unknown normalization events"
                ):
                    audit_module.validate_end_to_end(
                        report_path, source_hashes, set(case_ids)
                    )

            batch_path.write_text(json.dumps(batch) + "\n", encoding="utf-8")
            report["raw_contract_valid_batch_count"] = 1
            report["case_slot_retry_batch_count"] = 0
            report["artifacts"]["batch_records_sha256"] = audit_module.file_sha256(
                batch_path
            )
            report["identity"]["runner_source_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(audit_module, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "runner source hash drifted"):
                    audit_module.validate_end_to_end(
                        report_path, source_hashes, set(case_ids)
                    )

    def test_wire_contract_uses_short_ordered_case_slots(self):
        batch = [
            {"case_id": "benchmark:very-long-id-1", "query": "A"},
            {"case_id": "benchmark:very-long-id-2", "query": "B"},
        ]
        wire = contract_batch(batch)
        self.assertEqual([row["case_id"] for row in wire], ["C1", "C2"])
        self.assertEqual([row["query"] for row in wire], ["A", "B"])
        self.assertEqual(batch[0]["case_id"], "benchmark:very-long-id-1")

    def test_answer_scores_normalize_articles_and_punctuation(self):
        exact, f1 = answer_scores("The Paris.", ["Paris"])
        self.assertEqual(exact, 1.0)
        self.assertEqual(f1, 1.0)
        self.assertEqual(answer_scores(SENTINEL, []), (1.0, 1.0))

    def test_batch_output_rejects_cross_case_citations(self):
        batch = [
            {
                "case_id": "a",
                "benchmark_id": "hotpotqa",
                "contexts": [{"citation_id": "D1"}],
            },
            {
                "case_id": "b",
                "benchmark_id": "hotpotqa",
                "contexts": [{"citation_id": "D2"}],
            },
        ]
        value = {
            "results": [
                {"case_id": "a", "prediction": "x", "citation_ids": ["D2"]},
                {"case_id": "b", "prediction": "y", "citation_ids": ["D2"]},
            ]
        }
        self.assertIn("a: invalid citations", validate_output(value, batch))

    def test_strict_fever_schema_excludes_qa_abstention_token(self):
        batch = [
            {
                "case_id": "f1",
                "benchmark_id": "fever",
                "contexts": [{"citation_id": "D1"}],
            }
        ]
        schema = response_format(batch, strict=True)["json_schema"]["schema"]
        prediction = schema["properties"]["results"]["items"]["properties"][
            "prediction"
        ]
        self.assertEqual(
            prediction["enum"], ["not_enough_info", "refutes", "supports"]
        )
        self.assertNotIn(SENTINEL, prediction["enum"])
        citation = schema["properties"]["results"]["items"]["properties"][
            "citation_ids"
        ]["items"]
        self.assertEqual(citation, {"type": "string", "enum": ["D1"]})

    def test_prompt_example_is_valid_json_with_one_results_wrapper(self):
        batch = [
            {
                "case_id": "q1",
                "benchmark_id": "hotpotqa",
                "query": "Question?",
                "contexts": [
                    {"citation_id": "D1", "title": "Title", "text": "Text"}
                ],
            }
        ]
        rendered = prompt(batch)
        example = (
            '{"results":[{"case_id":"CASE_ID","prediction":"ANSWER",'
            '"citation_ids":["D1"]}]}'
        )
        self.assertIn(example, rendered)

    def test_provider_normalization_is_narrow_and_audited(self):
        batch = [
            {"case_id": "q1", "benchmark_id": "hotpotqa", "contexts": []}
        ]
        value, events, error = parse_provider_output(
            '{"case_id":"q1","prediction":"Paris","citation_ids":["D1"]}`',
            batch,
        )
        self.assertIsNone(error)
        self.assertEqual(
            events, ["stripped_trailing_backtick", "wrapped_single_result"]
        )
        self.assertEqual(value["results"][0]["prediction"], "Paris")
        self.assertEqual(value["results"][0]["citation_ids"], ["D1"])

        value, events, error = parse_provider_output(
            '{"results":[{"case_id":"q1","prediction":"__UNANSWERABLE__"}]}',
            batch,
        )
        self.assertIsNone(error)
        self.assertEqual(events, ["restored_empty_abstention_citations"])
        self.assertEqual(value["results"][0]["citation_ids"], [])

        value, events, error = parse_provider_output(
            '{"results":[{"case_id":"q1","prediction":"Paris"}]}', batch
        )
        self.assertIsNone(error)
        self.assertEqual(events, [])
        self.assertIn("q1: wrong fields", validate_output(value, batch))

        two = [
            {"case_id": "q1", "benchmark_id": "hotpotqa", "contexts": []},
            {"case_id": "q2", "benchmark_id": "hotpotqa", "contexts": []},
        ]
        value, events, error = parse_provider_output(
            '{"case_id":"q1","prediction":"A","citation_ids":["D1"]},'
            '{"case_id":"q2","prediction":"B","citation_ids":["D2"]}]',
            two,
        )
        self.assertIsNone(error)
        self.assertEqual(
            events, ["restored_missing_array_opener", "wrapped_results_array"]
        )
        self.assertEqual(
            [row["prediction"] for row in value["results"]], ["A", "B"]
        )

        value, events, error = parse_provider_output(
            '{"case_id":"q1","prediction":"A","citation_ids":["D1"]},'
            '{"case_id":"q2","prediction":"B","citation_ids":["D2"]}',
            two,
        )
        self.assertIsNone(error)
        self.assertEqual(
            events,
            ["wrapped_comma_separated_results", "wrapped_results_array"],
        )
        self.assertEqual(
            [row["case_id"] for row in value["results"]], ["q1", "q2"]
        )

        value, events, error = parse_provider_output(
            '{"case_id":"q1","prediction":"A","citation_ids":[]},'
            '{"case_id":"q2","prediction":"B","citation_ids":[]},'
            '{"case_id":"q3","prediction":"C","citation_ids":[]}',
            two,
        )
        self.assertIsNone(value)
        self.assertEqual(events, [])
        self.assertIn("invalid_json", error)

        fever = [
            {"case_id": "f1", "benchmark_id": "fever", "contexts": []}
        ]
        value, events, error = parse_provider_output(
            '{"results":[{"case_id":"f1","prediction":"__UNANSWERABLE__",'
            '"citation_ids":[]}]}',
            fever,
        )
        self.assertIsNone(error)
        self.assertEqual(events, ["canonicalized_fever_abstention_label"])
        self.assertEqual(value["results"][0]["prediction"], "not_enough_info")

        value, events, error = parse_provider_output(
            '{"results":[{"case_id":"q1","prediction":"Paris",'
            '"citation_ids":["D1"]}]}}',
            batch,
        )
        self.assertIsNone(error)
        self.assertEqual(events, ["stripped_extra_trailing_closer"])
        self.assertEqual(value["results"][0]["prediction"], "Paris")

        value, events, error = parse_provider_output(
            '{"results":[]}{"results":[]}', batch
        )
        self.assertIsNone(value)
        self.assertEqual(events, [])
        self.assertIn("invalid_json", error)

        expected = [
            {"case_id": "case-1234", "benchmark_id": "hotpotqa", "contexts": []},
            {"case_id": "case-5678", "benchmark_id": "hotpotqa", "contexts": []},
        ]
        value, events, error = parse_provider_output(
            '{"results":['
            '{"case_id":"case-123","prediction":"A","citation_ids":[]},'
            '{"case_id":"case-5678","prediction":"B","citation_ids":[]}] }',
            expected,
        )
        self.assertIsNone(error)
        self.assertEqual(events, ["restored_near_case_id_from_frozen_order"])
        self.assertEqual(value["results"][0]["case_id"], "case-1234")

        swapped, events, error = parse_provider_output(
            '{"results":['
            '{"case_id":"case-5678","prediction":"B","citation_ids":[]},'
            '{"case_id":"case-1234","prediction":"A","citation_ids":[]}] }',
            expected,
        )
        self.assertIsNone(error)
        self.assertEqual(events, [])
        self.assertIn("case-1234: ID/order mismatch", validate_output(swapped, expected))

        flat = [
            {"case_id": "C1", "benchmark_id": "hotpotqa", "contexts": []},
            {"case_id": "C2", "benchmark_id": "hotpotqa", "contexts": []},
        ]
        value, events, error = parse_provider_output(
            '{"case_id":"C1","prediction":"A","citation_ids":[],'
            '"case_id":"C2","prediction":"B","citation_ids":[]}',
            flat,
        )
        self.assertIsNone(error)
        self.assertEqual(events, ["regrouped_flattened_result_fields"])
        self.assertEqual(
            [result["prediction"] for result in value["results"]], ["A", "B"]
        )

        value, events, error = parse_provider_output(
            '{"case_id":"C1","prediction":"A","citation_ids":[],'
            '"case_id":"C2","prediction":"B"}',
            flat,
        )
        self.assertIsNone(error)
        self.assertEqual(events, [])
        self.assertTrue(
            any(
                "root must contain exactly" in item
                for item in validate_output(value, flat)
            )
        )

    def test_observed_nq_prediction_arrays_are_serialized_without_inference(self):
        batch = [
            {
                "case_id": "C1",
                "benchmark_id": "natural_questions",
                "contexts": [{"citation_id": "D1"}],
            }
        ]
        value, events, error = parse_provider_output(
            '{"results":[{"case_id":"C1","prediction":'
            '["New South Wales - Sydney","Victoria - Melbourne"],'
            '"citation_ids":["D1"]}]}',
            batch,
        )
        self.assertIsNone(error)
        self.assertEqual(events, ["serialized_nq_prediction_string_array"])
        self.assertEqual(
            value["results"][0]["prediction"],
            "New South Wales - Sydney; Victoria - Melbourne",
        )
        self.assertEqual(validate_output(value, batch), [])

        rejected_predictions = [
            ["Paris"],
            ["Paris", "paris"],
            ["Paris", ""],
            ["Paris", " London"],
            ["Paris", "London; England"],
            ["Paris", SENTINEL],
            ["Paris", 7],
            [["Paris"], "London"],
        ]
        for prediction in rejected_predictions:
            content = json.dumps(
                {
                    "results": [
                        {
                            "case_id": "C1",
                            "prediction": prediction,
                            "citation_ids": ["D1"],
                        }
                    ]
                }
            )
            rejected, rejected_events, rejected_error = parse_provider_output(
                content, batch
            )
            self.assertIsNone(rejected_error)
            self.assertEqual(rejected_events, [])
            self.assertIn("C1: empty prediction", validate_output(rejected, batch))

        hotpot = [{**batch[0], "benchmark_id": "hotpotqa"}]
        rejected, rejected_events, rejected_error = parse_provider_output(
            '{"results":[{"case_id":"C1","prediction":["A","B"],'
            '"citation_ids":["D1"]}]}',
            hotpot,
        )
        self.assertIsNone(rejected_error)
        self.assertEqual(rejected_events, [])
        self.assertIn("C1: empty prediction", validate_output(rejected, hotpot))

    def test_observed_merged_result_triplets_expand_only_exact_field_order(self):
        batch = [
            {
                "case_id": f"C{index}",
                "benchmark_id": "fever",
                "contexts": [{"citation_id": "D1"}],
            }
            for index in range(1, 5)
        ]
        merged = (
            '{"results":['
            '{"case_id":"C1","prediction":"supports","citation_ids":["D1"]},'
            '{"case_id":"C2","prediction":"refutes","citation_ids":["D1"],'
            '"case_id":"C3","prediction":"supports","citation_ids":["D1"]},'
            '{"case_id":"C4","prediction":"refutes","citation_ids":["D1"]}]}'
        )
        value, events, error = parse_provider_output(merged, batch)
        self.assertIsNone(error)
        self.assertEqual(events, ["split_merged_result_fields"])
        self.assertEqual(
            [result["case_id"] for result in value["results"]],
            ["C1", "C2", "C3", "C4"],
        )
        self.assertEqual(validate_output(value, batch), [])

        missing_root_closer = merged[:-1]
        value, events, error = parse_provider_output(missing_root_closer, batch)
        self.assertIsNone(error)
        self.assertEqual(
            events,
            ["restored_missing_root_closer", "split_merged_result_fields"],
        )
        self.assertEqual(validate_output(value, batch), [])

        wrong_count = merged.replace(
            ',{"case_id":"C4","prediction":"refutes","citation_ids":["D1"]}',
            "",
        )
        value, events, error = parse_provider_output(wrong_count, batch)
        self.assertIsNone(error)
        self.assertEqual(events, [])
        self.assertIn("result count does not match batch", validate_output(value, batch))

        wrong_order = merged.replace(
            '"case_id":"C3","prediction":"supports","citation_ids":["D1"]',
            '"prediction":"supports","case_id":"C3","citation_ids":["D1"]',
        )
        value, events, error = parse_provider_output(wrong_order, batch)
        self.assertIsNone(error)
        self.assertEqual(events, [])
        self.assertIn("result count does not match batch", validate_output(value, batch))

    def test_one_edit_apart_is_strict(self):
        self.assertTrue(one_edit_apart("abc", "abb"))
        self.assertTrue(one_edit_apart("abc", "abdc"))
        self.assertTrue(one_edit_apart("abc", "ac"))
        self.assertFalse(one_edit_apart("abc", "abc"))
        self.assertFalse(one_edit_apart("abc", "axy"))

    def test_fever_gold_document_ids_match_global_index_ids(self):
        case = {
            "id": "fever:dev:1",
            "benchmark_id": "fever",
            "query": "A claim",
            "gold": {
                "answerability": "answerable",
                "answers": [],
                "label": "supports",
            },
            "documents": [
                {"id": "local-1", "title": "New York City", "sentences": ["Text."]}
            ],
            "supporting_evidence": [
                {"document_id": "local-1", "sentence_index": 0, "text": "Text."}
            ],
            "annotation_votes": [
                {
                    "answerability": "answerable",
                    "answers": [],
                    "label": "supports",
                    "evidence": [
                        {"document_id": "local-1", "sentence_index": 0}
                    ],
                }
            ],
        }
        item = work_item(
            case,
            [{"id": "New_York_City", "title": "New York City", "sentences": ["Text."]}],
            6000,
        )
        self.assertEqual(item["gold_document_ids"], ["New_York_City"])
        self.assertEqual(item["gold_evidence_sets"], [["New_York_City"]])


if __name__ == "__main__":
    unittest.main()
