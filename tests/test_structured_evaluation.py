from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.claim_judge import judge_claims, validate_claim_judgment  # noqa: E402
import config  # noqa: E402
from rag.claim_verifier import (  # noqa: E402
    support_payload,
    validate_verification,
    verify_claims,
)
from rag.coverage_judge import (  # noqa: E402
    coverage_payload,
    judge_coverage,
    validate_coverage_judgment,
)
from rag.generate import structured_answer  # noqa: E402
from rag.llm import ChatResult, chat_with_metadata  # noqa: E402
from rag.judge_agreement import answer_sha256, cohen_kappa, compare_judgments  # noqa: E402
from rag.structured_answer import (  # noqa: E402
    context_sources,
    parse_structured_answer,
    render_response,
    structured_answer_response_format,
    validate_structured_answer,
)
from scripts.run_structured_eval import retrieval_diagnostics, verifier_support_pass  # noqa: E402
from scripts.run_judge_bakeoff import select_profiles as select_bakeoff_profiles  # noqa: E402
from scripts.run_operational_judge import direct_profile_predictions  # noqa: E402
from scripts.preflight_judge_models import (  # noqa: E402
    validate_profiles,
    validate_smoke_content,
)
from scripts.build_judge_calibration_queue import (  # noqa: E402
    build_queue,
    load_cases,
)


VALID_ANSWER = {
    "schema_version": "1.0.0",
    "action": "answer",
    "response": None,
    "claims": [
        {"id": "c1", "text": "Retry the same deployment first.", "citation_ids": ["S1"]},
        {"id": "c2", "text": "The circuit breaker follows.", "citation_ids": ["S2"]},
    ],
}


class StructuredAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads(
            (ROOT / "evals" / "schema" / "structured-answer.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.validator = Draft202012Validator(schema)

    def test_valid_answer_passes_schema_and_semantic_validation(self):
        self.assertEqual(list(self.validator.iter_errors(VALID_ANSWER)), [])
        result = validate_structured_answer(VALID_ANSWER, ["S1", "S2"])
        self.assertTrue(result.valid)
        self.assertEqual(result.metrics["citation_existence_precision"], 1.0)
        self.assertEqual(result.metrics["claim_valid_citation_coverage"], 1.0)

    def test_provider_schema_pins_actions_and_supplied_source_ids(self):
        response = structured_answer_response_format(
            ["S2", "S1", "S2"], ["answer", "abstain_absent"]
        )
        schema = response["json_schema"]["schema"]
        self.assertEqual(response["type"], "json_schema")
        self.assertTrue(response["json_schema"]["strict"])
        self.assertEqual(
            schema["properties"]["action"]["enum"],
            ["abstain_absent", "answer"],
        )
        citation = schema["properties"]["claims"]["items"]["properties"][
            "citation_ids"
        ]
        self.assertEqual(citation["items"]["enum"], ["S1", "S2"])
        self.assertEqual(citation["minItems"], 1)

    def test_nonexistent_citation_cannot_pass(self):
        answer = copy.deepcopy(VALID_ANSWER)
        answer["claims"][1]["citation_ids"] = ["S99"]
        result = validate_structured_answer(answer, ["S1", "S2"])
        self.assertFalse(result.valid)
        self.assertIn("nonexistent_citation", {item["code"] for item in result.errors})
        self.assertEqual(result.metrics["citation_existence_precision"], 0.5)

    def test_uncited_claim_cannot_pass(self):
        answer = copy.deepcopy(VALID_ANSWER)
        answer["claims"][0]["citation_ids"] = []
        self.assertTrue(list(self.validator.iter_errors(answer)))
        result = validate_structured_answer(answer, ["S1", "S2"])
        self.assertFalse(result.valid)
        self.assertIn("uncited_claim", {item["code"] for item in result.errors})

    def test_duplicate_and_out_of_order_claim_ids_cannot_pass(self):
        answer = copy.deepcopy(VALID_ANSWER)
        answer["claims"][1]["id"] = "c1"
        result = validate_structured_answer(answer, ["S1", "S2"])
        codes = {item["code"] for item in result.errors}
        self.assertFalse(result.valid)
        self.assertIn("duplicate_claim_id", codes)
        self.assertIn("claim_order", codes)

    def test_abstention_has_no_claims(self):
        answer = {
            "schema_version": "1.0.0",
            "action": "abstain_absent",
            "response": "The supplied sources do not contain that information.",
            "claims": [],
        }
        self.assertEqual(list(self.validator.iter_errors(answer)), [])
        self.assertTrue(validate_structured_answer(answer, ["S1"]).valid)
        answer["claims"] = copy.deepcopy(VALID_ANSWER["claims"][:1])
        self.assertFalse(validate_structured_answer(answer, ["S1"]).valid)

    def test_parser_rejects_markdown_fences_and_prose(self):
        raw = "```json\n" + json.dumps(VALID_ANSWER) + "\n```"
        with self.assertRaisesRegex(ValueError, "not exact JSON"):
            parse_structured_answer(raw)

    def test_answer_response_cannot_bypass_atomic_claim_checks(self):
        answer = copy.deepcopy(VALID_ANSWER)
        answer["response"] = "An uncaptured factual statement."
        result = validate_structured_answer(answer, ["S1", "S2"])
        self.assertFalse(result.valid)
        self.assertIn("answer_response", {item["code"] for item in result.errors})

    def test_user_facing_answer_is_derived_from_claims(self):
        self.assertEqual(
            render_response(VALID_ANSWER),
            "Retry the same deployment first. The circuit breaker follows.",
        )

    def test_context_sources_have_prompt_local_and_stable_ids(self):
        sources = context_sources(
            [
                {"id": "doc::heading::0", "heading_key": "doc::heading", "heading": "H", "text": "T"},
                {"id": "doc::other::0", "heading_key": "doc::other", "heading": "O", "text": "U"},
            ]
        )
        self.assertEqual([item["citation_id"] for item in sources], ["S1", "S2"])
        self.assertEqual(sources[0]["chunk_id"], "doc::heading::0")


class ClaimJudgeValidationTests(unittest.TestCase):
    def test_complete_supported_judgment_passes(self):
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {"claim_id": "c1", "status": "supported", "rationale": "S1 states it."},
                {"claim_id": "c2", "status": "supported", "rationale": "S2 states it."},
            ],
            "required_claim_verdicts": [
                {
                    "claim_id": "c1",
                    "status": "covered",
                    "generated_claim_ids": ["c1"],
                    "rationale": "Conveyed by c1.",
                },
                {
                    "claim_id": "c2",
                    "status": "covered",
                    "generated_claim_ids": ["c2"],
                    "rationale": "Conveyed by c2.",
                },
            ],
        }
        errors, metrics = validate_claim_judgment(judgment, ["c1", "c2"], ["c1", "c2"])
        self.assertEqual(errors, ())
        self.assertTrue(metrics["claim_level_pass"])

    def test_missing_or_invented_verdict_ids_fail_closed(self):
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {"claim_id": "c9", "status": "supported", "rationale": "Invented ID."}
            ],
            "required_claim_verdicts": [],
        }
        errors, metrics = validate_claim_judgment(judgment, ["c1"], ["c1"])
        codes = {item["code"] for item in errors}
        self.assertIn("generated_id_order", codes)
        self.assertIn("required_id_order", codes)
        self.assertFalse(metrics["claim_level_pass"])

    def test_missing_required_claim_cannot_link_generated_claims(self):
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {"claim_id": "c1", "status": "supported", "rationale": "Supported."}
            ],
            "required_claim_verdicts": [
                {
                    "claim_id": "c1",
                    "status": "missing",
                    "generated_claim_ids": ["c1"],
                    "rationale": "Not conveyed.",
                }
            ],
        }
        errors, _ = validate_claim_judgment(judgment, ["c1"], ["c1"])
        self.assertIn("missing_with_links", {item["code"] for item in errors})

    def test_covered_required_claim_must_link_generated_claims(self):
        judgment = {
            "schema_version": "2.0.0",
            "claim_verdicts": [
                {"claim_id": "c1", "status": "supported", "rationale": "Supported."}
            ],
            "required_claim_verdicts": [
                {
                    "claim_id": "c1",
                    "status": "covered",
                    "generated_claim_ids": [],
                    "rationale": "Covered somehow.",
                }
            ],
        }
        errors, _ = validate_claim_judgment(judgment, ["c1"], ["c1"])
        self.assertIn("verdict_without_links", {item["code"] for item in errors})


class SplitJudgeContractTests(unittest.TestCase):
    def setUp(self):
        self.answer = {
            "schema_version": "1.0.0",
            "action": "answer",
            "response": None,
            "claims": [
                {
                    "id": "c1",
                    "text": "The service listens on port 30301.",
                    "citation_ids": ["S1"],
                }
            ],
        }
        self.sources = [
            {
                "citation_id": "S1",
                "chunk_id": "doc::h::0",
                "text": "The service listens on port 30301 by default.",
            }
        ]

    def test_support_uses_namespaced_ids_and_exact_source_quotes(self):
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {
                    "generated_claim_id": "g1",
                    "status": "supported",
                    "evidence": [
                        {
                            "citation_id": "S1",
                            "quote": "listens on port 30301",
                        }
                    ],
                    "rationale": "The cited source directly states the port.",
                }
            ],
        }
        errors, metrics = validate_verification(judgment, self.answer, self.sources)
        self.assertEqual(errors, ())
        self.assertTrue(metrics["runtime_claim_support_pass"])
        payload = json.loads(support_payload("Which port?", self.answer, self.sources))
        self.assertEqual(payload["generated_claims"][0]["id"], "g1")
        self.assertNotIn("required_claims", payload)

    def test_support_rejects_non_verbatim_evidence(self):
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {
                    "generated_claim_id": "g1",
                    "status": "supported",
                    "evidence": [{"citation_id": "S1", "quote": "port is 30301"}],
                    "rationale": "Paraphrase, not a quote.",
                }
            ],
        }
        errors, _ = validate_verification(judgment, self.answer, self.sources)
        self.assertIn("quote_not_found", {item["code"] for item in errors})

    def test_support_accepts_unsupported_with_explicit_empty_evidence(self):
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {
                    "generated_claim_id": "g1",
                    "status": "unsupported",
                    "evidence": [],
                    "rationale": "The cited source does not establish the claim.",
                }
            ],
        }
        errors, metrics = validate_verification(judgment, self.answer, self.sources)
        self.assertEqual(errors, ())
        self.assertFalse(metrics["runtime_claim_support_pass"])

    def test_support_accepts_contradiction_with_exact_evidence(self):
        sources = copy.deepcopy(self.sources)
        sources[0]["text"] = "The service listens on port 3001, not port 30301."
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {
                    "generated_claim_id": "g1",
                    "status": "contradicted",
                    "evidence": [
                        {
                            "citation_id": "S1",
                            "quote": "listens on port 3001, not port 30301",
                        }
                    ],
                    "rationale": "The source gives a different port.",
                }
            ],
        }
        errors, metrics = validate_verification(judgment, self.answer, sources)
        self.assertEqual(errors, ())
        self.assertFalse(metrics["runtime_claim_support_pass"])

    def test_support_rejects_exact_value_absent_from_source(self):
        wrong_source = copy.deepcopy(self.sources)
        wrong_source[0]["text"] = "The service listens on port 3001 by default."
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {
                    "generated_claim_id": "g1",
                    "status": "supported",
                    "evidence": [
                        {"citation_id": "S1", "quote": "listens on port 3001"}
                    ],
                    "rationale": "Same topic, wrong exact value.",
                }
            ],
        }
        errors, _ = validate_verification(judgment, self.answer, wrong_source)
        self.assertIn("exact_anchor_missing", {item["code"] for item in errors})

    def test_support_rejects_model_size_as_pair_count(self):
        answer = copy.deepcopy(self.answer)
        answer["claims"][0]["text"] = "The run used 1.5B DPO pairs."
        sources = copy.deepcopy(self.sources)
        sources[0]["text"] = "A 1.5B model was used for DPO."
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {
                    "generated_claim_id": "g1",
                    "status": "supported",
                    "evidence": [
                        {"citation_id": "S1", "quote": "A 1.5B model was used for DPO."}
                    ],
                    "rationale": "The same number appears.",
                }
            ],
        }
        errors, _ = validate_verification(judgment, answer, sources)
        self.assertIn("exact_anchor_missing", {item["code"] for item in errors})

    def test_coverage_payload_has_no_sources_and_uses_g_r_ids(self):
        required = [{"id": "c1", "text": "The service listens on port 30301."}]
        payload = json.loads(coverage_payload("Which port?", required, self.answer))
        self.assertEqual(payload["generated_claims"][0]["id"], "g1")
        self.assertEqual(payload["required_claims"][0]["id"], "r1")
        self.assertNotIn("cited_sources", payload)
        self.assertNotIn("sources", payload)

    def test_coverage_rejects_specific_fact_from_generic_near_match(self):
        required = [{"id": "c1", "text": "Anonymous writes return HTTP 403."}]
        answer = copy.deepcopy(self.answer)
        answer["claims"][0]["text"] = "Public Grafana is available."
        judgment = {
            "schema_version": "1.0.0",
            "required_claim_verdicts": [
                {
                    "required_claim_id": "r1",
                    "status": "covered",
                    "generated_claim_ids": ["g1"],
                    "rationale": "Both refer to public Grafana.",
                }
            ],
        }
        errors, _ = validate_coverage_judgment(judgment, required, answer)
        self.assertIn(
            "coverage_exact_anchor_missing", {item["code"] for item in errors}
        )

    def test_coverage_rejects_covered_verdict_without_links(self):
        required = [{"id": "c1", "text": "Retry the empty response exactly once."}]
        judgment = {
            "schema_version": "1.0.0",
            "required_claim_verdicts": [
                {
                    "required_claim_id": "r1",
                    "status": "covered",
                    "generated_claim_ids": [],
                    "rationale": "No linked evidence.",
                }
            ],
        }
        errors, _ = validate_coverage_judgment(judgment, required, self.answer)
        self.assertIn("verdict_without_links", {item["code"] for item in errors})

    def test_coverage_accepts_missing_without_links(self):
        required = [{"id": "c1", "text": "Retry the empty response exactly once."}]
        judgment = {
            "schema_version": "1.0.0",
            "required_claim_verdicts": [
                {
                    "required_claim_id": "r1",
                    "status": "missing",
                    "generated_claim_ids": [],
                    "rationale": "No generated claim states the retry rule.",
                }
            ],
        }
        errors, metrics = validate_coverage_judgment(judgment, required, self.answer)
        self.assertEqual(errors, ())
        self.assertFalse(metrics["required_claim_coverage_pass"])

    def test_coverage_accepts_contradiction_with_link(self):
        required = [{"id": "c1", "text": "The service listens on port 3001."}]
        judgment = {
            "schema_version": "1.0.0",
            "required_claim_verdicts": [
                {
                    "required_claim_id": "r1",
                    "status": "contradicted",
                    "generated_claim_ids": ["g1"],
                    "rationale": "The generated claim gives port 30301 instead.",
                }
            ],
        }
        errors, metrics = validate_coverage_judgment(judgment, required, self.answer)
        self.assertEqual(errors, ())
        self.assertFalse(metrics["required_claim_coverage_pass"])

    def test_coverage_rejects_generic_deployment_for_specific_promotion(self):
        required = [
            {"id": "c1", "text": "PR #42 promotes the RP model after merge."}
        ]
        answer = copy.deepcopy(self.answer)
        answer["claims"][0]["text"] = "The model passes a gate and Argo deploys it."
        judgment = {
            "schema_version": "1.0.0",
            "required_claim_verdicts": [
                {
                    "required_claim_id": "r1",
                    "status": "covered",
                    "generated_claim_ids": ["g1"],
                    "rationale": "Both describe deployment.",
                }
            ],
        }
        errors, _ = validate_coverage_judgment(judgment, required, answer)
        self.assertIn(
            "coverage_exact_anchor_missing", {item["code"] for item in errors}
        )

    def test_coverage_rejects_post_retry_rate_for_retry_rule(self):
        required = [
            {"id": "c1", "text": "The client retries an empty response exactly once."}
        ]
        answer = copy.deepcopy(self.answer)
        answer["claims"][0]["text"] = "The post-retry failure rate is 2%."
        judgment = {
            "schema_version": "1.0.0",
            "required_claim_verdicts": [
                {
                    "required_claim_id": "r1",
                    "status": "covered",
                    "generated_claim_ids": ["g1"],
                    "rationale": "Both discuss retries.",
                }
            ],
        }
        errors, _ = validate_coverage_judgment(judgment, required, answer)
        self.assertIn(
            "coverage_exact_anchor_missing", {item["code"] for item in errors}
        )

    def test_coverage_rejects_lost_explicit_negation(self):
        required = [{"id": "c1", "text": "Anonymous writes do not succeed."}]
        answer = copy.deepcopy(self.answer)
        answer["claims"][0]["text"] = "Anonymous writes succeed."
        judgment = {
            "schema_version": "1.0.0",
            "required_claim_verdicts": [
                {
                    "required_claim_id": "r1",
                    "status": "covered",
                    "generated_claim_ids": ["g1"],
                    "rationale": "Same subject.",
                }
            ],
        }
        errors, _ = validate_coverage_judgment(judgment, required, answer)
        self.assertIn(
            "coverage_explicit_negation_missing", {item["code"] for item in errors}
        )


class LlmAuditTests(unittest.TestCase):
    @staticmethod
    def result(content, response_id):
        return ChatResult(
            content=content,
            requested_model="model",
            response_model="provider/model",
            response_id=response_id,
            input_sha256="a" * 64,
            latency_ms=1.0,
            attempts=1,
            retries=0,
            usage={},
            cost=None,
        )

    @patch("rag.llm.API_KEY", "test-key")
    @patch("rag.llm.requests.post")
    def test_chat_retains_provider_identity_usage_and_retry_count(self, post):
        empty = Mock(status_code=200)
        empty.json.return_value = {"choices": [{"message": {"content": ""}}]}
        success = Mock(status_code=200)
        success.json.return_value = {
            "id": "response-123",
            "model": "provider/model-revision",
            "provider": "provider-slug",
            "system_fingerprint": "fingerprint-1",
            "choices": [
                {
                    "message": {"content": '{"ok":true}'},
                    "finish_reason": "stop",
                    "native_finish_reason": "completed",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.001},
        }
        post.side_effect = [empty, success]
        with patch("rag.llm.time.sleep"):
            result = chat_with_metadata(
                "requested/model",
                [{"role": "user", "content": "hello"}],
                response_format={"type": "json_object"},
                request_options={
                    "reasoning": {"effort": "high", "exclude": True},
                    "provider": {"require_parameters": True},
                },
            )
        self.assertEqual(result.content, '{"ok":true}')
        self.assertEqual(result.requested_model, "requested/model")
        self.assertEqual(result.response_model, "provider/model-revision")
        self.assertEqual(result.response_id, "response-123")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.retries, 1)
        self.assertEqual(result.usage["completion_tokens"], 4)
        self.assertEqual(result.cost, 0.001)
        self.assertEqual(result.response_provider, "provider-slug")
        self.assertEqual(result.system_fingerprint, "fingerprint-1")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.native_finish_reason, "completed")
        self.assertEqual(result.request_options["reasoning"]["effort"], "high")
        self.assertEqual(
            post.call_args.kwargs["json"]["provider"], {"require_parameters": True}
        )
        self.assertRegex(result.input_sha256, "^[a-f0-9]{64}$")
        self.assertRegex(result.endpoint_sha256, "^[a-f0-9]{64}$")
        self.assertEqual(result.request_timeout_seconds, 90.0)
        self.assertEqual(post.call_args.kwargs["timeout"], 90.0)

    @patch("rag.llm.API_KEY", "test-key")
    @patch("rag.llm.requests.post")
    def test_chat_records_explicit_local_timeout(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "id": "response-123",
            "model": "local/model",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        post.return_value = response
        result = chat_with_metadata(
            "local/model",
            [{"role": "user", "content": "hello"}],
            timeout_seconds=180,
        )
        self.assertEqual(result.request_timeout_seconds, 180.0)
        self.assertEqual(post.call_args.kwargs["timeout"], 180.0)

    @patch("rag.llm.BASE_URL", "http://localhost:11434")
    @patch("rag.llm.API_KEY", "")
    @patch("rag.llm.requests.post")
    def test_native_ollama_uses_decoder_schema_and_needs_no_api_key(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "qwen-local",
            "message": {"content": '{"value":"ok"}'},
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 5,
        }
        post.return_value = response
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
        result = chat_with_metadata(
            "qwen-local",
            [{"role": "user", "content": "hello"}],
            max_tokens=100,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "test", "strict": True, "schema": schema},
            },
            request_options={"think": False},
        )
        request = post.call_args.kwargs
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/chat")
        self.assertNotIn("Authorization", request["headers"])
        self.assertEqual(request["json"]["format"], schema)
        self.assertEqual(request["json"]["options"]["num_predict"], 100)
        self.assertFalse(request["json"]["think"])
        self.assertEqual(result.endpoint_kind, "local_ollama_native")
        self.assertEqual(result.response_provider, "ollama-local")
        self.assertEqual(result.usage["total_tokens"], 17)

    @patch("rag.llm.BASE_URL", "http://localhost:11434")
    @patch("rag.llm.API_KEY", "")
    @patch("rag.llm.requests.post")
    def test_native_ollama_accepts_gpt_oss_reasoning_levels(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "gpt-oss:20b",
            "message": {"content": '{"value":"ok"}'},
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 8,
        }
        post.return_value = response
        chat_with_metadata(
            "gpt-oss:20b",
            [{"role": "user", "content": "hello"}],
            request_options={"think": "high"},
        )
        self.assertEqual(post.call_args.kwargs["json"]["think"], "high")

    @patch("rag.llm.BASE_URL", "https://api.deepseek.com")
    @patch("rag.llm.API_KEY", "test-key")
    @patch("rag.llm.requests.post")
    def test_custom_provider_does_not_receive_openrouter_metadata_header(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "id": "chat-1",
            "model": "deepseek-v4-flash",
            "choices": [
                {"message": {"content": '{"value":"ok"}'}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }
        post.return_value = response
        chat_with_metadata(
            "deepseek-v4-flash",
            [{"role": "user", "content": "hello"}],
            request_options={
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            },
        )
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertNotIn("X-OpenRouter-Metadata", headers)

    @patch("rag.llm.API_KEY", "test-key")
    @patch("rag.llm.requests.post")
    def test_reasoning_and_provider_options_change_input_hash(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "id": "response-123",
            "model": "provider/model-revision",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        post.return_value = response
        messages = [{"role": "user", "content": "hello"}]
        high = chat_with_metadata(
            "requested/model",
            messages,
            request_options={"reasoning": {"effort": "high"}},
        )
        maximum = chat_with_metadata(
            "requested/model",
            messages,
            request_options={"reasoning": {"effort": "xhigh"}},
        )
        self.assertNotEqual(high.input_sha256, maximum.input_sha256)

    @patch("rag.llm.API_KEY", "test-key")
    @patch("rag.llm.requests.post")
    def test_none_temperature_omits_parameter_for_reasoning_models(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "id": "response-123",
            "model": "provider/model-revision",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        post.return_value = response
        chat_with_metadata(
            "requested/model",
            [{"role": "user", "content": "hello"}],
            temperature=None,
        )
        self.assertNotIn("temperature", post.call_args.kwargs["json"])

    @patch("rag.llm.API_KEY", "test-key")
    @patch("rag.llm.requests.post")
    def test_request_options_cannot_override_explicit_contract(self, post):
        with self.assertRaisesRegex(ValueError, "cannot override"):
            chat_with_metadata(
                "requested/model",
                [{"role": "user", "content": "hello"}],
                request_options={"model": "different/model"},
            )
        post.assert_not_called()

    @patch("rag.generate.chat_with_metadata")
    def test_structured_generation_retries_invalid_contract_output(self, chat):
        chat.side_effect = [
            self.result("not-json", "r1"),
            self.result(json.dumps(VALID_ANSWER), "r2"),
        ]
        chunks = [
            {"id": "doc::h::0", "heading_key": "doc::h", "heading": "H", "text": "T"},
            {"id": "doc::o::0", "heading_key": "doc::o", "heading": "O", "text": "U"},
        ]
        answer, _, audit = structured_answer("question", chunks)
        self.assertEqual(answer, VALID_ANSWER)
        self.assertTrue(audit["validation"]["valid"])
        self.assertEqual(len(audit["calls"]), 2)
        self.assertEqual(audit["raw_outputs"][0], "not-json")
        self.assertEqual(len(audit["retry_feedback"]), 1)
        self.assertIn("invalid_json", audit["retry_feedback"][0])
        second_messages = chat.call_args_list[1].args[1]
        self.assertEqual(second_messages[-2]["role"], "assistant")
        self.assertEqual(second_messages[-2]["content"], "not-json")
        self.assertEqual(second_messages[-1]["role"], "user")
        self.assertIn("Return a new complete JSON object only", second_messages[-1]["content"])

    @patch("rag.claim_judge.judge_coverage")
    @patch("rag.claim_judge.verify_claims")
    def test_claim_judge_combines_independent_lanes(self, verify, coverage):
        verify.return_value.to_record.return_value = {
            "valid": True,
            "legacy_verdicts": [
                {
                    "claim_id": "c1",
                    "status": "supported",
                    "evidence": [{"citation_id": "S1", "quote": "Fact."}],
                    "rationale": "Supported.",
                }
            ],
            "errors": [],
            "metrics": {
                "generated_claim_count": 1,
                "supported_claim_count": 1,
                "generated_claim_support_rate": 1.0,
                "runtime_claim_support_pass": True,
            },
            "calls": [{"response_model": "support-model"}],
        }
        coverage.return_value.to_record.return_value = {
            "valid": True,
            "legacy_verdicts": [
                {
                    "claim_id": "c1",
                    "status": "covered",
                    "generated_claim_ids": ["c1"],
                    "rationale": "Covered.",
                }
            ],
            "errors": [],
            "metrics": {
                "required_claim_count": 1,
                "covered_required_claim_count": 1,
                "required_claim_coverage_rate": 1.0,
                "required_claim_coverage_pass": True,
            },
            "call": {"response_model": "coverage-model"},
            "calls": [{"response_model": "coverage-model"}],
            "raw_output": "{}",
            "raw_outputs": ["{}"],
        }
        answer = {
            "schema_version": "1.0.0",
            "action": "answer",
            "response": None,
            "claims": [{"id": "c1", "text": "Fact.", "citation_ids": ["S1"]}],
        }
        result = judge_claims(
            "question",
            [{"id": "c1", "text": "Fact."}],
            answer,
            [{"citation_id": "S1", "chunk_id": "doc::h::0", "text": "Fact."}],
        )
        self.assertTrue(result.valid)
        self.assertTrue(result.metrics["claim_level_pass"])
        self.assertEqual(result.judgment["schema_version"], "2.0.0")
        self.assertEqual(result.calls, ({"response_model": "coverage-model"},))

    @patch("rag.coverage_judge.chat_with_metadata")
    def test_coverage_judge_retries_malformed_output(self, chat):
        valid = {
            "schema_version": "1.0.0",
            "required_claim_verdicts": [
                {
                    "required_claim_id": "r1",
                    "status": "covered",
                    "generated_claim_ids": ["g1"],
                    "rationale": "Exact match.",
                }
            ],
        }
        chat.side_effect = [
            self.result('{"schema_version":', "j1"),
            self.result(json.dumps(valid), "j2"),
        ]
        answer = {
            "schema_version": "1.0.0",
            "action": "answer",
            "response": None,
            "claims": [{"id": "c1", "text": "Fact.", "citation_ids": ["S1"]}],
        }
        result = judge_coverage("question", [{"id": "c1", "text": "Fact."}], answer)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.calls), 2)

    @patch("rag.claim_verifier.chat_with_metadata")
    def test_gold_free_verifier_retries_and_emits_no_required_claims(self, chat):
        valid = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {
                    "generated_claim_id": "g1",
                    "status": "supported",
                    "evidence": [{"citation_id": "S1", "quote": "Fact."}],
                    "rationale": "S1 states it.",
                }
            ],
        }
        chat.side_effect = [
            self.result("not-json", "v1"),
            self.result(json.dumps(valid), "v2"),
        ]
        answer = {
            "schema_version": "1.0.0",
            "action": "answer",
            "response": None,
            "claims": [{"id": "c1", "text": "Fact.", "citation_ids": ["S1"]}],
        }
        result = verify_claims(
            "question",
            answer,
            [{"citation_id": "S1", "chunk_id": "doc::h::0", "text": "Fact."}],
        )
        self.assertTrue(result.valid)
        self.assertTrue(result.metrics["runtime_claim_support_pass"])
        self.assertEqual(len(result.calls), 2)

    def test_gold_free_verifier_rejects_invented_ids(self):
        judgment = {
            "schema_version": "1.0.0",
            "claim_verdicts": [
                {
                    "generated_claim_id": "g9",
                    "status": "supported",
                    "evidence": [{"citation_id": "S1", "quote": "Fact."}],
                    "rationale": "Wrong ID.",
                }
            ],
        }
        answer = {
            "claims": [{"id": "c1", "text": "Fact.", "citation_ids": ["S1"]}]
        }
        sources = [{"citation_id": "S1", "text": "Fact."}]
        errors, metrics = validate_verification(judgment, answer, sources)
        self.assertIn("id_order", {item["code"] for item in errors})
        self.assertFalse(verifier_support_pass({"valid": False, "metrics": metrics}))


class JudgeAgreementTests(unittest.TestCase):
    def test_agreement_reports_false_accepts_and_disagreements(self):
        answer = {"schema_version": "1.0.0", "action": "answer", "response": None, "claims": []}
        human = [
            {
                "case_id": "q1",
                "answer_sha256": answer_sha256(answer),
                "claim_labels": [
                    {"claim_id": "c1", "status": "unsupported"},
                    {"claim_id": "c2", "status": "supported"},
                ],
                "required_claim_labels": [
                    {"claim_id": "c1", "status": "missing", "generated_claim_ids": []}
                ],
            }
        ]
        items = [
            {
                "id": "q1",
                "answer": answer,
                "claim_judge": {
                    "judgment": {
                        "claim_verdicts": [
                            {"claim_id": "c1", "status": "supported", "rationale": "wrong"},
                            {"claim_id": "c2", "status": "supported", "rationale": "right"},
                        ],
                        "required_claim_verdicts": [
                            {
                                "claim_id": "c1",
                                "status": "covered",
                                "generated_claim_ids": ["c1"],
                                "rationale": "wrong",
                            }
                        ],
                    }
                },
            }
        ]
        report = compare_judgments(human, items)
        self.assertEqual(report["disagreement_count"], 2)
        self.assertEqual(report["generated_support"]["false_accept_count"], 1)
        self.assertEqual(report["required_coverage"]["false_accept_count"], 1)

    def test_kappa_handles_perfect_single_class_agreement(self):
        self.assertEqual(cohen_kappa([("supported", "supported")] * 3), 1.0)

    def test_human_calibration_schema_accepts_final_label(self):
        schema = json.loads(
            (ROOT / "evals" / "schema" / "judge-calibration-label.schema.json").read_text(
                encoding="utf-8"
            )
        )
        label = {
            "schema_version": "1.0.0",
            "case_id": "q1",
            "run_id": "run-1",
            "answer_sha256": "a" * 64,
            "claim_labels": [
                {
                    "claim_id": "c1",
                    "status": "supported",
                    "evidence": [
                        {"citation_id": "S1", "quote": "First exact span."},
                        {"citation_id": "S1", "quote": "Second exact span."},
                    ],
                    "rationale": "Both disconnected spans are required.",
                }
            ],
            "required_claim_labels": [
                {
                    "claim_id": "c1",
                    "status": "covered",
                    "generated_claim_ids": ["c1"],
                    "rationale": "The generated claim covers the requirement.",
                }
            ],
            "reviewer_id": "owner-reviewer-01",
            "reviewed_at": "2026-07-14T10:00:00+00:00",
            "status": "final",
            "notes": "",
        }
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(label)), [])


class StructuredRunnerTests(unittest.TestCase):
    def test_frozen_calibration_queue_has_all_175_blinded_tasks(self):
        report_path = ROOT / "reports" / "v2" / "structured-dev-current-k4.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        dataset_path = ROOT / report["dataset"]["path"]
        tasks, metadata = build_queue(report, load_cases(dataset_path))
        self.assertEqual(metadata["summary"]["support_task_count"], 80)
        self.assertEqual(metadata["summary"]["coverage_task_count"], 95)
        self.assertEqual(metadata["summary"]["total_task_count"], 175)
        self.assertEqual(metadata["summary"]["answered_case_count"], 42)
        self.assertEqual(
            metadata["summary"]["development_case_count"]
            + metadata["summary"]["confirmation_case_count"],
            42,
        )
        self.assertTrue(all("claim_judge" not in task for task in tasks))
        self.assertTrue(all("status" not in task for task in tasks))
        coverage = next(
            task for task in tasks if task["task_type"] == "required_claim_coverage"
        )
        self.assertNotIn("cited_sources", coverage)

    def test_bakeoff_profiles_are_strict_and_v4_max_maps_to_xhigh(self):
        validate_profiles(config.JUDGE_BAKEOFF_PROFILES)
        profiles = {item["id"]: item for item in config.JUDGE_BAKEOFF_PROFILES}
        self.assertEqual(
            profiles["v4-flash-max"]["request_options"]["reasoning"]["effort"],
            "xhigh",
        )
        self.assertTrue(
            profiles["gemini-3.5-flash-high"]["request_options"]["provider"][
                "require_parameters"
            ]
        )
        for profile in profiles.values():
            provider = profile["request_options"]["provider"]
            self.assertEqual(len(provider["only"]), 1)
            self.assertFalse(provider["allow_fallbacks"])

    def test_local_judge_is_explicit_diagnostic_not_bakeoff_roster(self):
        self.assertNotIn(
            "qwen3.5-9b-local-diagnostic",
            {item["id"] for item in config.JUDGE_BAKEOFF_PROFILES},
        )
        selected = select_bakeoff_profiles("qwen3.5-9b-local-diagnostic")
        self.assertEqual(
            [item["id"] for item in selected],
            ["qwen3.5-9b-local-diagnostic"],
        )
        self.assertEqual(selected[0]["request_options"], {"think": False})
        self.assertEqual(
            selected[0]["weights_blob_sha256"],
            "dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c",
        )

    def test_glm_candidates_are_provider_pinned_and_outside_frozen_roster(self):
        validate_profiles(config.EXPERIMENTAL_JUDGE_PROFILES)
        frozen_ids = {item["id"] for item in config.JUDGE_BAKEOFF_PROFILES}
        candidates = {
            item["id"]: item for item in config.EXPERIMENTAL_JUDGE_PROFILES
        }
        self.assertTrue(frozen_ids.isdisjoint(candidates))
        self.assertEqual(len(candidates), 4)
        for profile in candidates.values():
            provider = profile["request_options"]["provider"]
            self.assertIn(provider["only"], (["baidu/fp8"], ["streamlake/fp8"]))
            self.assertFalse(provider["allow_fallbacks"])
            self.assertTrue(provider["require_parameters"])
            self.assertEqual(
                provider["max_price"], {"prompt": 0.30, "completion": 0.90}
            )
        selected = select_bakeoff_profiles("glm-5.2-high-baidu")
        self.assertEqual([item["id"] for item in selected], ["glm-5.2-high-baidu"])

    def test_glm_operational_route_is_ordered_and_bounded(self):
        self.assertEqual(len(config.OPERATIONAL_JUDGE_PROFILES), 1)
        profile = config.OPERATIONAL_JUDGE_PROFILES[0]
        self.assertNotIn(
            profile["id"],
            {item["id"] for item in config.JUDGE_BAKEOFF_PROFILES},
        )
        provider = profile["request_options"]["provider"]
        expected = ["baidu/fp8", "streamlake/fp8"]
        self.assertEqual(provider["order"], expected)
        self.assertEqual(provider["only"], expected)
        self.assertTrue(provider["allow_fallbacks"])
        self.assertTrue(provider["require_parameters"])
        self.assertEqual(provider["max_price"], {"prompt": 0.30, "completion": 0.90})
        self.assertEqual(
            profile["calibration_profile_ids"],
            ["glm-5.2-xhigh-baidu", "glm-5.2-xhigh-streamlake"],
        )
        self.assertTrue((ROOT / profile["provider_parity_report"]).is_file())

    def test_glm_case_authoring_route_uses_high_effort_and_qualified_order(self):
        profile = config.OPERATIONAL_CASE_AUTHORING_PROFILES[0]
        self.assertEqual(
            profile["request_options"]["reasoning"],
            {"effort": "high", "exclude": True},
        )
        provider = profile["request_options"]["provider"]
        self.assertEqual(provider["order"], ["baidu/fp8", "streamlake/fp8"])
        self.assertEqual(provider["only"], ["baidu/fp8", "streamlake/fp8"])
        self.assertTrue(provider["allow_fallbacks"])
        self.assertTrue(provider["require_parameters"])

    def test_invalid_local_diagnostic_group_fails_every_task_closed(self):
        tasks = [
            {"task_id": "q1:support:c1"},
            {"task_id": "q1:support:c2"},
        ]
        predictions = direct_profile_predictions(
            {"q1:support": tasks},
            {"q1:support": {"valid": False, "verdicts": []}},
        )
        self.assertEqual(set(predictions), {row["task_id"] for row in tasks})
        self.assertTrue(
            all(row["decision"] == "fail_closed" for row in predictions.values())
        )

    def test_preflight_strict_response_validation(self):
        self.assertEqual(
            validate_smoke_content(
                '{"ok": true, "profile_id": "v4-flash-high"}',
                "v4-flash-high",
            ),
            {"ok": True, "profile_id": "v4-flash-high"},
        )
        with self.assertRaisesRegex(ValueError, "strict response mismatch"):
            validate_smoke_content(
                '{"ok": true, "profile_id": "wrong"}', "v4-flash-high"
            )

    def test_runtime_verifier_fails_closed(self):
        supported = {
            "valid": True,
            "metrics": {"generated_claim_count": 2, "supported_claim_count": 2},
        }
        unsupported = {
            "valid": True,
            "metrics": {"generated_claim_count": 2, "supported_claim_count": 1},
        }
        malformed = {
            "valid": False,
            "metrics": {"generated_claim_count": 2, "supported_claim_count": 2},
        }
        self.assertTrue(verifier_support_pass(supported))
        self.assertFalse(verifier_support_pass(unsupported))
        self.assertFalse(verifier_support_pass(malformed))

    def test_retrieval_diagnostics_require_complete_claim_evidence(self):
        case = {
            "answerability": "answerable",
            "acceptable_evidence": ["doc::a::span-1", "doc::b::span-2"],
            "required_claims": [
                {"id": "c1", "evidence_ids": ["doc::a::span-1"]},
                {"id": "c2", "evidence_ids": ["doc::a::span-1", "doc::b::span-2"]},
            ],
        }
        result = retrieval_diagnostics(
            case, [{"heading_key": "doc::a"}, {"heading_key": "doc::distractor"}]
        )
        self.assertTrue(result["retrieval_any_gold_evidence"])
        self.assertFalse(result["retrieval_complete_gold_evidence"])
        self.assertEqual(result["retrieval_required_claim_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
