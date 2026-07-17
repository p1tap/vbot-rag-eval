import json
from argparse import Namespace
from pathlib import Path

import pytest

from rag.openai_batch import (
    OpenAIBatchClient,
    extract_responses_output_text,
    index_batch_results,
    make_responses_request,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    sha256_file,
    shard_requests,
)
from scripts.benchmarks.run_end_to_end import response_format
from scripts.benchmarks import run_openai_batch_end_to_end as batch_runner
from scripts.benchmarks.run_openai_batch_end_to_end import evaluated_attempt


def sample_work(case_id="hotpotqa:test"):
    return {
        "case_id": case_id,
        "benchmark_id": "hotpotqa",
        "query": "Who wrote the book?",
        "gold": {"answers": ["Ada"], "answerability": "answerable"},
        "gold_document_ids": ["doc-1"],
        "gold_evidence_sets": [["doc-1"]],
        "contexts": [
            {
                "citation_id": "D1",
                "document_id": "doc-1",
                "title": "Book",
                "text": "The book was written by Ada.",
            }
        ],
        "retrieved_document_ids": ["doc-1"],
    }


def sample_request(custom_id="rag-1", *, user="question"):
    work = sample_work()
    wire = [{**work, "case_id": "C1"}]
    return make_responses_request(
        custom_id=custom_id,
        model="gpt-5.6-sol",
        system="system",
        user=user,
        chat_response_format=response_format(wire, True),
        reasoning_effort="xhigh",
        reasoning_mode="pro",
        max_output_tokens=8000,
    )


def test_responses_request_preserves_strict_schema_and_reasoning_mode():
    request = sample_request()

    assert request["url"] == "/v1/responses"
    assert request["body"]["model"] == "gpt-5.6-sol"
    assert request["body"]["reasoning"] == {"effort": "xhigh", "mode": "pro"}
    assert request["body"]["store"] is False
    text_format = request["body"]["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert text_format["schema"]["additionalProperties"] is False


def test_sharding_is_ordered_and_respects_token_budget():
    requests = [
        sample_request(f"rag-{index}", user="x" * 300) for index in range(5)
    ]
    one_request_tokens = sum(
        shard["estimated_prompt_tokens"]
        for shard in shard_requests(requests[:1], max_estimated_prompt_tokens=100_000)
    )
    shards = shard_requests(
        requests,
        max_requests=50,
        max_file_bytes=1_000_000,
        max_estimated_prompt_tokens=one_request_tokens * 2,
    )

    assert [row["request_count"] for row in shards] == [2, 2, 1]
    assert [
        request["custom_id"] for shard in shards for request in shard["requests"]
    ] == [f"rag-{index}" for index in range(5)]


def test_sharding_rejects_duplicate_custom_ids():
    with pytest.raises(ValueError, match="duplicate custom_id"):
        shard_requests([sample_request(), sample_request()])


def test_extract_responses_output_text_ignores_reasoning_items():
    body = {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": '{"results":[]}'},
                    {"type": "refusal", "refusal": "unused"},
                ],
            },
        ],
    }

    assert extract_responses_output_text(body) == '{"results":[]}'


def test_index_batch_results_matches_unordered_custom_ids_and_rejects_duplicates():
    indexed = index_batch_results(
        [{"custom_id": "b", "response": {}}],
        [{"custom_id": "a", "error": {"code": "expired"}}],
    )
    assert indexed["a"]["source"] == "error"
    assert indexed["b"]["source"] == "output"
    with pytest.raises(ValueError, match="duplicate provider result"):
        index_batch_results(
            [{"custom_id": "a", "response": {}}],
            [{"custom_id": "a", "error": {}}],
        )


def test_provider_attempt_is_validated_and_mapped_back_to_original_case_id():
    work = sample_work()
    content = json.dumps(
        {
            "results": [
                {"case_id": "C1", "prediction": "Ada", "citation_ids": ["D1"]}
            ]
        }
    )
    entry = {
        "source": "output",
        "batch_id": "batch-1",
        "generation": 0,
        "shard_index": 0,
        "row": {
            "custom_id": "rag-1",
            "response": {
                "status_code": 200,
                "request_id": "req-1",
                "body": {
                    "id": "resp-1",
                    "model": "gpt-5.6-sol",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 25},
                },
            },
            "error": None,
        },
    }

    record = evaluated_attempt(entry, work, "gpt-5.6-sol-pro-xhigh")

    assert record["valid"] is True
    assert record["results"] == [
        {
            "case_id": "hotpotqa:test",
            "prediction": "Ada",
            "citation_ids": ["D1"],
        }
    ]
    assert record["calls"][0]["usage"]["input_tokens"] == 100


class FakeResponse:
    def __init__(self, *, json_value=None, content=b"", status_code=200):
        self._json = json_value
        self.content = content
        self.status_code = status_code
        self.headers = {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def test_client_uses_direct_files_batches_and_content_endpoints(tmp_path):
    input_path = tmp_path / "batch.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    session = FakeSession(
        [
            FakeResponse(json_value={"id": "file-1"}),
            FakeResponse(json_value={"id": "batch-1", "status": "validating"}),
            FakeResponse(json_value={"id": "batch-1", "status": "completed"}),
            FakeResponse(content=b'{"custom_id":"rag-1"}\n'),
        ]
    )
    client = OpenAIBatchClient(api_key="test-key", session=session)

    assert client.upload_batch_input(input_path)["id"] == "file-1"
    assert client.create_batch("file-1", idempotency_key="idem-1")["id"] == "batch-1"
    assert client.retrieve_batch("batch-1")["status"] == "completed"
    assert client.download_file("file-output") == b'{"custom_id":"rag-1"}\n'

    assert [call[0:2] for call in session.calls] == [
        ("POST", "https://api.openai.com/v1/files"),
        ("POST", "https://api.openai.com/v1/batches"),
        ("GET", "https://api.openai.com/v1/batches/batch-1"),
        ("GET", "https://api.openai.com/v1/files/file-output/content"),
    ]
    assert session.calls[1][2]["json"]["endpoint"] == "/v1/responses"
    assert session.calls[1][2]["headers"]["Idempotency-Key"] == "idem-1"
    assert session.calls[0][2]["headers"]["Authorization"] == "Bearer test-key"


def test_finalize_replays_real_contract_and_publishes_hash_bound_report(
    tmp_path, monkeypatch
):
    work = sample_work()
    custom_id = "rag-integration"
    works_path = tmp_path / "work-items.jsonl"
    output_path = tmp_path / "provider-output.jsonl"
    report_path = tmp_path / "report.json"
    atomic_write_jsonl(works_path, [work])
    generated = json.dumps(
        {
            "results": [
                {"case_id": "C1", "prediction": "Ada", "citation_ids": ["D1"]}
            ]
        }
    )
    atomic_write_jsonl(
        output_path,
        [
            {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "request_id": "req-integration",
                    "body": {
                        "id": "resp-integration",
                        "model": "gpt-5.6-sol",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": generated}
                                ],
                            }
                        ],
                        "usage": {"input_tokens": 120, "output_tokens": 30},
                    },
                },
                "error": None,
            }
        ],
    )
    identity = {
        "runner_source_sha256": sha256_file(Path(batch_runner.__file__)),
        "transport_source_sha256": sha256_file(
            Path(batch_runner.__file__).parents[2] / "rag" / "openai_batch.py"
        ),
        "profile_id": "gpt-5.6-sol-pro-xhigh",
        "benchmarks": ["hotpotqa"],
        "shard_limits": {
            "max_requests": 50_000,
            "max_file_bytes": 190_000_000,
            "max_estimated_prompt_tokens": 1_000_000,
        },
    }
    manifest = {
        "schema_version": "1.0.0",
        "state": "downloaded",
        "identity_sha256": canonical_sha256(identity),
        "identity": identity,
        "output_report_path": str(report_path),
        "work_items_path": str(works_path),
        "work_items_sha256": sha256_file(works_path),
        "case_count": 1,
        "request_index": {
            custom_id: {
                "case_id": work["case_id"],
                "benchmark_id": work["benchmark_id"],
                "work_sha256": canonical_sha256(work),
                "request_sha256": "fixture",
            }
        },
        "shards": [
            {
                "generation": 0,
                "shard_index": 0,
                "state": "downloaded",
                "batch_id": "batch-integration",
                "batch_status": "completed",
                "output_path": str(output_path),
                "output_sha256": sha256_file(output_path),
            }
        ],
    }
    atomic_write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(
        batch_runner,
        "audit_suite",
        lambda: {
            "public_case_count": 10_000,
            "status": "machine_validated_ai_spot_audit_passed",
            "benchmarks": {"hotpotqa": {"cases": 3000}},
        },
    )

    batch_runner.finalize(Namespace(run_dir=tmp_path))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "sample_complete"
    assert report["valid_batch_count"] == 1
    assert report["fail_closed_case_count"] == 0
    assert report["metrics"]["hotpotqa"]["joint_correct_rate"] == 1.0
    assert report["operation"]["estimated_batch_cost_usd"] == pytest.approx(
        (120 * 2.5 + 30 * 15) / 1_000_000
    )
    assert report["artifacts"]["run_manifest_sha256"] == sha256_file(
        tmp_path / "manifest.json"
    )
