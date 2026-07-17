"""Small, auditable OpenAI Batch API primitives.

The module intentionally keeps network transport separate from benchmark
construction and scoring. Preparing and validating a run therefore needs no
API key, while paid submission requires an explicit caller decision.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import requests


RESPONSES_ENDPOINT = "/v1/responses"
TERMINAL_BATCH_STATUSES = {
    "completed",
    "expired",
    "failed",
    "cancelled",
}
OFFICIAL_MAX_REQUESTS_PER_FILE = 50_000
OFFICIAL_MAX_FILE_BYTES = 200_000_000


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_bytes(rows: Iterable[dict]) -> bytes:
    return b"".join(
        (canonical_json(row) + "\n").encode("utf-8") for row in rows
    )


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    atomic_write_bytes(path, jsonl_bytes(rows))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def responses_text_format(chat_response_format: dict) -> dict:
    """Convert a Chat Completions strict schema to Responses text.format."""

    if chat_response_format.get("type") != "json_schema":
        raise ValueError("Responses Batch requires a strict json_schema format")
    definition = chat_response_format.get("json_schema")
    if not isinstance(definition, dict):
        raise ValueError("json_schema definition is missing")
    required = {"name", "schema", "strict"}
    if not required <= set(definition):
        raise ValueError("json_schema must contain name, schema, and strict")
    return {
        "type": "json_schema",
        "name": definition["name"],
        "schema": definition["schema"],
        "strict": bool(definition["strict"]),
    }


def make_responses_request(
    *,
    custom_id: str,
    model: str,
    system: str,
    user: str,
    chat_response_format: dict,
    reasoning_effort: str,
    reasoning_mode: str,
    max_output_tokens: int,
) -> dict:
    if not custom_id or len(custom_id) > 128:
        raise ValueError("custom_id must contain 1 to 128 characters")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": RESPONSES_ENDPOINT,
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "reasoning": {
                "effort": reasoning_effort,
                "mode": reasoning_mode,
            },
            "max_output_tokens": max_output_tokens,
            "text": {"format": responses_text_format(chat_response_format)},
            "store": False,
        },
    }


def estimate_prompt_tokens(request: dict) -> int:
    """Return a conservative tokenizer-free prompt estimate for sharding.

    UTF-8 byte count divided by three intentionally overestimates typical
    English prompts. The estimate is a local queue-planning guard, not a claim
    about provider-billed tokens; actual usage is retained from responses.
    """

    body = request.get("body", {})
    input_value = body.get("input", []) if isinstance(body, dict) else []
    prompt_bytes = canonical_json(input_value).encode("utf-8")
    schema_value = body.get("text", {}) if isinstance(body, dict) else {}
    schema_bytes = canonical_json(schema_value).encode("utf-8")
    return max(1, math.ceil((len(prompt_bytes) + len(schema_bytes)) / 3) + 64)


def validate_requests(requests_: Sequence[dict]) -> None:
    seen = set()
    models = set()
    urls = set()
    for index, request in enumerate(requests_):
        custom_id = request.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"request {index} has no custom_id")
        if custom_id in seen:
            raise ValueError(f"duplicate custom_id: {custom_id}")
        seen.add(custom_id)
        if request.get("method") != "POST":
            raise ValueError(f"{custom_id}: method must be POST")
        urls.add(request.get("url"))
        body = request.get("body")
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            raise ValueError(f"{custom_id}: request body/model is invalid")
        models.add(body["model"])
    if len(urls) > 1:
        raise ValueError("one input file cannot mix endpoints")
    if len(models) > 1:
        raise ValueError("one input file cannot mix models")


def shard_requests(
    requests_: Sequence[dict],
    *,
    max_requests: int = OFFICIAL_MAX_REQUESTS_PER_FILE,
    max_file_bytes: int = 190_000_000,
    max_estimated_prompt_tokens: int = 1_000_000,
) -> list[dict]:
    """Split requests without changing their order or content."""

    if not requests_:
        raise ValueError("at least one request is required")
    if not 1 <= max_requests <= OFFICIAL_MAX_REQUESTS_PER_FILE:
        raise ValueError("max_requests exceeds the official 50,000 limit")
    if not 1 <= max_file_bytes < OFFICIAL_MAX_FILE_BYTES:
        raise ValueError("max_file_bytes must stay below the official 200 MiB limit")
    if max_estimated_prompt_tokens < 1:
        raise ValueError("max_estimated_prompt_tokens must be positive")
    validate_requests(requests_)

    shards: list[dict] = []
    current: list[dict] = []
    current_bytes = 0
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_bytes, current_tokens
        if not current:
            return
        shards.append(
            {
                "requests": current,
                "request_count": len(current),
                "jsonl_bytes": current_bytes,
                "estimated_prompt_tokens": current_tokens,
                "first_custom_id": current[0]["custom_id"],
                "last_custom_id": current[-1]["custom_id"],
            }
        )
        current = []
        current_bytes = 0
        current_tokens = 0

    for request in requests_:
        line_bytes = len((canonical_json(request) + "\n").encode("utf-8"))
        prompt_tokens = estimate_prompt_tokens(request)
        if line_bytes > max_file_bytes:
            raise ValueError(f"{request['custom_id']}: request exceeds file byte limit")
        if prompt_tokens > max_estimated_prompt_tokens:
            raise ValueError(
                f"{request['custom_id']}: request exceeds estimated prompt-token limit"
            )
        if current and (
            len(current) >= max_requests
            or current_bytes + line_bytes > max_file_bytes
            or current_tokens + prompt_tokens > max_estimated_prompt_tokens
        ):
            flush()
        current.append(request)
        current_bytes += line_bytes
        current_tokens += prompt_tokens
    flush()
    return shards


def extract_responses_output_text(response_body: dict) -> str:
    if not isinstance(response_body, dict):
        raise ValueError("response body must be an object")
    if response_body.get("status") not in {None, "completed"}:
        details = response_body.get("incomplete_details")
        raise ValueError(f"response status is {response_body.get('status')}: {details}")
    texts = []
    for output in response_body.get("output", []):
        if not isinstance(output, dict) or output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    texts.append(text)
    if not texts:
        raise ValueError("response contains no output_text")
    return "".join(texts)


def index_batch_results(output_rows: Sequence[dict], error_rows: Sequence[dict]) -> dict:
    """Index unordered provider rows and reject ambiguous duplicate IDs."""

    indexed: dict[str, dict] = {}
    for source, rows in (("output", output_rows), ("error", error_rows)):
        for row in rows:
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise ValueError(f"{source} row is missing custom_id")
            if custom_id in indexed:
                raise ValueError(f"duplicate provider result for {custom_id}")
            indexed[custom_id] = {"source": source, "row": row}
    return indexed


class OpenAIBatchClient:
    """Minimal REST client with bounded retries and dependency injection."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 4,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_response = None
        request_headers = {**self.headers, **kwargs.pop("headers", {})}
        for attempt in range(1, self.max_attempts + 1):
            response = self.session.request(
                method,
                url,
                headers=request_headers,
                timeout=self.timeout_seconds,
                **kwargs,
            )
            last_response = response
            if response.status_code not in {408, 409, 429} and response.status_code < 500:
                response.raise_for_status()
                return response
            if attempt < self.max_attempts:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2 ** (attempt - 1), 8)
                time.sleep(min(delay, 30.0))
        assert last_response is not None
        last_response.raise_for_status()
        return last_response

    def upload_batch_input(self, path: Path) -> dict:
        with path.open("rb") as handle:
            response = self._request(
                "POST",
                "/files",
                data={"purpose": "batch"},
                files={"file": (path.name, handle, "application/jsonl")},
            )
        return response.json()

    def create_batch(
        self, input_file_id: str, *, endpoint: str = RESPONSES_ENDPOINT,
        metadata: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": "24h",
        }
        if metadata:
            payload["metadata"] = metadata
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        return self._request("POST", "/batches", json=payload, headers=headers).json()

    def retrieve_batch(self, batch_id: str) -> dict:
        return self._request("GET", f"/batches/{batch_id}").json()

    def cancel_batch(self, batch_id: str) -> dict:
        return self._request("POST", f"/batches/{batch_id}/cancel").json()

    def download_file(self, file_id: str) -> bytes:
        return self._request("GET", f"/files/{file_id}/content").content
