"""Thin auditable chat client for OpenAI-compatible and local Ollama routes.

Base URL defaults to OpenRouter; set RAG_LLM_BASE_URL to route through the
Vbot gateway's budget-capped virtual key instead (spend-ledger story).

Implements the gateway project's own hard-won contract: an HTTP 200 with
empty content is a real failure mode (reasoning-spirals), so it is retried,
not returned blank. Status-code-only error handling ships empty answers.
"""
import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import requests

BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("RAG_LLM_KEY") or ""
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("RAG_LLM_TIMEOUT_SECONDS", "90"))


def _is_native_ollama() -> bool:
    forced = os.environ.get("RAG_LLM_TRANSPORT", "").strip().casefold()
    return forced == "ollama_native" or BASE_URL in {
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    }


@dataclass(frozen=True)
class ChatResult:
    """Exact model output plus enough metadata to audit the call."""

    content: str
    requested_model: str
    response_model: str | None
    response_id: str | None
    input_sha256: str
    latency_ms: float
    attempts: int
    retries: int
    usage: dict
    cost: float | None
    request_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    endpoint_kind: str = ""
    endpoint_sha256: str = ""
    request_options: dict = field(default_factory=dict)
    response_provider: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    quantization: str | None = None
    service_tier: str | None = None

    def to_record(self) -> dict:
        return asdict(self)


def _request_payload(
    model, messages, max_tokens, temperature, response_format, request_options
):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    # Some reasoning models do not advertise sampling-temperature support.
    # ``None`` is an explicit request to omit the parameter, not JSON null.
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format
    options = deepcopy(request_options or {})
    if not isinstance(options, dict):
        raise TypeError("request_options must be an object")
    protected = set(options) & {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "response_format",
    }
    if protected:
        raise ValueError(
            f"request_options cannot override explicit parameters: {sorted(protected)}"
        )
    payload.update(options)
    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return payload, options


def _ollama_request_payload(
    model, messages, max_tokens, temperature, response_format, request_options
):
    audited_options = deepcopy(request_options or {})
    if not isinstance(audited_options, dict):
        raise TypeError("request_options must be an object")
    protected = set(audited_options) & {
        "model",
        "messages",
        "stream",
        "format",
        "options",
    }
    if protected:
        raise ValueError(
            f"request_options cannot override explicit parameters: {sorted(protected)}"
        )
    options = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if "think" in audited_options:
        payload["think"] = bool(audited_options["think"])
    unknown = set(audited_options) - {"think"}
    if unknown:
        raise ValueError(f"unsupported native Ollama request options: {sorted(unknown)}")
    if response_format is not None:
        response_type = response_format.get("type")
        if response_type == "json_object":
            payload["format"] = "json"
        elif response_type == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema")
            if not isinstance(schema, dict):
                raise ValueError("native Ollama JSON Schema response is missing schema")
            payload["format"] = schema
        else:
            raise ValueError(f"unsupported native Ollama response format: {response_type}")
    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return payload, audited_options


def _ollama_usage(body: dict) -> dict:
    prompt = int(body.get("prompt_eval_count") or 0)
    completion = int(body.get("eval_count") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _input_hash(endpoint: str, payload: dict):
    value = {"endpoint": endpoint, "payload": payload}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def chat_with_metadata(
    model,
    messages,
    max_tokens=400,
    temperature=0.0,
    retries=4,
    response_format=None,
    request_options=None,
    timeout_seconds=None,
):
    native_ollama = _is_native_ollama()
    if not API_KEY and not native_ollama:
        raise RuntimeError("No API key: set OPENROUTER_API_KEY (or RAG_LLM_KEY).")
    last = None
    started = time.perf_counter()
    endpoint = f"{BASE_URL}/api/chat" if native_ollama else f"{BASE_URL}/chat/completions"
    payload, audited_options = (
        _ollama_request_payload(
            model,
            messages,
            max_tokens,
            temperature,
            response_format,
            request_options,
        )
        if native_ollama
        else _request_payload(
            model,
            messages,
            max_tokens,
            temperature,
            response_format,
            request_options,
        )
    )
    timeout_seconds = (
        DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    input_sha256 = _input_hash(endpoint, payload)
    for attempt in range(retries):
        try:
            headers = {"Content-Type": "application/json"}
            if not native_ollama:
                headers.update(
                    {
                        "Authorization": f"Bearer {API_KEY}",
                        "X-OpenRouter-Metadata": "enabled",
                    }
                )
            r = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            if r.status_code == 200:
                body = r.json()
                content = (
                    body.get("message", {}).get("content", "")
                    if native_ollama
                    else (body.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content.strip():
                    usage = (
                        _ollama_usage(body)
                        if native_ollama
                        else body.get("usage")
                        if isinstance(body.get("usage"), dict)
                        else {}
                    )
                    raw_cost = usage.get("cost")
                    cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else None
                    choice = {} if native_ollama else (body.get("choices") or [{}])[0]
                    metadata = body.get("openrouter_metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                    return ChatResult(
                        content=content.strip(),
                        requested_model=model,
                        response_model=body.get("model"),
                        response_id=None if native_ollama else body.get("id"),
                        input_sha256=input_sha256,
                        latency_ms=round((time.perf_counter() - started) * 1000, 3),
                        attempts=attempt + 1,
                        retries=attempt,
                        usage=usage,
                        cost=cost,
                        request_timeout_seconds=timeout_seconds,
                        endpoint_kind=(
                            "local_ollama_native"
                            if native_ollama
                            else "openrouter_default"
                            if BASE_URL == "https://openrouter.ai/api/v1"
                            else "custom_openai_compatible"
                        ),
                        endpoint_sha256=hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
                        request_options=audited_options,
                        response_provider="ollama-local"
                        if native_ollama
                        else body.get("provider")
                        or metadata.get("provider_name")
                        or metadata.get("provider"),
                        system_fingerprint=(
                            None if native_ollama else body.get("system_fingerprint")
                        ),
                        finish_reason=(
                            body.get("done_reason")
                            if native_ollama
                            else choice.get("finish_reason")
                        ),
                        native_finish_reason=(
                            body.get("done_reason")
                            if native_ollama
                            else choice.get("native_finish_reason")
                        ),
                        quantization=body.get("quantization")
                        or metadata.get("quantization"),
                        service_tier=body.get("service_tier")
                        or metadata.get("service_tier"),
                    )
                last = "HTTP 200 empty content (retryable)"
            else:
                last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:  # noqa: BLE001 — transient network, retry
            last = repr(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"chat failed after {retries} tries: {last}")


def chat(model, messages, max_tokens=400, temperature=0.0, retries=4):
    """Compatibility wrapper for the frozen V1 free-form generation path."""

    return chat_with_metadata(
        model,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        retries=retries,
    ).content


__all__ = ["ChatResult", "chat", "chat_with_metadata"]
