"""Thin OpenAI-compatible chat client.

Base URL defaults to OpenRouter; set RAG_LLM_BASE_URL to route through the
Vbot gateway's budget-capped virtual key instead (spend-ledger story).

Implements the gateway project's own hard-won contract: an HTTP 200 with
empty content is a real failure mode (reasoning-spirals), so it is retried,
not returned blank. Status-code-only error handling ships empty answers.
"""
import os
import time

import requests

BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("RAG_LLM_KEY") or ""


def chat(model, messages, max_tokens=400, temperature=0.0, retries=4):
    if not API_KEY:
        raise RuntimeError("No API key: set OPENROUTER_API_KEY (or RAG_LLM_KEY).")
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=90,
            )
            if r.status_code == 200:
                content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
                if content.strip():
                    return content.strip()
                last = "HTTP 200 empty content (retryable)"
            else:
                last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:  # noqa: BLE001 — transient network, retry
            last = repr(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"chat failed after {retries} tries: {last}")
