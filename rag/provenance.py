"""Scrubbed, machine-readable provenance for evaluation reports.

Only public-safe identifiers are recorded. Provider keys, endpoint values,
questions, and retrieved text never enter this metadata.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = (
    "huggingface-hub",
    "numpy",
    "requests",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _dependencies() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def capture_run_provenance(*, llm_enabled: bool) -> dict:
    corpus = sorted((ROOT / "corpus").glob("*.md"))
    inputs = [ROOT / "config.py", ROOT / "evals" / "golden.jsonl", *corpus]
    index = [
        ROOT / "index" / "meta.json",
        ROOT / "index" / "chunks.jsonl",
        ROOT / "index" / "embeddings.npy",
    ]
    route_kind = (
        "custom_openai_compatible"
        if os.environ.get("RAG_LLM_BASE_URL")
        else "openrouter_default"
    )
    return {
        "schema_version": "1.0.0",
        "run_id": f"rag-eval-{uuid.uuid4()}",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "commit": _git_value("rev-parse", "HEAD"),
            "tree": _git_value("show", "-s", "--format=%T", "HEAD"),
            "dirty": _git_dirty(),
        },
        "inputs_sha256": _hashes(inputs),
        "index_sha256": _hashes(index),
        "models": {
            "embedding": {
                "requested_id": config.EMBED_MODEL,
                "revision": config.EMBED_MODEL_REVISION,
            },
            "generator": {
                "requested_id": config.GEN_MODEL,
                "provider_revision": None,
                "request_options_sha256": sha256_text(
                    json.dumps(config.GEN_REQUEST_OPTIONS, sort_keys=True, separators=(",", ":"))
                ),
            },
            "judge": {
                "requested_id": config.JUDGE_MODEL,
                "provider_revision": None,
                "request_options_sha256": sha256_text(
                    json.dumps(config.JUDGE_REQUEST_OPTIONS, sort_keys=True, separators=(",", ":"))
                ),
            },
        },
        "generation": {
            "enabled": llm_enabled,
            "route_kind": route_kind,
            "answer_prompt_sha256": sha256_text(config.ANSWER_SYSTEM),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": _dependencies(),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
    }
