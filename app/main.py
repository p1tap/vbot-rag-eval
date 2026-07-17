"""FastAPI service exposing the frozen retrieval and structured-answer paths."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

import config
from rag import llm
from rag.generate import structured_answer
from rag.provenance import sha256_file
from rag.retrieve import load_index, retrieve
from rag.structured_answer import render_response

ROOT = Path(__file__).resolve().parent.parent
SERVICE_VERSION = "1.0.0"
REQUESTS = Counter(
    "vbot_rag_http_requests_total",
    "HTTP requests handled by the Vbot RAG service.",
    ("method", "route", "status"),
)
REQUEST_DURATION = Histogram(
    "vbot_rag_http_request_duration_seconds",
    "End-to-end HTTP request latency.",
    ("method", "route"),
)
STAGE_DURATION = Histogram(
    "vbot_rag_stage_duration_seconds",
    "Latency of retrieval and generation stages.",
    ("stage",),
)


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=4, ge=1, le=20)
    unique_headings: bool = True


class AnswerRequest(RetrieveRequest):
    pass


class Source(BaseModel):
    citation_id: str
    chunk_id: str | None
    heading_key: str | None
    heading: str
    text: str
    score: float | None


class Claim(BaseModel):
    id: str
    text: str
    citation_ids: list[str]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def select_generator_profile(profile_id: str | None = None) -> dict[str, Any]:
    wanted = profile_id or os.environ.get(
        "RAG_SERVICE_GENERATOR_PROFILE", "qwen3.5-9b-local"
    )
    profiles = {profile["id"]: profile for profile in config.GENERATOR_EVAL_PROFILES}
    if wanted not in profiles:
        raise ValueError(f"unknown RAG service generator profile: {wanted}")
    return dict(profiles[wanted])


def service_versions(profile: dict[str, Any]) -> dict[str, Any]:
    index_meta_path = ROOT / "index" / "meta.json"
    corpus_path = ROOT / "corpus" / "manifest.json"
    dataset_path = ROOT / "evals" / "v2" / "dataset-manifest.json"
    index_meta = _json(index_meta_path)
    corpus = _json(corpus_path)
    dataset = _json(dataset_path)
    index_files = {
        "meta.json": _optional_sha256(index_meta_path),
        "chunks.jsonl": _optional_sha256(ROOT / "index" / "chunks.jsonl"),
        "embeddings.npy": _optional_sha256(ROOT / "index" / "embeddings.npy"),
    }
    index_fingerprint = hashlib.sha256(
        json.dumps(index_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "service_version": SERVICE_VERSION,
        "answer_contract_version": "1.0.0",
        "index": {**index_meta, "fingerprint_sha256": index_fingerprint},
        "corpus": {
            "id": corpus["corpus_id"],
            "version": corpus["corpus_version"],
            "fingerprint_sha256": corpus["fingerprint_sha256"],
        },
        "evaluation_dataset": {
            "version": dataset["dataset_version"],
            "fingerprint_sha256": dataset["fingerprint_sha256"],
        },
        "generator": {
            "profile_id": profile["id"],
            "model": profile["model"],
            "deployment_kind": profile.get("deployment_kind"),
            "provider_pinned": profile.get("provider_pinned", False),
        },
    }


def generator_readiness(profile: dict[str, Any]) -> tuple[bool, str | None]:
    expected = profile.get("endpoint")
    forced_native = os.environ.get("RAG_LLM_TRANSPORT", "").casefold() == "ollama_native"
    if expected and llm.BASE_URL != expected and not forced_native:
        return False, f"generator endpoint mismatch; expected {expected}"
    local_native = forced_native
    local_native = local_native or llm.BASE_URL in {
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    }
    if not llm.API_KEY and not local_native:
        return False, "generator API key is unavailable"
    return True, None


def default_retriever(
    question: str, top_k: int, index: tuple[Any, Any], unique_headings: bool
) -> list[dict[str, Any]]:
    return retrieve(
        question, k=top_k, index=index, unique_headings=unique_headings
    )


def default_answerer(
    question: str, chunks: list[dict[str, Any]], profile: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    answer, _, audit = structured_answer(
        question,
        chunks,
        allowed_actions=config.CURRENT_SUPPORTED_ACTIONS,
        model=profile["model"],
        max_tokens=profile["max_tokens"],
        temperature=profile["temperature"],
        request_options=profile["request_options"],
    )
    return answer, audit


def create_app(
    *,
    index_loader: Callable[[], tuple[Any, Any]] = load_index,
    retriever: Callable[..., list[dict[str, Any]]] = default_retriever,
    answerer: Callable[..., tuple[dict[str, Any] | None, dict[str, Any]]] = default_answerer,
    versions_loader: Callable[[dict[str, Any]], dict[str, Any]] = service_versions,
    readiness_check: Callable[[dict[str, Any]], tuple[bool, str | None]] = generator_readiness,
    generator_profile: dict[str, Any] | None = None,
) -> FastAPI:
    profile = generator_profile or select_generator_profile()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.index = await run_in_threadpool(index_loader)
        # E5 loads lazily. Warm it before readiness so concurrent first requests
        # cannot race inside tokenizer/model initialization.
        await run_in_threadpool(
            retriever,
            "service readiness probe",
            1,
            application.state.index,
            True,
        )
        application.state.versions = versions_loader(profile)
        ready, reason = readiness_check(profile)
        application.state.generator_ready = ready
        application.state.generator_not_ready_reason = reason
        yield

    application = FastAPI(
        title="Vbot RAG API",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def observe_request(request: Request, call_next):
        request_id = request.headers.get("x-request-id", "")
        if not request_id or len(request_id) > 128:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            route_name = getattr(route, "path", request.url.path)
            REQUESTS.labels(request.method, route_name, str(status)).inc()
            REQUEST_DURATION.labels(request.method, route_name).observe(
                time.perf_counter() - started
            )
        response.headers["x-request-id"] = request_id
        return response

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {"code": "http_error", "message": str(exc.detail)},
                "request_id": request.state.request_id,
            },
        )

    @application.get("/health/live")
    async def live():
        return {"status": "ok", "service_version": SERVICE_VERSION}

    @application.get("/health/ready")
    async def ready(request: Request):
        if not request.app.state.generator_ready:
            raise HTTPException(
                503, request.app.state.generator_not_ready_reason or "generator unavailable"
            )
        return {"status": "ready", "versions": request.app.state.versions}

    @application.get("/version")
    async def version(request: Request):
        return request.app.state.versions

    @application.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.post("/v1/retrieve")
    async def retrieve_endpoint(payload: RetrieveRequest, request: Request):
        started = time.perf_counter()
        chunks = await run_in_threadpool(
            retriever,
            payload.question,
            payload.top_k,
            request.app.state.index,
            payload.unique_headings,
        )
        elapsed = time.perf_counter() - started
        STAGE_DURATION.labels("retrieval").observe(elapsed)
        return {
            "request_id": request.state.request_id,
            "sources": chunks,
            "latency_ms": elapsed * 1000,
            "versions": request.app.state.versions,
        }

    @application.post("/v1/answer")
    async def answer_endpoint(payload: AnswerRequest, request: Request):
        if not request.app.state.generator_ready:
            raise HTTPException(
                503, request.app.state.generator_not_ready_reason or "generator unavailable"
            )
        retrieval_started = time.perf_counter()
        chunks = await run_in_threadpool(
            retriever,
            payload.question,
            payload.top_k,
            request.app.state.index,
            payload.unique_headings,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started
        STAGE_DURATION.labels("retrieval").observe(retrieval_seconds)
        generation_started = time.perf_counter()
        answer, audit = await run_in_threadpool(answerer, payload.question, chunks, profile)
        generation_seconds = time.perf_counter() - generation_started
        STAGE_DURATION.labels("generation").observe(generation_seconds)
        validation = audit.get("validation") or {}
        if answer is None or validation.get("valid") is not True:
            raise HTTPException(502, "generator failed the structured-answer contract")
        sources = audit.get("sources", [])
        return {
            "request_id": request.state.request_id,
            "action": answer["action"],
            "response": render_response(answer),
            "claims": answer["claims"],
            "sources": sources,
            "validation": validation,
            "latency_ms": {
                "retrieval": retrieval_seconds * 1000,
                "generation": generation_seconds * 1000,
                "total": (retrieval_seconds + generation_seconds) * 1000,
            },
            "versions": request.app.state.versions,
        }

    return application


app = create_app()
