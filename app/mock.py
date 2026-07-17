"""Explicit local-only mock-generation service for control-plane load tests."""
from __future__ import annotations

import os

from rag.structured_answer import context_sources

from app.main import create_app, service_versions

if os.environ.get("ALLOW_RAG_MOCK_SERVICE") != "1":
    raise RuntimeError(
        "mock service is disabled; set ALLOW_RAG_MOCK_SERVICE=1 only for load tests"
    )

PROFILE = {
    "id": "mock-generation-load-test",
    "model": "deterministic-mock-no-inference",
    "max_tokens": 0,
    "temperature": 0.0,
    "request_options": {},
    "provider_pinned": True,
    "deployment_kind": "mock_generation",
}


def mock_answerer(question, chunks, profile):
    sources = context_sources(chunks)
    if not sources:
        answer = {
            "schema_version": "1.0.0",
            "action": "abstain_absent",
            "response": "No source was retrieved by the load-test fixture.",
            "claims": [],
        }
    else:
        answer = {
            "schema_version": "1.0.0",
            "action": "answer",
            "response": None,
            "claims": [
                {
                    "id": "c1",
                    "text": "Mock generation completed; this is not a quality answer.",
                    "citation_ids": ["S1"],
                }
            ],
        }
    return answer, {
        "validation": {
            "valid": True,
            "errors": [],
            "metrics": {"mock_generation": True},
        },
        "sources": sources,
    }


def mock_versions(profile):
    versions = service_versions(profile)
    versions["generation_mode"] = "mock_no_model_inference"
    versions["not_model_capacity_evidence"] = True
    return versions


app = create_app(
    answerer=mock_answerer,
    versions_loader=mock_versions,
    readiness_check=lambda profile: (True, None),
    generator_profile=PROFILE,
)

