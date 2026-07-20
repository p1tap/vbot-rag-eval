from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import create_app  # noqa: E402


class ApiServiceTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {
            "id": "doc::heading::0",
            "heading_key": "doc::heading",
            "heading": "Heading",
            "text": "The first step is retrying the deployment.",
            "score": 0.9,
        }
        self.profile = {
            "id": "test-generator",
            "model": "test/model",
            "max_tokens": 100,
            "temperature": 0.0,
            "request_options": {},
            "provider_pinned": True,
        }

    def app(self, *, ready=True, valid=True):
        def answerer(question, chunks, profile):
            answer = {
                "schema_version": "1.0.0",
                "action": "answer",
                "response": None,
                "claims": [
                    {
                        "id": "c1",
                        "text": "Retry the deployment first.",
                        "citation_ids": ["S1"],
                    }
                ],
            }
            return answer, {
                "validation": {"valid": valid, "errors": [], "metrics": {}},
                "sources": [
                    {
                        "citation_id": "S1",
                        "chunk_id": self.chunk["id"],
                        "heading_key": self.chunk["heading_key"],
                        "heading": self.chunk["heading"],
                        "text": self.chunk["text"],
                        "score": self.chunk["score"],
                    }
                ],
            }

        return create_app(
            index_loader=lambda: (np.zeros((1, 2)), [self.chunk]),
            retriever=lambda question, top_k, index, unique: [self.chunk],
            answerer=answerer,
            versions_loader=lambda profile: {
                "service_version": "test",
                "generator": {"profile_id": profile["id"]},
            },
            readiness_check=lambda profile: (
                ready,
                None if ready else "generator unavailable",
            ),
            generator_profile=self.profile,
        )

    def test_health_version_and_request_id(self):
        with TestClient(self.app()) as client:
            live = client.get("/health/live", headers={"x-request-id": "fixed-id"})
            self.assertEqual(live.status_code, 200)
            self.assertEqual(live.headers["x-request-id"], "fixed-id")
            self.assertEqual(client.get("/health/ready").status_code, 200)
            self.assertEqual(
                client.get("/version").json()["generator"]["profile_id"],
                "test-generator",
            )

    def test_retrieval_and_answer_share_the_same_source(self):
        with TestClient(self.app()) as client:
            retrieved = client.post(
                "/v1/retrieve", json={"question": "What comes first?"}
            )
            answered = client.post(
                "/v1/answer", json={"question": "What comes first?"}
            )
            self.assertEqual(retrieved.status_code, 200)
            self.assertEqual(answered.status_code, 200)
            self.assertEqual(retrieved.json()["sources"][0]["id"], self.chunk["id"])
            self.assertEqual(answered.json()["sources"][0]["chunk_id"], self.chunk["id"])
            self.assertEqual(answered.json()["claims"][0]["citation_ids"], ["S1"])

    def test_unready_and_invalid_generation_fail_closed(self):
        with TestClient(self.app(ready=False)) as client:
            response = client.post("/v1/answer", json={"question": "Question"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "http_error")
        with TestClient(self.app(valid=False)) as client:
            response = client.post("/v1/answer", json={"question": "Question"})
            self.assertEqual(response.status_code, 502)

    def test_validation_and_metrics_surface(self):
        with TestClient(self.app()) as client:
            invalid = client.post("/v1/retrieve", json={"question": ""})
            self.assertEqual(invalid.status_code, 422)
            metrics = client.get("/metrics")
            self.assertEqual(metrics.status_code, 200)
            self.assertIn("vbot_rag_http_requests_total", metrics.text)


if __name__ == "__main__":
    unittest.main()

