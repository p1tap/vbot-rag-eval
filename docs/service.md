# Vbot RAG service

The Phase 7 service exposes the same frozen local index and V2 structured-answer
contract used by offline evaluation. It is intentionally small: one immutable
index is loaded during process lifespan, retrieval is read-only, and malformed
model output fails with HTTP 502 instead of leaking unvalidated prose.

## Run locally

```powershell
$env:RAG_LLM_BASE_URL='http://localhost:11434'
$env:RAG_LLM_TRANSPORT='ollama_native'
$env:RAG_SERVICE_GENERATOR_PROFILE='qwen3.5-9b-local'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

- `GET /health/live` confirms the HTTP process is alive.
- `GET /health/ready` confirms the index loaded and the configured generator
  route is usable.
- `GET /version` reports service, answer contract, index, corpus, dataset, and
  generator identities.
- `POST /v1/retrieve` returns stable chunk IDs, scores, and version metadata.
- `POST /v1/answer` returns only validated action/claims/citations plus per-stage
  latency. Raw provider output is never exposed.
- `GET /metrics` exports request and retrieval/generation latency metrics in
  Prometheus text format.

Every response gets an `x-request-id`. Questions are not used as metric labels.
The process keeps one immutable index snapshot; an index update is activated by
starting a new process and switching traffic only after `/health/ready` passes.
The lifespan hook also runs one E5 retrieval before readiness, preventing
concurrent first requests from racing during lazy model initialization.

## Container

Build with `docker build -f Dockerfile.api -t vbot-rag-api .`. When Ollama runs
on the host, set `RAG_LLM_BASE_URL=http://host.docker.internal:11434` and
`RAG_LLM_TRANSPORT=ollama_native`. The experiment profile's exact endpoint is
still reported separately in evaluation artifacts.

The clean-image mock smoke is recorded in
`reports/service/container-smoke.json`: readiness, version reporting, E5
retrieval, answer routing, and structured validation all passed. Its explicit
mock-generation label prevents it from being cited as inference evidence.

The final local real-model report is
`reports/service/load-test-real-qwen35-20.json`: 20/20 requests succeeded at
concurrency one with zero errors, 0.302 requests/s, mean 3.31 s, p95 3.39 s,
and p99 3.44 s. It targeted `app.main` with pinned local Qwen3.5 9B. The final
`vbot-rag-api:v2-dev5` image digest is
`sha256:d4732ac777cac259ba6ac203e094d2a40e1c6d72e1a21d8c54afc8925390b4b6`;
its separate 100-request mock report is
`reports/service/load-test-mock-dev5.json` and is not model-capacity evidence.

`scripts/check_service_parity.py` additionally compares four frozen retrieval
fixtures across the offline function and live `/v1/retrieve`; expected chunk
IDs are pinned in `benchmarks/fixtures/service-parity.json`.

## Bounded load test

Mock generation is opt-in and refuses to import unless explicitly enabled:

```powershell
$env:ALLOW_RAG_MOCK_SERVICE='1'
uvicorn app.mock:app --host 127.0.0.1 --port 8001
python scripts/load_test_api.py --base-url http://127.0.0.1:8001 `
  --requests 100 --concurrency 4 --generation-mode mock_no_model_inference `
  --out reports/service/load-test-mock.json
```

That report measures HTTP plus real E5 retrieval/control-plane behavior. It is
explicitly not model-inference capacity. Real inference uses
`--generation-mode real_model_inference` and must target `app.main`; keep it
bounded to avoid conflating provider limits with service throughput. The
deferred 100-hour endurance test remains outside the RAG completion run.
