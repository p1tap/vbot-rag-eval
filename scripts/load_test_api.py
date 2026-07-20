"""Bounded HTTP load test with an explicit mock-vs-real inference declaration."""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def request_once(url: str, question: str, timeout: float) -> dict:
    started = time.perf_counter()
    try:
        response = httpx.post(
            url,
            json={"question": question, "top_k": 4},
            timeout=timeout,
        )
        return {
            "status": response.status_code,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "request_id": response.headers.get("x-request-id"),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - load report retains transport errors
        return {
            "status": None,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "request_id": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--generation-mode",
        choices=("mock_no_model_inference", "real_model_inference"),
        required=True,
    )
    parser.add_argument(
        "--out", type=Path, default=Path("reports/service/load-test.json")
    )
    args = parser.parse_args()
    if min(args.requests, args.concurrency) < 1:
        raise SystemExit("requests and concurrency must be positive")
    url = args.base_url.rstrip("/") + "/v1/answer"
    question = "What is the gateway's failover ladder, in order?"
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(request_once, url, question, args.timeout)
            for _ in range(args.requests)
        ]
        rows = [future.result() for future in as_completed(futures)]
    duration = time.perf_counter() - started
    latencies = [row["latency_ms"] for row in rows]
    successes = sum(row["status"] == 200 for row in rows)
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded_http_service_load_test",
        "generation_mode": args.generation_mode,
        "model_inference_capacity_evidence": args.generation_mode
        == "real_model_inference",
        "warning": (
            "Mock generation measures retrieval and HTTP control-plane throughput only; "
            "it is not model-inference capacity."
            if args.generation_mode == "mock_no_model_inference"
            else None
        ),
        "configuration": {
            "url": url,
            "request_count": args.requests,
            "concurrency": args.concurrency,
            "timeout_seconds": args.timeout,
        },
        "results": {
            "wall_seconds": duration,
            "success_count": successes,
            "error_count": len(rows) - successes,
            "error_rate": (len(rows) - successes) / len(rows),
            "throughput_requests_per_second": len(rows) / duration,
            "latency_ms": {
                "mean": sum(latencies) / len(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
                "max": max(latencies),
            },
        },
        "errors": [row for row in rows if row["status"] != 200],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if successes != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

