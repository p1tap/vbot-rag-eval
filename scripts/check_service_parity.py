"""Compare frozen retrieval fixtures across offline and live service paths."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from rag.retrieve import load_index, retrieve  # noqa: E402

FIXTURE = ROOT / "benchmarks" / "fixtures" / "service-parity.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "reports" / "service" / "parity.json"
    )
    args = parser.parse_args()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    index = load_index()
    rows = []
    for position, case in enumerate(fixture["cases"], 1):
        offline = [
            row["id"]
            for row in retrieve(
                case["question"],
                k=case["top_k"],
                index=index,
                unique_headings=True,
            )
        ]
        response = httpx.post(
            args.base_url.rstrip("/") + "/v1/retrieve",
            json={"question": case["question"], "top_k": case["top_k"]},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        service = [source["id"] for source in payload["sources"]]
        expected = case["expected_chunk_ids"]
        rows.append(
            {
                "case": position,
                "expected": expected,
                "offline": offline,
                "service": service,
                "passed": offline == service == expected,
                "request_id": payload.get("request_id"),
            }
        )
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "fixture": {
            "path": FIXTURE.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(FIXTURE),
            "index_fingerprint_sha256": fixture["index_fingerprint_sha256"],
        },
        "service_base_url": args.base_url,
        "case_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

