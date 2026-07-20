"""Publish scrubbed deterministic quality evidence from local retrieval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "public-benchmarks" / "retrieval"


def publish(input_path: Path, output_root: Path) -> Path:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    required = {"report_version", "benchmark_id", "retriever", "source", "cases", "metrics"}
    missing = sorted(required - report.keys())
    if missing:
        raise ValueError(f"retrieval report is missing fields: {', '.join(missing)}")
    scrubbed = {
        key: value
        for key, value in report.items()
        if key not in {"latency_ms_per_query", "ranking_latency_ms_per_query"}
    }
    scrubbed["status"] = "retrieval_only_generation_pending"
    retriever_id = report["retriever"]["name"].replace("_", "-")
    output = output_root / f"{report['benchmark_id']}-{retriever_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(scrubbed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for input_path in args.inputs:
        print(f"published retrieval evidence: {publish(input_path, args.output_root)}")


if __name__ == "__main__":
    main()
