"""Normalize and validate the pinned official SQuAD 2.0 dev set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.benchmarks import SquadV2Adapter  # noqa: E402
from rag.benchmarks.base import render_jsonl  # noqa: E402
from scripts.benchmarks.prepare import (  # noqa: E402
    build_report,
    file_sha256,
    source_artifacts,
    validate_normalized,
)

DEFAULT_SOURCE = ROOT / "artifacts" / "benchmarks" / "squad_v2" / "source" / "dev-v2.0.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "benchmarks" / "squad_v2" / "normalized" / "dev.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "public-benchmarks" / "squad-v2-prepare.json"


def load_registry_entry() -> dict:
    registry_path = ROOT / "benchmarks" / "single-hop-registry.json"
    schema_path = ROOT / "benchmarks" / "registry.schema.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(registry)
    return registry["benchmarks"][0]


def flatten_source(path: Path) -> list[dict]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("version") != "v2.0" or not isinstance(source.get("data"), list):
        raise ValueError("source is not the official SQuAD v2.0 JSON shape")
    rows = []
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                rows.append(
                    {
                        "id": qa["id"],
                        "title": article["title"],
                        "context": paragraph["context"],
                        "question": qa["question"],
                        "answers": qa["answers"],
                        "is_impossible": qa["is_impossible"],
                    }
                )
    return rows


def prepare(source: Path, output: Path, report_path: Path) -> dict:
    benchmark = load_registry_entry()
    actual_hash = file_sha256(source)
    if actual_hash != benchmark["raw_sha256"]:
        raise ValueError(
            f"raw checksum mismatch: expected {benchmark['raw_sha256']}, got {actual_hash}"
        )
    rows = flatten_source(source)
    if len(rows) != benchmark["target_cases"]:
        raise ValueError(
            f"source case count mismatch: expected {benchmark['target_cases']}, got {len(rows)}"
        )
    normalized = SquadV2Adapter().normalize_many(rows, "dev")
    validate_normalized(normalized)
    output_bytes = render_jsonl(normalized)
    artifacts = source_artifacts(source)
    report = build_report(
        benchmark,
        source,
        len(rows),
        artifacts,
        {
            "method": "all official dev cases in source order after unique-ID validation",
            "considered_cases": len(rows),
            "excluded_cases": 0,
            "exclusions": [],
        },
        normalized,
        output_bytes,
    )
    report["evaluation_lane"] = "oracle_context_reader"
    report["retrieval_included"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(output_bytes)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = prepare(args.source, args.output, args.report)
    print(f"SQuAD 2.0 normalized: cases={report['conversion']['normalized_cases']}")


if __name__ == "__main__":
    main()
