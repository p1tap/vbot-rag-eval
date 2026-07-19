"""Export the exact public benchmark questions used by an evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def export_questions(*, cases_path: Path, normalized_paths: list[Path], output_path: Path) -> None:
    case_ids = [str(record["case_id"]) for record in _read_jsonl(cases_path)]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate case IDs in {cases_path}")

    selected = set(case_ids)
    questions: dict[str, dict[str, object]] = {}
    for normalized_path in normalized_paths:
        for record in _read_jsonl(normalized_path):
            case_id = str(record["id"])
            if case_id not in selected:
                continue
            provenance = record.get("provenance", {})
            questions[case_id] = {
                "schema_version": "1.0.0",
                "case_id": case_id,
                "benchmark_id": record["benchmark_id"],
                "source_split": record["source_split"],
                "question": record["query"],
                "provenance": {
                    "source_dataset": provenance.get("source_dataset"),
                    "source_version": provenance.get("source_version"),
                    "source_record_id": provenance.get("source_record_id"),
                    "source_record_sha256": provenance.get("source_record_sha256"),
                    "human_annotated": provenance.get("human_annotated"),
                },
            }

    missing = [case_id for case_id in case_ids if case_id not in questions]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing {len(missing)} selected questions: {preview}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case_id in case_ids:
            handle.write(json.dumps(questions[case_id], ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_questions(
        cases_path=args.cases,
        normalized_paths=args.normalized,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
