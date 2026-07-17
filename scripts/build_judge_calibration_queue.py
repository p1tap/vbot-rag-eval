"""Freeze a blinded, answer-hash-bound human judge calibration queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.judge_agreement import answer_sha256  # noqa: E402
from rag.provenance import sha256_file  # noqa: E402

DEFAULT_REPORT = ROOT / "reports" / "v2" / "structured-dev-current-k4.json"
DEFAULT_DIR = ROOT / "evals" / "v2" / "judge-calibration"


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cases(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("cases"), list):
        return value["cases"]
    raise ValueError(f"{path} does not contain a case array")


def _confirmation_cases(case_ids: list[str], run_id: str) -> set[str]:
    count = max(1, round(len(case_ids) * 0.20))
    ranked = sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(
            f"{run_id}:{case_id}:confirmation-v1".encode("utf-8")
        ).hexdigest(),
    )
    return set(ranked[:count])


def _source_view(source: dict) -> dict:
    """Remove retrieval scores and other signals that could bias a reviewer."""
    return {
        key: source[key]
        for key in ("citation_id", "chunk_id", "heading_key", "heading", "text")
        if key in source
    }


def build_queue(report: dict, cases: list[dict]) -> tuple[list[dict], dict]:
    case_by_id = {case["id"]: case for case in cases}
    report_ids = [item["id"] for item in report.get("items", [])]
    if len(report_ids) != len(set(report_ids)):
        raise ValueError("evaluation report has duplicate case IDs")
    missing_cases = set(report_ids) - set(case_by_id)
    if missing_cases:
        raise ValueError(f"dataset is missing report cases: {sorted(missing_cases)}")
    run_id = report.get("provenance", {}).get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("evaluation report is missing provenance.run_id")

    answered_ids = [
        item["id"]
        for item in report["items"]
        if item.get("answer", {}).get("action") == "answer"
    ]
    confirmation = _confirmation_cases(answered_ids, run_id)
    tasks: list[dict] = []
    answer_hashes: dict[str, str] = {}
    case_partitions: dict[str, str] = {}

    for item in report["items"]:
        answer = item.get("answer")
        if not isinstance(answer, dict) or answer.get("action") != "answer":
            continue
        case = case_by_id[item["id"]]
        digest = answer_sha256(answer)
        answer_hashes[item["id"]] = digest
        case_partitions[item["id"]] = (
            "confirmation" if item["id"] in confirmation else "development"
        )
        sources = item.get("generation", {}).get("sources", [])
        source_by_id = {
            source["citation_id"]: source
            for source in sources
            if isinstance(source, dict) and isinstance(source.get("citation_id"), str)
        }
        generated_claims = answer.get("claims", [])
        for claim in generated_claims:
            citation_ids = claim.get("citation_ids", [])
            if any(citation_id not in source_by_id for citation_id in citation_ids):
                raise ValueError(
                    f"{item['id']} claim {claim.get('id')} cites an absent source"
                )
            tasks.append(
                {
                    "schema_version": "1.0.0",
                    "task_id": f"{item['id']}:support:{claim['id']}",
                    "task_type": "generated_claim_support",
                    "case_id": item["id"],
                    "run_id": run_id,
                    "answer_sha256": digest,
                    "lane": item.get("lane"),
                    "severity": item.get("severity"),
                    "question": case["question"],
                    "generated_claim": claim,
                    "cited_sources": [
                        _source_view(source_by_id[value]) for value in citation_ids
                    ],
                }
            )
        generated_view = [
            {"id": claim["id"], "text": claim["text"]} for claim in generated_claims
        ]
        generated_response = " ".join(claim["text"].strip() for claim in generated_claims)
        for required in case.get("required_claims", []):
            tasks.append(
                {
                    "schema_version": "1.0.0",
                    "task_id": f"{item['id']}:coverage:{required['id']}",
                    "task_type": "required_claim_coverage",
                    "case_id": item["id"],
                    "run_id": run_id,
                    "answer_sha256": digest,
                    "lane": item.get("lane"),
                    "severity": item.get("severity"),
                    "question": case["question"],
                    "generated_response": generated_response,
                    "generated_claims": generated_view,
                    "required_claim": {
                        "id": required["id"],
                        "text": required["text"],
                    },
                }
            )

    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("calibration task IDs are not unique")
    support_count = sum(task["task_type"] == "generated_claim_support" for task in tasks)
    coverage_count = sum(task["task_type"] == "required_claim_coverage" for task in tasks)
    summary = {
        "report_case_count": len(report_ids),
        "answered_case_count": len(answered_ids),
        "support_task_count": support_count,
        "coverage_task_count": coverage_count,
        "total_task_count": len(tasks),
        "development_case_count": sum(
            value == "development" for value in case_partitions.values()
        ),
        "confirmation_case_count": sum(
            value == "confirmation" for value in case_partitions.values()
        ),
    }
    metadata = {
        "run_id": run_id,
        "report_case_ids": report_ids,
        "answered_case_ids": answered_ids,
        "answer_hashes": answer_hashes,
        "case_partitions": case_partitions,
        "summary": summary,
        "task_ids_sha256": canonical_sha256(task_ids),
    }
    return tasks, metadata


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--expected-total", type=int, default=175)
    args = parser.parse_args()

    report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    dataset_value = report.get("dataset", {}).get("path")
    if not isinstance(dataset_value, str) or not dataset_value:
        raise SystemExit("evaluation report is missing dataset.path")
    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if sha256_file(dataset_path) != report.get("dataset", {}).get("sha256"):
        raise SystemExit("dataset hash does not match the frozen evaluation report")

    tasks, metadata = build_queue(report, load_cases(dataset_path))
    if args.expected_total and len(tasks) != args.expected_total:
        raise SystemExit(
            f"expected {args.expected_total} verdict tasks, built {len(tasks)}"
        )
    queue_path = args.out_dir / "queue.jsonl"
    queue_text = "".join(
        json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n"
        for task in tasks
    )
    write_atomic(queue_path, queue_text)
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_eval_report": {
            "path": _relative(args.eval_report),
            "sha256": sha256_file(args.eval_report),
        },
        "dataset": {
            "path": _relative(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "queue": {
            "path": _relative(queue_path),
            "sha256": sha256_file(queue_path),
        },
        "frozen_report_identity": {
            "models": report.get("models"),
            "retrieval": report.get("retrieval"),
            "prompt_hashes": report.get("prompt_hashes"),
            "provenance": report.get("provenance"),
        },
        **metadata,
    }
    manifest_path = args.out_dir / "manifest.json"
    write_atomic(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": _relative(manifest_path), **metadata["summary"]}, indent=2))


if __name__ == "__main__":
    main()
