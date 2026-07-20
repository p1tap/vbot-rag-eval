"""Compare final human claim labels with a structured-eval judge run."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.judge_agreement import compare_judgments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-report", type=Path, required=True)
    parser.add_argument("--human-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.eval_report.read_text(encoding="utf-8"))
    labels = [
        json.loads(line)
        for line in args.human_labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final_labels = [row for row in labels if row["status"] in {"final", "adjudicated"}]
    schema = json.loads(
        (ROOT / "evals" / "schema" / "judge-calibration-label.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for row in final_labels:
        errors = sorted(validator.iter_errors(row), key=lambda item: list(item.path))
        if errors:
            raise SystemExit(f"invalid human label {row.get('case_id')}: {errors[0].message}")
        for field in ("claim_labels", "required_claim_labels"):
            ids = [label["claim_id"] for label in row[field]]
            if len(ids) != len(set(ids)):
                raise SystemExit(f"duplicate claim IDs in {row['case_id']} {field}")
    case_ids = [row["case_id"] for row in final_labels]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("human labels contain multiple final/adjudicated records for a case")
    eval_run_id = evaluation.get("provenance", {}).get("run_id")
    wrong_runs = sorted(row["case_id"] for row in final_labels if row["run_id"] != eval_run_id)
    if wrong_runs:
        raise SystemExit(f"human labels target a different eval run: {wrong_runs}")
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_run_id": eval_run_id,
        "eval_report": str(args.eval_report),
        "human_labels": str(args.human_labels),
        "draft_labels_excluded": len(labels) - len(final_labels),
        "agreement": compare_judgments(final_labels, evaluation["items"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["agreement"], indent=2))


if __name__ == "__main__":
    main()
