"""Fit a provisional scalar abstention threshold on development labels only."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.abstention import (  # noqa: E402
    binary_auc,
    risk_coverage_curve,
    select_operating_point,
)
from rag.provenance import sha256_file  # noqa: E402

DEFAULT_RETRIEVAL = ROOT / "reports" / "v2" / "v2-retrieval-depth.json"
DEFAULT_OUT = ROOT / "reports" / "v2" / "abstention-calibration-dev.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-report", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature", default="top1_score")
    parser.add_argument("--max-false-answer-rate", type=float, default=0.05)
    parser.add_argument("--max-false-refusal-rate", type=float, default=0.15)
    args = parser.parse_args()
    for value in (args.max_false_answer_rate, args.max_false_refusal_rate):
        if not 0 <= value <= 1:
            raise SystemExit("rate guardrails must be between 0 and 1")

    source = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
    rows = source["confidence_features"]
    curve = risk_coverage_curve(rows, args.feature)
    selected = select_operating_point(
        curve,
        max_false_answer_rate=args.max_false_answer_rate,
        max_false_refusal_rate=args.max_false_refusal_rate,
    )
    source_approved = source.get("label_status") == "human_approved"
    ignored = {"id", "lane", "answerability"}
    numeric_features = sorted(
        key
        for key, value in rows[0].items()
        if key not in ignored and isinstance(value, (int, float))
    )
    feature_diagnostics = {}
    for feature in numeric_features:
        feature_curve = risk_coverage_curve(rows, feature)
        feature_diagnostics[feature] = {
            "answerability_auc": binary_auc(rows, feature),
            "selected_operating_point": select_operating_point(
                feature_curve,
                max_false_answer_rate=args.max_false_answer_rate,
                max_false_refusal_rate=args.max_false_refusal_rate,
            ),
        }
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen" if source_approved and selected else "provisional",
        "release_eligible": bool(source_approved and selected),
        "freeze_blockers": [
            blocker
            for blocker, blocked in (
                ("source labels are not human-approved", not source_approved),
                ("no operating point satisfies both guardrails", selected is None),
            )
            if blocked
        ],
        "source": {
            "path": str(args.retrieval_report),
            "sha256": sha256_file(args.retrieval_report),
            "dataset": source["dataset"],
            "feature_scope": source["confidence_feature_scope"],
        },
        "policy": {
            "kind": "answer_if_feature_greater_than_or_equal_to_threshold",
            "feature": args.feature,
            "answerability_auc": binary_auc(rows, args.feature),
            "guardrails": {
                "max_false_answer_rate": args.max_false_answer_rate,
                "max_false_refusal_rate": args.max_false_refusal_rate,
            },
            "selected_operating_point": selected,
        },
        "scalar_feature_diagnostics": feature_diagnostics,
        "sample": {
            "cases": len(rows),
            "answerable": sum(row["answerability"] == "answerable" for row in rows),
            "unanswerable": sum(row["answerability"] == "unanswerable" for row in rows),
        },
        "risk_coverage_curve": curve,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "freeze_blockers", "policy")}, indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
