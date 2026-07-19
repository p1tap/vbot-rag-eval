"""Compare paired SQuAD 2.0 oracle-context reader reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.compare_end_to_end import exact_mcnemar_p  # noqa: E402


TARGET_RATE = 0.85


def load_rows(report: dict) -> dict[str, dict]:
    path = Path(report["artifacts"]["case_records_path"])
    if not path.is_absolute():
        path = ROOT / path
    return {
        row["case_id"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_rows = load_rows(baseline)
    candidate_rows = load_rows(candidate)
    if set(baseline_rows) != set(candidate_rows):
        raise SystemExit("paired comparison requires identical case IDs")
    for key in ("source_sha256", "case_ids_sha256", "sample", "sample_seed"):
        if baseline["identity"][key] != candidate["identity"][key]:
            raise SystemExit(f"comparison identity mismatch: {key}")

    baseline_metrics = baseline["metrics"]["squad_v2"]
    candidate_metrics = candidate["metrics"]["squad_v2"]
    metric_names = (
        "answer_exact_match",
        "answer_f1",
        "answerability_accuracy",
        "citation_precision",
        "citation_recall",
        "joint_correct_rate",
    )
    metric_deltas = {
        name: candidate_metrics[name] - baseline_metrics[name]
        for name in metric_names
    }

    wins = losses = unchanged = 0
    for case_id, base in baseline_rows.items():
        cand = candidate_rows[case_id]
        base_exact = bool(base["answer_exact_match"])
        cand_exact = bool(cand["answer_exact_match"])
        if cand_exact and not base_exact:
            wins += 1
        elif base_exact and not cand_exact:
            losses += 1
        else:
            unchanged += 1

    rejection_reasons = []
    if candidate_metrics["answer_exact_match"] < TARGET_RATE:
        rejection_reasons.append("candidate exact match is below 85%")
    if candidate_metrics["answer_f1"] < TARGET_RATE:
        rejection_reasons.append("candidate F1 is below 85%")
    if candidate["fail_closed_case_count"] != 0:
        rejection_reasons.append("candidate has fail-closed cases")

    report = {
        "schema_version": "1.0.0",
        "decision": "ACCEPT" if not rejection_reasons else "REJECT",
        "scope": "paired_squad_v2_oracle_context_reader",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "identity": {
            "source_sha256": candidate["identity"]["source_sha256"],
            "case_ids_sha256": candidate["identity"]["case_ids_sha256"],
            "sample": candidate["identity"]["sample"],
            "sample_seed": candidate["identity"]["sample_seed"],
            "baseline_profile": baseline["identity"]["profile_id"],
            "candidate_profile": candidate["identity"]["profile_id"],
        },
        "baseline_metrics": {
            name: baseline_metrics[name] for name in metric_names
        },
        "candidate_metrics": {
            name: candidate_metrics[name] for name in metric_names
        },
        "metric_deltas": metric_deltas,
        "paired_exact_match_transitions": {
            "wins": wins,
            "losses": losses,
            "unchanged": unchanged,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(wins, losses),
        },
        "fail_closed": {
            "baseline": baseline["fail_closed_case_count"],
            "candidate": candidate["fail_closed_case_count"],
        },
        "acceptance_rules": {
            "minimum_exact_match": TARGET_RATE,
            "minimum_f1": TARGET_RATE,
            "candidate_fail_closed_count": 0,
        },
        "rejection_reasons": rejection_reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
