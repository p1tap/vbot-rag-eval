"""Compare paired public end-to-end pilot runs with fixed promotion rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FLOAT_TOLERANCE = 1e-12


def load_rows(report: dict, report_path: Path) -> dict[str, dict]:
    path = Path(report["artifacts"]["case_records_path"])
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[2]
        path = root / path
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {row["case_id"]: row for row in rows}


def exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_rows = load_rows(baseline, args.baseline)
    candidate_rows = load_rows(candidate, args.candidate)
    if set(baseline_rows) != set(candidate_rows):
        raise SystemExit("paired comparison requires identical case IDs")
    for key in ("source_sha256", "sample_per_benchmark"):
        if baseline["identity"][key] != candidate["identity"][key]:
            raise SystemExit(f"comparison identity mismatch: {key}")

    benchmarks = sorted(key for key in baseline["metrics"] if key != "macro")
    deltas = {}
    regressions = []
    for benchmark_id in benchmarks:
        base = baseline["metrics"][benchmark_id]
        cand = candidate["metrics"][benchmark_id]
        deltas[benchmark_id] = {
            key: cand.get(key) - base.get(key)
            for key in sorted(set(base) & set(cand))
            if isinstance(base.get(key), (int, float))
            and isinstance(cand.get(key), (int, float))
            and key != "case_count"
        }
        if deltas[benchmark_id]["joint_correct_rate"] < -0.02 - FLOAT_TOLERANCE:
            regressions.append(
                f"{benchmark_id} joint correctness regressed by more than 0.02"
            )
        citation_delta = deltas[benchmark_id].get("citation_precision")
        if (
            citation_delta is not None
            and citation_delta < -0.03 - FLOAT_TOLERANCE
        ):
            regressions.append(
                f"{benchmark_id} citation precision regressed by more than 0.03"
            )

    wins = losses = unchanged = 0
    by_benchmark = {}
    for benchmark_id in benchmarks:
        local_wins = local_losses = local_unchanged = 0
        for case_id, base in baseline_rows.items():
            if base["benchmark_id"] != benchmark_id:
                continue
            cand = candidate_rows[case_id]
            left = bool(base["joint_correct"])
            right = bool(cand["joint_correct"])
            if right and not left:
                local_wins += 1
            elif left and not right:
                local_losses += 1
            else:
                local_unchanged += 1
        wins += local_wins
        losses += local_losses
        unchanged += local_unchanged
        by_benchmark[benchmark_id] = {
            "wins": local_wins,
            "losses": local_losses,
            "unchanged": local_unchanged,
        }

    baseline_failures = baseline["fail_closed_case_count"]
    candidate_failures = candidate["fail_closed_case_count"]
    if candidate_failures > baseline_failures:
        regressions.append("fail-closed case count increased")
    macro_delta = (
        candidate["metrics"]["macro"]["joint_correct_rate"]
        - baseline["metrics"]["macro"]["joint_correct_rate"]
    )
    if macro_delta < 0.01 - FLOAT_TOLERANCE:
        regressions.append("macro joint correctness did not improve by at least 0.01")
    decision = "PROMOTE" if not regressions else "REJECT"
    report = {
        "schema_version": "1.0.0",
        "decision": decision,
        "scope": "paired_generator_or_retrieval_pilot_not_release_promotion",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "identity": {
            "baseline_profile": baseline["identity"]["profile_id"],
            "candidate_profile": candidate["identity"]["profile_id"],
            "baseline_top_k": baseline["identity"].get("top_k"),
            "candidate_top_k": candidate["identity"].get("top_k"),
            "baseline_top_k_by_benchmark": baseline["identity"].get(
                "top_k_by_benchmark"
            ),
            "candidate_top_k_by_benchmark": candidate["identity"].get(
                "top_k_by_benchmark"
            ),
            "baseline_retrieval": baseline["identity"].get("retrieval"),
            "candidate_retrieval": candidate["identity"].get("retrieval"),
            "baseline_compression": baseline["identity"].get("compression"),
            "candidate_compression": candidate["identity"].get("compression"),
            "baseline_prompt_policy": baseline["identity"].get(
                "prompt_policy", "standard"
            ),
            "candidate_prompt_policy": candidate["identity"].get(
                "prompt_policy", "standard"
            ),
            "case_count": len(baseline_rows),
        },
        "macro_joint_correct_delta": macro_delta,
        "metric_deltas": deltas,
        "paired_joint_transitions": {
            "wins": wins,
            "losses": losses,
            "unchanged": unchanged,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(wins, losses),
            "by_benchmark": by_benchmark,
        },
        "fail_closed": {
            "baseline": baseline_failures,
            "candidate": candidate_failures,
        },
        "rejection_reasons": regressions,
        "promotion_rules": {
            "minimum_macro_joint_correct_improvement": 0.01,
            "maximum_per_benchmark_joint_regression": 0.02,
            "maximum_per_benchmark_citation_precision_regression": 0.03,
            "fail_closed_count_must_not_increase": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
