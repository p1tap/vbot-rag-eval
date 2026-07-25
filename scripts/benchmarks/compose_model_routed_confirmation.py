"""Compose and evaluate a predeclared benchmark-to-model routing policy."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    aggregate,
    canonical_sha256,
    load_jsonl,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.0.0"


def exact_paired_p_value(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, position)
        for position in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def paired(left: list[dict], right: list[dict]) -> dict:
    left_by_id = {row["case_id"]: row for row in left}
    right_by_id = {row["case_id"]: row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise SystemExit("paired case universes differ")
    left_only = sum(
        left_by_id[case_id]["joint_correct"]
        and not right_by_id[case_id]["joint_correct"]
        for case_id in left_by_id
    )
    right_only = sum(
        right_by_id[case_id]["joint_correct"]
        and not left_by_id[case_id]["joint_correct"]
        for case_id in left_by_id
    )
    return {
        "case_count": len(left_by_id),
        "left_only_win_count": left_only,
        "right_only_win_count": right_only,
        "net_win_count": left_only - right_only,
        "exact_paired_two_sided_p_value": exact_paired_p_value(
            left_only, right_only
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--deepseek", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    glm_report = json.loads(args.glm.read_text(encoding="utf-8"))
    deepseek_report = json.loads(args.deepseek.read_text(encoding="utf-8"))
    development_report = json.loads(args.development.read_text(encoding="utf-8"))
    if glm_report["identity"]["source_sha256"] != deepseek_report["identity"][
        "source_sha256"
    ]:
        raise SystemExit("GLM and DeepSeek source datasets differ")
    if glm_report["identity"].get("sample_offset_per_benchmark") != 100:
        raise SystemExit("GLM report is not the predeclared offset-100 confirmation")

    glm_rows = load_jsonl(ROOT / glm_report["artifacts"]["case_records_path"])
    deepseek_all = {
        row["case_id"]: row
        for row in load_jsonl(ROOT / deepseek_report["artifacts"]["case_records_path"])
    }
    development_ids = {
        row["case_id"]
        for row in load_jsonl(
            ROOT / development_report["artifacts"]["case_records_path"]
        )
    }
    confirmation_ids = {row["case_id"] for row in glm_rows}
    if confirmation_ids & development_ids:
        raise SystemExit("confirmation cases overlap the development pilot")
    if not confirmation_ids <= set(deepseek_all):
        raise SystemExit("promoted DeepSeek report does not cover every confirmation case")
    deepseek_rows = [deepseek_all[row["case_id"]] for row in glm_rows]
    for glm_row, deepseek_row in zip(glm_rows, deepseek_rows):
        if (
            glm_row["benchmark_id"] != deepseek_row["benchmark_id"]
            or glm_row["gold"] != deepseek_row["gold"]
        ):
            raise SystemExit(f"case identity/gold mismatch: {glm_row['case_id']}")

    routing = {
        "hotpotqa": "glm",
        "natural_questions": "deepseek",
        "fever": "deepseek",
    }
    hybrid_rows = [
        dict(glm_row if routing[glm_row["benchmark_id"]] == "glm" else deepseek_row)
        for glm_row, deepseek_row in zip(glm_rows, deepseek_rows)
    ]
    baseline_metrics = aggregate(deepseek_rows)
    hybrid_metrics = aggregate(hybrid_rows)
    overall_pair = paired(hybrid_rows, deepseek_rows)
    by_benchmark = {}
    for benchmark_id in sorted(routing):
        hybrid_lane = [
            row for row in hybrid_rows if row["benchmark_id"] == benchmark_id
        ]
        baseline_lane = [
            row for row in deepseek_rows if row["benchmark_id"] == benchmark_id
        ]
        by_benchmark[benchmark_id] = paired(hybrid_lane, baseline_lane)

    macro_delta = (
        hybrid_metrics["macro"]["joint_correct_rate"]
        - baseline_metrics["macro"]["joint_correct_rate"]
    )
    confirmation_significant = (
        by_benchmark["hotpotqa"]["exact_paired_two_sided_p_value"] <= 0.05
    )
    promotion_ready = (
        macro_delta > 0
        and not any(row["fail_closed"] for row in hybrid_rows)
        and confirmation_significant
    )

    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(case_path, hybrid_rows)
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "confirmation_complete",
        "promotion_ready": promotion_ready,
        "promotion_decision": (
            "promote_model_routed_policy"
            if promotion_ready
            else "retain_existing_promoted_pipeline"
        ),
        "identity": {
            "runner_version": RUNNER_VERSION,
            "runner_source_sha256": sha256_file(Path(__file__)),
            "selection_stage": "predeclared_after_offset0_development_pilot",
            "confirmation_sample_offset_per_benchmark": 100,
            "confirmation_sample_per_benchmark": 100,
            "routing": routing,
            "case_ids_sha256": canonical_sha256(sorted(confirmation_ids)),
            "development_case_ids_sha256": canonical_sha256(
                sorted(development_ids)
            ),
            "development_overlap_count": 0,
        },
        "inputs": {
            "glm_report": {
                "path": args.glm.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.glm),
            },
            "deepseek_report": {
                "path": args.deepseek.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.deepseek),
            },
            "development_report": {
                "path": args.development.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.development),
            },
        },
        "case_count": len(hybrid_rows),
        "fail_closed_case_count": sum(row["fail_closed"] for row in hybrid_rows),
        "baseline_metrics": baseline_metrics,
        "glm_all_lane_metrics": aggregate(glm_rows),
        "hybrid_metrics": hybrid_metrics,
        "macro_joint_correct_delta": macro_delta,
        "paired_comparison": {
            "overall": overall_pair,
            "by_benchmark": by_benchmark,
        },
        "promotion_gate": {
            "positive_macro_delta": macro_delta > 0,
            "zero_fail_closed": not any(row["fail_closed"] for row in hybrid_rows),
            "hotpot_confirmation_exact_p_at_most_0_05": confirmation_significant,
        },
        "artifacts": {
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(hybrid_rows),
        },
        "limitations": [
            "The routing rule was selected on the offset-0 development pilot and evaluated once on this disjoint offset-100 confirmation window.",
            "The Hotpot direction reproduced, but the confirmation effect is not statistically decisive at the predeclared 0.05 threshold.",
            "The existing 10,000-case promoted report remains the release claim until a routed policy passes confirmation and a full audited rerun.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "baseline_macro": baseline_metrics["macro"]["joint_correct_rate"],
                "hybrid_macro": hybrid_metrics["macro"]["joint_correct_rate"],
                "delta": macro_delta,
                "hotpot_pair": by_benchmark["hotpotqa"],
                "promotion_ready": promotion_ready,
                "promotion_decision": report["promotion_decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
