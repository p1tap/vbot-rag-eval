"""Render a compact, provenance-aware Markdown summary for CI and PRs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value) -> str:
    return "n/a" if value is None else f"{100 * float(value):.2f}%"


def render(
    gate_decision: str,
    v1: dict,
    dataset: dict,
    public: dict,
    cascade: dict,
) -> str:
    v1_summary = v1.get("summary", v1)
    expected_cases = sum(
        int(partition.get("expected_cases", 0))
        for partition in dataset.get("partitions", [])
    )
    public_e2e = public.get("end_to_end_evaluation") or {}
    public_metrics = public_e2e.get("metrics") or {}
    macro = public_metrics.get("macro") or {}
    selected_id = (cascade.get("selection") or {}).get("selected_policy_id")
    selected_policy = (cascade.get("policies") or {}).get(selected_id, {})
    cascade_metrics = (
        (selected_policy.get("metrics") or {}).get("overall")
        or cascade.get("metrics")
        or {}
    )
    lines = [
        "## RAG evidence gate",
        "",
        f"**Decision:** {gate_decision}",
        "",
        "| Evidence | Result |",
        "|---|---:|",
        f"| V1 retrieval recall@k | {percent(v1_summary.get('recall_at_k'))} |",
        f"| V1 correctness | {percent(v1_summary.get('correctness'))} |",
        f"| V2 dataset | {dataset.get('dataset_version')} · {expected_cases} cases |",
        f"| Public suite integrity | {public.get('status')} · {public.get('public_case_count', 0)} cases |",
        f"| Public end-to-end macro joint | {percent(macro.get('joint_correct_rate'))} |",
        f"| Human-reference judge agreement | {percent(cascade_metrics.get('exact_agreement') or cascade_metrics.get('overall_agreement'))} |",
        "",
        "Provenance: the public 10K inherits publisher human annotations; it is not locally human-reviewed. "
        "The Vbot domain seed contains 100 locally owner-reviewed cases. PROMOTE/REJECT examples are hash-locked in `docs/promotion-evidence.json`.",
    ]
    blockers = public.get("resume_claim_blockers") or []
    if blockers:
        lines.extend(["", "Public claim blockers: " + "; ".join(blockers)])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-decision", default="NOT_RUN")
    parser.add_argument("--v1", type=Path, default=ROOT / "eval-report.json")
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "evals" / "v2" / "dataset-manifest.json"
    )
    parser.add_argument(
        "--public",
        type=Path,
        default=ROOT / "reports" / "public-benchmarks" / "public-suite-10000.json",
    )
    parser.add_argument(
        "--cascade",
        type=Path,
        default=ROOT / "reports" / "v2" / "judge-cascade-simulation-human.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = render(
        args.gate_decision,
        load(args.v1),
        load(args.dataset),
        load(args.public),
        load(args.cascade),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
