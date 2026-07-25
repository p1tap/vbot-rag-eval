"""Freeze and verify the canonical V2 review and no-human judge contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "evals" / "v2" / "v2-contract-freeze.json"

FILES = (
    "config.py",
    "evals/schema/case-v2.schema.json",
    "evals/schema/judge-calibration-label.schema.json",
    "evals/v2/dataset-manifest.json",
    "evals/v2/review-freeze-manifest.json",
    "evals/v2/review-inventory.json",
    "evals/v2/coverage-matrix.json",
    "evals/v2/reviewed/vbot-human-reviewed-100.jsonl",
    "evals/v2/release/vbot-ai-reviewed-release-50.jsonl",
    "evals/v2/judge-calibration/manifest.json",
    "evals/v2/judge-calibration/queue.jsonl",
    "evals/v2/judge-calibration/human-labels.jsonl",
    "evals/v2/judge-calibration/ai-adjudication-overrides.jsonl",
    "rag/claim_verifier.py",
    "rag/coverage_judge.py",
    "rag/judge_contract.py",
    "rag/abstention.py",
    "rag/embed.py",
    "rag/llm.py",
    "rag/generate.py",
    "rag/retrieve.py",
    "rag/structured_answer.py",
    "reports/v2/judge-bakeoff-summary.json",
    "reports/v2/ai-release-seed-audit.json",
    "reports/v2/judge-calibration-consensus-audit.json",
    "reports/v2/judge-cascade-simulation-human.json",
    "reports/v2/judge-cascade-simulation.json",
    "reports/v2/judge-bakeoff-decision-human.md",
    "reports/v2/judge-bakeoff-decision.md",
    "scripts/import_completed_reviews.py",
    "scripts/run_judge_bakeoff.py",
    "scripts/audit_calibration_consensus.py",
    "scripts/audit_ai_release_seed.py",
    "scripts/analyze_action_policy.py",
    "scripts/build_ai_release_seed.py",
    "scripts/simulate_judge_cascade.py",
    "scripts/run_operational_judge.py",
    "scripts/run_structured_eval.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_contract_text(path: Path) -> str:
    """Hash the text contract with platform-neutral LF line endings."""
    value = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(value).hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def build() -> dict:
    missing = [relative for relative in FILES if not (ROOT / relative).is_file()]
    if missing:
        raise ValueError(f"V2 contract files are missing: {missing}")
    review = load("evals/v2/review-freeze-manifest.json")
    dataset = load("evals/v2/dataset-manifest.json")
    bakeoff = load("reports/v2/judge-bakeoff-summary.json")
    consensus = load("reports/v2/judge-calibration-consensus-audit.json")
    human_policy = load("reports/v2/judge-cascade-simulation-human.json")
    ai_policy = load("reports/v2/judge-cascade-simulation.json")

    reviewed = review["reviewed_cases"]
    calibration = review["judge_calibration"]
    if sha256_file(ROOT / reviewed["path"]) != reviewed["sha256"]:
        raise ValueError("reviewed-case freeze hash mismatch")
    if sha256_file(ROOT / calibration["path"]) != calibration["sha256"]:
        raise ValueError("human calibration freeze hash mismatch")
    canonical = dataset["partitions"][0]["files"][0]
    if canonical["sha256"] != reviewed["sha256"] or canonical["cases"] != 100:
        raise ValueError("canonical dataset does not point to the 100-case freeze")

    frozen_inputs = bakeoff["frozen_inputs"]
    if frozen_inputs["human_labels_sha256"] != calibration["sha256"]:
        raise ValueError("judge bakeoff targets different human labels")
    included = set(bakeoff["profile_roster"]["included"])
    expected_profiles = {
        "v4-flash-high",
        "gemini-3.5-flash-high",
        "gpt-5.4-high",
    }
    if included != expected_profiles or bakeoff["profile_roster"]["complete"]:
        raise ValueError("judge decision roster drifted")
    if bakeoff["profile_roster"]["omitted"] != ["v4-flash-max"]:
        raise ValueError("incomplete V4 max profile is not declared")
    for profile_id, profile in bakeoff["profiles"].items():
        if sha256_file(ROOT / profile["records_path"]) != profile["records_sha256"]:
            raise ValueError(f"judge checkpoint drifted: {profile_id}")

    bakeoff_sha = sha256_file(ROOT / "reports/v2/judge-bakeoff-summary.json")
    override_sha = sha256_file(
        ROOT / "evals/v2/judge-calibration/ai-adjudication-overrides.jsonl"
    )
    if consensus["inputs"]["bakeoff_sha256"] != bakeoff_sha:
        raise ValueError("consensus audit targets a different bakeoff")
    if consensus["override_artifact"]["sha256"] != override_sha:
        raise ValueError("AI adjudication override hash mismatch")
    for policy, override in ((human_policy, False), (ai_policy, True)):
        if policy["inputs"]["bakeoff_sha256"] != bakeoff_sha:
            raise ValueError("cascade targets a different bakeoff")
        if policy["selection"]["selected_policy_id"] != "v4-primary-failure-cascade":
            raise ValueError("selected no-human cascade drifted")
        if bool(policy["inputs"].get("ai_override_sha256")) != override:
            raise ValueError("cascade reference provenance is ambiguous")
    if ai_policy["inputs"]["ai_override_sha256"] != override_sha:
        raise ValueError("AI-adjusted cascade targets different overrides")

    return {
        "schema_version": "1.0.0",
        "dataset": {
            "version": dataset["dataset_version"],
            "fingerprint_sha256": dataset["fingerprint_sha256"],
            "reviewed_cases": 100,
            "reviewed_cases_sha256": reviewed["sha256"],
            "ai_reviewed_release_cases": 50,
        },
        "human_calibration": {
            "task_count": calibration["task_count"],
            "case_count": calibration["case_count"],
            "sha256": calibration["sha256"],
        },
        "judge_policy": {
            "included_profiles": sorted(included),
            "omitted_incomplete_profiles": bakeoff["profile_roster"]["omitted"],
            "selected_policy_id": ai_policy["selection"]["selected_policy_id"],
            "future_human_queue": False,
            "ai_override_count": consensus["override_artifact"]["count"],
        },
        "file_hash_canonicalization": "utf8_text_lf",
        "files_sha256": {
            relative: sha256_contract_text(ROOT / relative)
            for relative in sorted(FILES)
        },
    }


def render(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    expected = render(build())
    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(expected)
        action = "wrote"
    elif not args.out.is_file() or args.out.read_bytes() != expected:
        raise SystemExit("V2 contract freeze is stale; regenerate with --write")
    else:
        action = "verified"
    print(f"{action} V2 contract freeze: {sha256_file(args.out)}")


if __name__ == "__main__":
    main()
