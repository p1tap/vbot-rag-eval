"""Verify the promoted RAG, reader, and adversarial evidence from raw artifacts.

This gate is deliberately stricter than checking that report files exist:

* the 10,000-case specialist RAG metrics are recomputed from case records;
* the disjoint 1,000-case SQuAD reader confirmation is recomputed separately;
* the rejected adversarial action-policy result is reproduced and must remain
  visibly release-ineligible.

No provider call is made. The script validates committed evidence only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.analyze_action_policy import analyze  # noqa: E402
from scripts.benchmarks.audit_specialist_full import audit as audit_specialist  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    aggregate,
    canonical_sha256,
    load_jsonl,
)

SPECIALIST_REPORT = (
    ROOT
    / "reports"
    / "public-benchmarks"
    / "end-to-end-10000-specialist-promoted.json"
)
SPECIALIST_AUDIT = (
    ROOT
    / "reports"
    / "public-benchmarks"
    / "end-to-end-10000-specialist-promoted-audit.json"
)
SQUAD_REPORT = (
    ROOT
    / "reports"
    / "public-benchmarks"
    / "squad-v2-oracle-context-deberta-v3-large-confirmation-1000.json"
)
SQUAD_COMPARISON = (
    ROOT
    / "reports"
    / "public-benchmarks"
    / "comparisons"
    / "squad-v2-oracle-context-deberta-v3-large-confirmation-1000.json"
)
ADVERSARIAL_REPORT = (
    ROOT / "reports" / "v2" / "vbot-ai-adversarial-12-qwen35-action-policy.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_specialist_rag() -> dict[str, Any]:
    committed = _load(SPECIALIST_AUDIT)
    recomputed = audit_specialist(SPECIALIST_REPORT)
    _require(committed["status"] == "passed", "committed specialist audit is not passing")
    _require(recomputed["status"] == "passed", "recomputed specialist audit failed")
    for field in ("auditor", "audited_report", "checks", "specialist_checks", "recomputed"):
        _require(
            committed[field] == recomputed[field],
            f"specialist audit drifted: {field}",
        )
    strict_macro = recomputed["recomputed"]["strict_macro_joint_correct_rate"]
    _require(strict_macro >= 0.60, "specialist RAG macro gate is below 60%")
    return {
        "case_count": recomputed["recomputed"]["case_count"],
        "strict_macro_joint_correct_rate": strict_macro,
        "zero_fail_closed": recomputed["checks"]["zero_fail_closed"],
        "status": "verified",
    }


def verify_squad_confirmation() -> dict[str, Any]:
    report = _load(SQUAD_REPORT)
    comparison = _load(SQUAD_COMPARISON)
    case_path = ROOT / report["artifacts"]["case_records_path"]
    rows = load_jsonl(case_path)

    _require(report["status"] == "complete", "SQuAD confirmation is incomplete")
    _require(
        report["identity"]["evaluation_lane"] == "oracle_context_extractive_reader"
        and report["identity"]["retrieval_included"] is False,
        "SQuAD reader scope changed or was conflated with retrieval",
    )
    _require(
        len(rows) == report["case_count"] == 1000,
        "SQuAD confirmation must contain exactly 1,000 cases",
    )
    _require(
        sha256_file(case_path) == report["artifacts"]["case_records_sha256"],
        "SQuAD case artifact hash mismatch",
    )
    _require(
        canonical_sha256(rows)
        == report["artifacts"]["case_results_canonical_sha256"],
        "SQuAD canonical case hash mismatch",
    )
    metrics = aggregate(rows)
    _require(metrics == report["metrics"], "SQuAD metrics do not recompute")
    fail_closed = sum(bool(row["fail_closed"]) for row in rows)
    _require(
        fail_closed == report["fail_closed_case_count"] == 0,
        "SQuAD confirmation contains fail-closed cases",
    )
    _require(comparison["decision"] == "ACCEPT", "SQuAD comparison is not accepted")
    _require(
        comparison["identity"]["source_sha256"]
        == report["identity"]["source_sha256"],
        "SQuAD comparison targets a different source dataset",
    )
    _require(
        comparison["identity"]["sample"] == report["case_count"],
        "SQuAD comparison sample count drifted",
    )
    return {
        "case_count": len(rows),
        "exact_match": metrics["squad_v2"]["answer_exact_match"],
        "token_f1": metrics["squad_v2"]["answer_f1"],
        "answerability_accuracy": metrics["squad_v2"]["answerability_accuracy"],
        "retrieval_included": False,
        "status": "verified",
    }


def verify_adversarial_rejection() -> dict[str, Any]:
    report = _load(ADVERSARIAL_REPORT)
    generation_path = ROOT / report["source"]["path"]
    generation = _load(generation_path)
    _require(
        sha256_file(generation_path) == report["source"]["sha256"],
        "adversarial generation hash mismatch",
    )
    guardrails = report["guardrails"]
    recomputed = analyze(
        generation,
        max_false_answer_rate=guardrails["max_false_answer_rate"],
        max_false_refusal_rate=guardrails["max_false_refusal_rate"],
    )
    for field in ("sample", "metrics", "guardrails", "failure_case_ids", "by_lane", "items"):
        _require(
            report[field] == recomputed[field],
            f"adversarial action report drifted: {field}",
        )
    _require(
        report["status"] == "development_policy_rejected"
        and report["release_eligible"] is False
        and report["guardrails"]["passes"] is False,
        "failed adversarial policy was incorrectly promoted",
    )
    non_answer_rows = [
        row for row in recomputed["items"] if row["expected_action"] != "answer"
    ]
    contained = sum(row["delivered_action"] != "answer" for row in non_answer_rows)
    _require(contained == len(non_answer_rows), "adversarial case produced an unsafe answer")
    return {
        "case_count": recomputed["sample"]["case_count"],
        "strict_action_accuracy": recomputed["metrics"]["action_accuracy"],
        "false_answer_count": recomputed["metrics"]["false_answer_count"],
        "non_answer_containment_rate": contained / len(non_answer_rows),
        "status": "rejection_reproduced",
    }


def verify_all() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "status": "verified",
        "specialist_rag_10000": verify_specialist_rag(),
        "squad_reader_confirmation_1000": verify_squad_confirmation(),
        "adversarial_regression_12": verify_adversarial_rejection(),
        "claim_boundary": {
            "publisher_annotations_are_not_local_human_review": True,
            "squad_reader_excludes_retrieval": True,
            "failed_adversarial_policy_remains_unpromoted": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify_all()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        rag = result["specialist_rag_10000"]
        squad = result["squad_reader_confirmation_1000"]
        adversarial = result["adversarial_regression_12"]
        print(
            "portfolio evidence verified: "
            f"RAG={rag['case_count']}@{rag['strict_macro_joint_correct_rate']:.2%}, "
            f"SQuAD={squad['case_count']}@{squad['exact_match']:.2%} EM, "
            f"adversarial={adversarial['case_count']} "
            f"({adversarial['status']})"
        )


if __name__ == "__main__":
    main()
