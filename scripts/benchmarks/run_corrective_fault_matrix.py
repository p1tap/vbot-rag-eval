"""Inject retrieval faults and verify the corrective path fails closed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.retrievers import (  # noqa: E402
    CorrectiveRetriever,
    MinScoreConfidenceEvaluator,
    RetrievalDocument,
    RetrievalHit,
)


DOCUMENTS = [
    RetrievalDocument("safe", "Safe", "approved evidence"),
    RetrievalDocument("other", "Other", "irrelevant evidence"),
]


class StaticRetriever:
    name = "static"

    def __init__(self, hits):
        self.hits = hits

    def rank(self, query, documents, k):
        return self.hits[:k]


class RaisingRetriever:
    name = "raising"

    def __init__(self, error):
        self.error = error

    def rank(self, query, documents, k):
        raise self.error


def hits(top=0.9, second=0.2):
    return [
        RetrievalHit(DOCUMENTS[0], top, 1),
        RetrievalHit(DOCUMENTS[1], second, 2),
    ]


def controller(primary, fallback, rewriter=None):
    gate = MinScoreConfidenceEvaluator(
        min_hits=1,
        min_top_score=0.7,
        min_margin=0.1,
    )
    return CorrectiveRetriever(
        primary=primary,
        fallback=fallback,
        primary_evaluator=gate,
        fallback_evaluator=gate,
        query_rewriter=rewriter,
    )


def run_matrix() -> list[dict]:
    scenarios = [
        (
            "primary_success",
            controller(StaticRetriever(hits()), StaticRetriever(hits())),
            "primary",
        ),
        (
            "low_confidence_then_fallback",
            controller(StaticRetriever(hits(0.4, 0.3)), StaticRetriever(hits())),
            "fallback",
        ),
        (
            "primary_timeout_then_fallback",
            controller(
                RaisingRetriever(TimeoutError("injected")),
                StaticRetriever(hits()),
            ),
            "fallback",
        ),
        (
            "both_low_confidence",
            controller(
                StaticRetriever(hits(0.4, 0.3)),
                StaticRetriever(hits(0.5, 0.45)),
            ),
            "abstain",
        ),
        (
            "fallback_timeout",
            controller(
                StaticRetriever(hits(0.4, 0.3)),
                RaisingRetriever(TimeoutError("injected")),
            ),
            "abstain",
        ),
        (
            "invalid_query_rewrite",
            controller(
                StaticRetriever(hits(0.4, 0.3)),
                StaticRetriever(hits()),
                rewriter=lambda _: "",
            ),
            "abstain",
        ),
        (
            "non_finite_primary_score",
            controller(
                StaticRetriever(hits(float("nan"), 0.2)),
                StaticRetriever(hits()),
            ),
            "fallback",
        ),
        (
            "duplicate_fallback_documents",
            controller(
                StaticRetriever(hits(0.4, 0.3)),
                StaticRetriever(
                    [
                        RetrievalHit(DOCUMENTS[0], 0.9, 1),
                        RetrievalHit(DOCUMENTS[0], 0.8, 2),
                    ]
                ),
            ),
            "abstain",
        ),
    ]
    rows = []
    for scenario_id, subject, expected in scenarios:
        result = subject.retrieve("approved evidence", DOCUMENTS, 2)
        rows.append(
            {
                "id": scenario_id,
                "expected_status": expected,
                "observed_status": result.status,
                "passed": result.status == expected,
                "returned_document_ids": [hit.document.id for hit in result.hits],
                "attempts": [
                    {
                        "stage": attempt.stage,
                        "outcome": attempt.outcome,
                        "reason": attempt.reason,
                        "error_type": attempt.error_type,
                    }
                    for attempt in result.attempts
                ],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / "validation"
        / "corrective-retrieval-fault-matrix.json",
    )
    args = parser.parse_args()
    rows = run_matrix()
    passed = sum(row["passed"] for row in rows)
    report = {
        "report_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed == len(rows) else "failed",
        "scope": "deterministic_corrective_control_invariants",
        "summary": {
            "scenarios": len(rows),
            "passed": passed,
            "failed": len(rows) - passed,
        },
        "policy": {
            "attempts": ["primary", "one_approved_corpus_fallback"],
            "external_web_fallback": False,
            "terminal_failure": "abstain_with_empty_hits",
        },
        "scenarios": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"corrective fault matrix: {passed}/{len(rows)} passed")
    print(f"report -> {args.output}")
    if passed != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
