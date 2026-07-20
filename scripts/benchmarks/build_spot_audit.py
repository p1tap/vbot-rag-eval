from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from scripts.benchmarks.audit_suite import DEFAULT_INPUTS

ROOT = Path(__file__).resolve().parents[2]

STRATA: list[tuple[str, str, int, Callable[[dict[str, Any]], bool]]] = [
    (
        "hotpotqa",
        "bridge",
        8,
        lambda case: case["metadata"].get("question_type") == "bridge",
    ),
    (
        "hotpotqa",
        "comparison",
        4,
        lambda case: case["metadata"].get("question_type") == "comparison",
    ),
    (
        "natural_questions",
        "answerable",
        10,
        lambda case: case["gold"]["answerability"] == "answerable",
    ),
    (
        "natural_questions",
        "unanswerable",
        10,
        lambda case: case["gold"]["answerability"] == "unanswerable",
    ),
    (
        "fever",
        "supports",
        6,
        lambda case: case["gold"]["label"] == "supports",
    ),
    (
        "fever",
        "refutes",
        5,
        lambda case: case["gold"]["label"] == "refutes",
    ),
    (
        "fever",
        "not_enough_info",
        5,
        lambda case: case["gold"]["label"] == "not_enough_info",
    ),
]


def rank(case_id: str, stratum: str) -> str:
    return hashlib.sha256(f"adapter-audit-v1:{stratum}:{case_id}".encode()).hexdigest()


def audit_view(case: dict[str, Any], stratum: str) -> dict[str, Any]:
    documents = {document["id"]: document for document in case["documents"]}
    evidence = [
        {
            "document_title": documents[item["document_id"]]["title"],
            "sentence_index": item["sentence_index"],
            "text": item["text"],
        }
        for item in case["supporting_evidence"]
    ]
    vote_answerability = Counter(
        vote["answerability"] for vote in case["annotation_votes"]
    )
    return {
        "audit_id": f"public-audit:{case['id']}",
        "benchmark_id": case["benchmark_id"],
        "stratum": stratum,
        "case_id": case["id"],
        "source_record_id": case["provenance"]["source_record_id"],
        "source_record_sha256": case["provenance"]["source_record_sha256"],
        "query": case["query"],
        "gold": case["gold"],
        "supporting_evidence": evidence,
        "annotation_summary": {
            "vote_count": len(case["annotation_votes"]),
            "answerability_votes": dict(sorted(vote_answerability.items())),
            "evidence_sets": [vote["evidence"] for vote in case["annotation_votes"]],
        },
        "review": {
            "status": "pending",
            "note": "",
        },
    }


def build() -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    achieved: dict[str, int] = {}
    for benchmark_id, stratum, target, predicate in STRATA:
        candidates: list[tuple[str, dict[str, Any]]] = []
        with DEFAULT_INPUTS[benchmark_id].open("r", encoding="utf-8") as handle:
            for line in handle:
                case = json.loads(line)
                if predicate(case):
                    candidates.append((rank(case["id"], stratum), case))
        candidates.sort(key=lambda item: (item[0], item[1]["id"]))
        if len(candidates) < target:
            raise ValueError(
                f"audit stratum {benchmark_id}/{stratum} has {len(candidates)} "
                f"candidates; expected at least {target}"
            )
        selected.extend(
            audit_view(case, stratum) for _, case in candidates[:target]
        )
        achieved[f"{benchmark_id}/{stratum}"] = target

    return {
        "audit_version": "1.0.0",
        "purpose": (
            "Verify that public-source labels and evidence survived adapter "
            "normalization; this is not a re-annotation of the public benchmark."
        ),
        "instructions": (
            "Approve when the displayed gold state and evidence are internally "
            "consistent with the query and annotation summary. Mark revision for "
            "a conversion/display problem; reject only when the normalized case "
            "cannot faithfully represent the source annotation."
        ),
        "case_count": len(selected),
        "strata": achieved,
        "cases": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the minimal stratified public-adapter human audit batch."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT.parent
            / "rag-eval-human-review"
            / "public-benchmark-audit"
            / "batch-01.json"
        ),
    )
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"public adapter spot audit built: cases={payload['case_count']}")


if __name__ == "__main__":
    main()
