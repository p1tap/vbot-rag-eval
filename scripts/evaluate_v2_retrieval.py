"""Evaluate Vbot V2 heading retrieval and export abstention features.

Candidate labels are unreviewed by default, so the report is diagnostic only.
No threshold or retriever is promoted by this script.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag.embed import embed_queries  # noqa: E402
from rag.provenance import sha256_file  # noqa: E402
from rag.retrieve import load_index  # noqa: E402
from scripts.run_structured_eval import evidence_heading, load_cases  # noqa: E402

DEFAULT_CASES = ROOT / "evals" / "v2" / "candidates" / "phase1-new-candidates.json"
DEFAULT_OUT = ROOT / "reports" / "v2" / "v2-retrieval-depth.json"


def rank_headings(scores: np.ndarray, chunks: list[dict], max_k: int, *, deduplicate: bool) -> list[dict]:
    ranked = []
    seen = set()
    for chunk_index in np.argsort(-scores):
        chunk = chunks[int(chunk_index)]
        heading = chunk["heading_key"]
        if deduplicate and heading in seen:
            continue
        seen.add(heading)
        ranked.append(
            {
                "heading_key": heading,
                "chunk_id": chunk["id"],
                "score": float(scores[chunk_index]),
            }
        )
        if len(ranked) == max_k:
            break
    return ranked


def score_case(case: dict, ranked: list[dict], k: int) -> dict:
    retrieved = [row["heading_key"] for row in ranked[:k]]
    retrieved_set = set(retrieved)
    gold = {evidence_heading(item) for item in case.get("acceptable_evidence", [])}
    claim_sets = [
        {evidence_heading(item) for item in claim["evidence_ids"]}
        for claim in case.get("required_claims", [])
    ]
    overlap = gold & retrieved_set
    first = next(
        (position for position, heading in enumerate(retrieved, 1) if heading in gold),
        None,
    )
    covered_claims = sum(bool(required) and required <= retrieved_set for required in claim_sets)
    return {
        "any_gold": bool(overlap),
        "complete_gold": bool(gold) and gold <= retrieved_set,
        "gold_heading_recall": len(overlap) / len(gold) if gold else None,
        "required_claim_coverage": covered_claims / len(claim_sets) if claim_sets else None,
        "complete_required_claims": bool(claim_sets) and covered_claims == len(claim_sets),
        "reciprocal_rank": 1.0 / first if first else 0.0,
    }


def aggregate(cases: list[dict], rankings: dict[str, list[dict]], ks: list[int]) -> dict:
    answerable = [case for case in cases if case["answerability"] == "answerable"]
    result = {}
    for k in ks:
        rows = [score_case(case, rankings[case["id"]], k) for case in answerable]
        result[str(k)] = {
            "answerable_cases": len(rows),
            "case_recall_any": _mean(row["any_gold"] for row in rows),
            "case_recall_all_gold": _mean(row["complete_gold"] for row in rows),
            "gold_heading_recall": _mean(row["gold_heading_recall"] for row in rows),
            "required_claim_coverage": _mean(row["required_claim_coverage"] for row in rows),
            "complete_required_claim_recall": _mean(
                row["complete_required_claims"] for row in rows
            ),
            "mrr": _mean(row["reciprocal_rank"] for row in rows),
        }
    return result


def _mean(values) -> float | None:
    rows = [float(value) for value in values if value is not None]
    return round(sum(rows) / len(rows), 6) if rows else None


def confidence_features(ranked: list[dict], raw_ranked: list[dict], k: int = 4) -> dict:
    selected = [row["score"] for row in ranked[:k]]
    top1 = selected[0]
    top2 = selected[1] if len(selected) > 1 else selected[0]
    mean = sum(selected) / len(selected)
    variance = sum((score - mean) ** 2 for score in selected) / len(selected)
    distinct_needed = next(
        (
            index
            for index in range(1, len(raw_ranked) + 1)
            if len({row["heading_key"] for row in raw_ranked[:index]}) >= min(k, len(ranked))
        ),
        len(raw_ranked),
    )
    return {
        "top1_score": top1,
        "top1_top2_margin": top1 - top2,
        "topk_mean_score": mean,
        "topk_min_score": min(selected),
        "topk_score_std": math.sqrt(variance),
        "raw_chunks_needed_for_distinct_topk": distinct_needed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ks", default="1,2,4,6,8,12")
    args = parser.parse_args()
    ks = sorted({int(item) for item in args.ks.split(",") if item.strip()})
    if not ks or min(ks) < 1:
        raise SystemExit("--ks must contain positive integers")

    cases = load_cases(args.cases)
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case IDs")
    vectors, chunks = load_index()
    started = time.perf_counter()
    queries = embed_queries([case["question"] for case in cases])
    raw_rankings, unique_rankings = {}, {}
    feature_rows = []
    raw_depth = max(len(chunks), max(ks))
    for case, query in zip(cases, queries):
        scores = vectors @ query
        raw = rank_headings(scores, chunks, raw_depth, deduplicate=False)
        unique = rank_headings(scores, chunks, max(ks), deduplicate=True)
        raw_rankings[case["id"]] = raw
        unique_rankings[case["id"]] = unique
        feature_rows.append(
            {
                "id": case["id"],
                "lane": case["lane"],
                "answerability": case["answerability"],
                **confidence_features(unique, raw),
            }
        )

    source_payload = (
        json.loads(args.cases.read_text(encoding="utf-8"))
        if args.cases.suffix == ".json"
        else None
    )
    locally_approved = sum(
        case.get("review", {}).get("status") == "approved" for case in cases
    )
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_eligible": locally_approved == len(cases) and bool(cases),
        "label_status": "human_approved" if locally_approved == len(cases) else "unreviewed_proposals",
        "dataset": {
            "path": args.cases.relative_to(ROOT).as_posix()
            if args.cases.is_relative_to(ROOT)
            else str(args.cases),
            "sha256": sha256_file(args.cases),
            "dataset_fingerprint_sha256": source_payload.get("dataset_fingerprint_sha256")
            if isinstance(source_payload, dict)
            else None,
            "case_count": len(cases),
            "locally_approved_cases": locally_approved,
        },
        "index": {
            "embedding_model": config.EMBED_MODEL,
            "embedding_revision": config.EMBED_MODEL_REVISION,
            "chunks_sha256": sha256_file(ROOT / "index" / "chunks.jsonl"),
            "embeddings_sha256": sha256_file(ROOT / "index" / "embeddings.npy"),
            "chunk_count": len(chunks),
        },
        "ks": ks,
        "variants": {
            "raw_chunk_rank": aggregate(cases, raw_rankings, ks),
            "unique_heading_rank": aggregate(cases, unique_rankings, ks),
        },
        "confidence_feature_scope": "unique_heading_rank_at_k4",
        "confidence_features": feature_rows,
        "runtime": {"elapsed_seconds": round(time.perf_counter() - started, 3)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["variants"], indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
