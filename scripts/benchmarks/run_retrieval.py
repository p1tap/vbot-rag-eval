"""Run benchmark-scope-aware retrieval baselines on normalized public cases."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import statistics
import sys
from time import perf_counter
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.retrieval_eval import (  # noqa: E402
    CaseRetrievalResult,
    aggregate_results,
    evaluate_case,
)
from rag.retrievers import BM25Retriever, TwoStageFeverBM25Index  # noqa: E402
from scripts.benchmarks.audit_suite import DEFAULT_INPUTS, file_sha256  # noqa: E402


SCOPES = {
    "hotpotqa": "per_query_10_document_distractor_corpus",
    "natural_questions": "per_query_long_answer_candidates",
}
DEFAULT_FEVER_INDEX = (
    ROOT / "artifacts" / "benchmarks" / "fever" / "indexes" / "wikipedia-bm25.sqlite"
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def run_fever(index_path: Path, ks: list[int]) -> dict:
    if not index_path.exists():
        raise ValueError(
            "FEVER requires a global index over the pinned Wikipedia dump; run "
            "python -m scripts.benchmarks.build_fever_index first."
        )
    index = TwoStageFeverBM25Index(index_path)
    results: list[CaseRetrievalResult] = []
    latencies_ms: list[float] = []
    cases = [
        json.loads(line)
        for line in DEFAULT_INPUTS["fever"].read_text(encoding="utf-8").splitlines()
    ]
    relevant_by_case: dict[str, frozenset[str]] = {}
    all_relevant_pages: set[str] = set()
    for case in cases:
        evidence_document_ids = {
            evidence["document_id"] for evidence in case["supporting_evidence"]
        }
        relevant_page_ids = frozenset(
            unicodedata.normalize("NFC", document["title"].replace(" ", "_"))
            for document in case["documents"]
            if document["id"] in evidence_document_ids
        )
        relevant_by_case[case["id"]] = relevant_page_ids
        all_relevant_pages.update(relevant_page_ids)
    found_relevant_pages: set[str] = set()
    with closing(
        sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    ) as connection:
        for start in range(0, len(all_relevant_pages), 500):
            page_ids = sorted(all_relevant_pages)[start : start + 500]
            placeholders = ",".join("?" for _ in page_ids)
            found_relevant_pages.update(
                row[0]
                for row in connection.execute(
                    f"SELECT page_id FROM page_lookup WHERE page_id IN ({placeholders})",
                    page_ids,
                )
            )
    missing_pages = sorted(all_relevant_pages - found_relevant_pages)
    if missing_pages:
        raise ValueError(
            f"FEVER global index is missing {len(missing_pages)} gold pages; "
            f"first missing page: {missing_pages[0]}"
        )

    total_cases = len(cases)
    answerable_cases = sum(bool(pages) for pages in relevant_by_case.values())
    with index:
        for position, case in enumerate(cases, start=1):
            started = perf_counter()
            hits = index.rank(case["query"], max(ks))
            latencies_ms.append((perf_counter() - started) * 1000)
            results.append(
                CaseRetrievalResult(
                    case["id"],
                    relevant_by_case[case["id"]],
                    tuple(hit.document.id for hit in hits),
                )
            )
            if position % 100 == 0:
                print(f"scored FEVER claims: {position}/{total_cases}", flush=True)
    with closing(
        sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    ) as connection:
        metadata = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key, value FROM metadata")
        }
    return {
        "report_version": "1.0.0",
        "benchmark_id": "fever",
        "retriever": {
            "name": index.name,
            "parameters": {
                "tokenizer": metadata["fts_tokenizer"],
                "candidate_limit_per_lane": index.candidate_limit,
                "candidate_generation": (
                    "exact title n-grams plus strict full-text AND; "
                    "leave-one-term relaxation only when fewer than k candidates"
                ),
                "reranker": "in_memory_okapi_bm25",
                "query_terms": "unique non-stopwords",
            },
        },
        "retrieval_scope": "global_pinned_fever_wikipedia",
        "source": {
            "cases_path": DEFAULT_INPUTS["fever"].relative_to(ROOT).as_posix(),
            "cases_sha256": file_sha256(DEFAULT_INPUTS["fever"]),
            "index_path": index_path.relative_to(ROOT).as_posix(),
            "wikipedia_source_sha256": metadata["source_sha256"],
        },
        "cases": {
            "total": total_cases,
            "answerable_scored": answerable_cases,
            "without_gold_evidence": total_cases - answerable_cases,
            "corpus_documents": metadata["page_count"],
        },
        "metrics": aggregate_results(results, ks),
        "latency_ms_per_query": {
            "median": round(statistics.median(latencies_ms), 3),
            "p95": round(percentile(latencies_ms, 0.95), 3),
            "total": round(sum(latencies_ms), 3),
        },
        "limitations": [
            "FEVER Not Enough Information cases have no positive evidence and are excluded from retrieval quality metrics.",
            "End-to-end label prediction, evidence-set completeness, and generation are not measured here.",
        ],
    }


def run(benchmark_id: str, ks: list[int], index_path: Path | None = None) -> dict:
    if benchmark_id == "fever":
        return run_fever(index_path or DEFAULT_FEVER_INDEX, ks)
    input_path = DEFAULT_INPUTS[benchmark_id]
    retriever = BM25Retriever()
    results = []
    latencies_ms: list[float] = []
    total_cases = 0
    answerable_cases = 0
    candidate_documents = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            total_cases += 1
            candidate_documents += len(case["documents"])
            if case["supporting_evidence"]:
                answerable_cases += 1
            started = perf_counter()
            results.append(evaluate_case(case, retriever, max_k=max(ks)))
            latencies_ms.append((perf_counter() - started) * 1000)

    return {
        "report_version": "1.0.0",
        "benchmark_id": benchmark_id,
        "retriever": {
            "name": retriever.name,
            "parameters": {"k1": retriever.k1, "b": retriever.b},
        },
        "retrieval_scope": SCOPES[benchmark_id],
        "source": {
            "path": input_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(input_path),
        },
        "cases": {
            "total": total_cases,
            "answerable_scored": answerable_cases,
            "without_gold_evidence": total_cases - answerable_cases,
            "candidate_documents_ranked": candidate_documents,
        },
        "metrics": aggregate_results(results, ks),
        "latency_ms_per_query": {
            "median": round(statistics.median(latencies_ms), 3),
            "p95": round(percentile(latencies_ms, 0.95), 3),
            "total": round(sum(latencies_ms), 3),
        },
        "limitations": [
            "This is bounded candidate ranking, not open-corpus retrieval.",
            "Cases without gold evidence are excluded from retrieval quality metrics.",
            "End-to-end generation and answer correctness are not measured here.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(DEFAULT_INPUTS),
    )
    parser.add_argument("--k", default="1,2,4,5,10")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    ks = sorted({int(value) for value in args.k.split(",")})
    if not ks or ks[0] < 1:
        parser.error("--k must contain positive integers")
    try:
        report = run(args.benchmark, ks, args.index)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or (
        ROOT
        / "artifacts"
        / "benchmarks"
        / args.benchmark
        / "retrieval"
        / f"{args.benchmark}-bm25.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "retrieval baseline complete: "
        f"benchmark={args.benchmark} cases={report['cases']['total']} "
        f"scope={report['retrieval_scope']}"
    )


if __name__ == "__main__":
    main()
