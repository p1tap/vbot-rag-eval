"""Compare gold-span retention for paired public evidence compressors."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.audit_suite import DEFAULT_INPUTS  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    dense_work_items,
    fever_work_items,
    ranked_ids,
)


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip().casefold()


def evaluation_document_id(case: dict, document_id: str) -> str:
    if case["benchmark_id"] != "fever":
        return document_id
    document = next(item for item in case["documents"] if item["id"] == document_id)
    return unicodedata.normalize("NFC", document["title"].replace(" ", "_"))


def evidence_sets(case: dict) -> list[list[tuple[str, str]]]:
    documents = {document["id"]: document for document in case["documents"]}
    sets = []
    for vote in case["annotation_votes"]:
        evidence = []
        for item in vote["evidence"]:
            document = documents[item["document_id"]]
            evidence.append(
                (
                    evaluation_document_id(case, item["document_id"]),
                    document["sentences"][item["sentence_index"]],
                )
            )
        if evidence and evidence not in sets:
            sets.append(evidence)
    if not sets and case["supporting_evidence"]:
        sets.append(
            [
                (
                    evaluation_document_id(case, item["document_id"]),
                    item["text"],
                )
                for item in case["supporting_evidence"]
            ]
        )
    return sets


def load_cases(sample_per_benchmark: int) -> dict[str, dict]:
    cases = {}
    for source in DEFAULT_INPUTS.values():
        selected = ranked_ids(source, sample_per_benchmark)
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                case = json.loads(line)
                if selected is None or case["id"] in selected:
                    cases[case["id"]] = case
    return cases


def prepare(
    sample_per_benchmark: int, max_document_chars: int, compression_mode: str
) -> list[dict]:
    works = []
    for benchmark_id in DEFAULT_INPUTS:
        if benchmark_id == "fever":
            works.extend(
                fever_work_items(
                    top_k=4,
                    max_document_chars=max_document_chars,
                    sample_per_benchmark=sample_per_benchmark,
                    compression_mode=compression_mode,
                )
            )
        else:
            works.extend(
                dense_work_items(
                    benchmark_id,
                    top_k=4,
                    max_document_chars=max_document_chars,
                    sample_per_benchmark=sample_per_benchmark,
                    retrieval_mode="dense",
                    compression_mode=compression_mode,
                )
            )
    return works


def retention_metrics(works: list[dict], cases: dict[str, dict]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for work in works:
        benchmark_id = work["benchmark_id"]
        bucket = counts.setdefault(
            benchmark_id,
            {
                "answerable_cases": 0,
                "any_gold_span_retained": 0,
                "complete_gold_span_set_retained": 0,
                "complete_gold_document_set_retrieved": 0,
            },
        )
        sets = evidence_sets(cases[work["case_id"]])
        if not sets:
            continue
        bucket["answerable_cases"] += 1
        context_by_id = {
            context["document_id"]: normalized_text(context["text"])
            for context in work["contexts"]
        }
        retained = [
            [
                document_id in context_by_id
                and normalized_text(text) in context_by_id[document_id]
                for document_id, text in evidence_set
            ]
            for evidence_set in sets
        ]
        if any(any(items) for items in retained):
            bucket["any_gold_span_retained"] += 1
        if any(all(items) for items in retained):
            bucket["complete_gold_span_set_retained"] += 1
        if any(
            all(document_id in context_by_id for document_id, _ in evidence_set)
            for evidence_set in sets
        ):
            bucket["complete_gold_document_set_retrieved"] += 1

    metrics = {}
    for benchmark_id, bucket in counts.items():
        denominator = bucket["answerable_cases"]
        metrics[benchmark_id] = {
            **bucket,
            "any_gold_span_retention_rate": bucket["any_gold_span_retained"]
            / denominator,
            "complete_gold_span_set_retention_rate": bucket[
                "complete_gold_span_set_retained"
            ]
            / denominator,
            "complete_gold_document_set_retrieval_rate": bucket[
                "complete_gold_document_set_retrieved"
            ]
            / denominator,
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-benchmark", type=int, default=100)
    parser.add_argument("--candidate-max-document-chars", type=int, default=800)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.sample_per_benchmark)
    configurations = [
        ("production_lexical_6000", "lexical", 6000),
        (
            f"budget_matched_lexical_{args.candidate_max_document_chars}",
            "lexical",
            args.candidate_max_document_chars,
        ),
        (
            f"candidate_semantic_e5_{args.candidate_max_document_chars}",
            "semantic_e5",
            args.candidate_max_document_chars,
        ),
    ]
    results = {}
    for config_id, mode, max_chars in configurations:
        print(f"preparing {config_id}", flush=True)
        works = prepare(args.sample_per_benchmark, max_chars, mode)
        results[config_id] = {
            "compression_mode": mode,
            "max_document_chars": max_chars,
            "metrics": retention_metrics(works, cases),
        }
    report = {
        "schema_version": "1.0.0",
        "scope": "paired_offline_gold_span_retention_diagnostic",
        "identity": {
            "sample_per_benchmark": args.sample_per_benchmark,
            "case_count": len(cases),
            "top_k": 4,
            "retrieval_mode": "dense",
            "source_sha256": {
                benchmark_id: sha256_file(path)
                for benchmark_id, path in DEFAULT_INPUTS.items()
            },
            "runner_source_sha256": sha256_file(
                ROOT / "scripts" / "benchmarks" / "run_end_to_end.py"
            ),
            "compressor_source_sha256": sha256_file(
                ROOT / "rag" / "evidence_compression.py"
            ),
        },
        "configurations": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["configurations"], indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
