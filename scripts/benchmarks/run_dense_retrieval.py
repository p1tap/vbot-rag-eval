"""Build pinned E5 candidate vectors and score bounded retrieval tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EMBED_MODEL, EMBED_MODEL_REVISION  # noqa: E402
from rag.embed import embed_passages, embed_queries  # noqa: E402
from rag.retrieval_eval import CaseRetrievalResult, aggregate_results  # noqa: E402
from scripts.benchmarks.audit_suite import DEFAULT_INPUTS, file_sha256  # noqa: E402
from scripts.benchmarks.run_retrieval import SCOPES, percentile  # noqa: E402


SUPPORTED = ("hotpotqa", "natural_questions")


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def count_records(path: Path) -> tuple[int, int]:
    cases = 0
    documents = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            cases += 1
            documents += len(case["documents"])
    return cases, documents


def flush_embeddings(
    texts: list[str],
    target: np.memmap,
    offset: int,
    *,
    query: bool,
) -> int:
    if not texts:
        return offset
    encoder = embed_queries if query else embed_passages
    vectors = encoder(texts, batch_size=8, max_length=512)
    target[offset : offset + len(texts)] = vectors
    target.flush()
    return offset + len(texts)


def build(benchmark_id: str) -> tuple[Path, dict]:
    source = DEFAULT_INPUTS[benchmark_id]
    source_sha256 = file_sha256(source)
    root = (
        ROOT
        / "artifacts"
        / "benchmarks"
        / benchmark_id
        / "indexes"
        / "e5-small-v2-candidates"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    passages_path = root / "passages.npy"
    queries_path = root / "queries.npy"
    progress_path = root / "progress.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "complete"
            and manifest.get("source_sha256") == source_sha256
            and manifest.get("model_revision") == EMBED_MODEL_REVISION
            and passages_path.exists()
            and queries_path.exists()
        ):
            return root, manifest

    case_count, document_count = count_records(source)
    dimension = 384
    progress_identity = {
        "source_sha256": source_sha256,
        "model_revision": EMBED_MODEL_REVISION,
        "case_count": case_count,
        "document_count": document_count,
        "dimension": dimension,
    }
    saved_progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {}
    )
    can_resume = all(
        saved_progress.get(key) == value for key, value in progress_identity.items()
    )
    passage_offset = (
        int(saved_progress.get("passages_completed", 0)) if can_resume else 0
    )
    if passage_offset and passages_path.exists():
        passage_vectors = np.lib.format.open_memmap(passages_path, mode="r+")
        if passage_vectors.shape != (document_count, dimension):
            raise ValueError("dense passage checkpoint shape drifted")
    else:
        passage_offset = 0
        passage_vectors = np.lib.format.open_memmap(
            passages_path,
            mode="w+",
            dtype=np.float32,
            shape=(document_count, dimension),
        )
    progress = {
        **progress_identity,
        "passages_completed": passage_offset,
        "queries_completed": (
            int(saved_progress.get("queries_completed", 0)) if can_resume else 0
        ),
        "status": "building",
    }
    write_json_atomic(progress_path, progress)
    passage_texts: list[str] = []
    seen_documents = 0
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            for document in case["documents"]:
                seen_documents += 1
                if seen_documents <= passage_offset:
                    continue
                passage_texts.append(
                    f"{document['title']}. {' '.join(document['sentences'])}"
                )
                if len(passage_texts) >= 256:
                    passage_offset = flush_embeddings(
                        passage_texts,
                        passage_vectors,
                        passage_offset,
                        query=False,
                    )
                    passage_texts.clear()
                    progress["passages_completed"] = passage_offset
                    write_json_atomic(progress_path, progress)
                    print(
                        f"embedded {benchmark_id} passages: "
                        f"{passage_offset}/{document_count}",
                        flush=True,
                    )
    passage_offset = flush_embeddings(
        passage_texts, passage_vectors, passage_offset, query=False
    )
    progress["passages_completed"] = passage_offset
    write_json_atomic(progress_path, progress)
    if passage_offset != document_count:
        raise ValueError("dense passage count drifted during embedding")

    query_offset = int(progress["queries_completed"])
    if query_offset and queries_path.exists():
        query_vectors = np.lib.format.open_memmap(queries_path, mode="r+")
        if query_vectors.shape != (case_count, dimension):
            raise ValueError("dense query checkpoint shape drifted")
    else:
        query_offset = 0
        query_vectors = np.lib.format.open_memmap(
            queries_path,
            mode="w+",
            dtype=np.float32,
            shape=(case_count, dimension),
        )
    query_texts: list[str] = []
    seen_queries = 0
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            seen_queries += 1
            if seen_queries <= query_offset:
                continue
            query_texts.append(case["query"])
            if len(query_texts) >= 256:
                query_offset = flush_embeddings(
                    query_texts, query_vectors, query_offset, query=True
                )
                query_texts.clear()
                progress["queries_completed"] = query_offset
                write_json_atomic(progress_path, progress)
                print(
                    f"embedded {benchmark_id} queries: {query_offset}/{case_count}",
                    flush=True,
                )
    query_offset = flush_embeddings(query_texts, query_vectors, query_offset, query=True)
    progress["queries_completed"] = query_offset
    write_json_atomic(progress_path, progress)
    if query_offset != case_count:
        raise ValueError("dense query count drifted during embedding")
    del passage_vectors, query_vectors

    manifest = {
        "manifest_version": "1.0.0",
        "status": "complete",
        "benchmark_id": benchmark_id,
        "source_sha256": source_sha256,
        "model": EMBED_MODEL,
        "model_revision": EMBED_MODEL_REVISION,
        "pooling": "attention_mask_mean_then_l2_normalize",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "max_tokens": 512,
        "dimension": dimension,
        "case_count": case_count,
        "document_count": document_count,
        "passages_sha256": file_sha256(passages_path),
        "queries_sha256": file_sha256(queries_path),
    }
    write_json_atomic(manifest_path, manifest)
    progress_path.unlink(missing_ok=True)
    return root, manifest


def score(benchmark_id: str, root: Path, manifest: dict, ks: list[int]) -> dict:
    passages = np.load(root / "passages.npy", mmap_mode="r")
    queries = np.load(root / "queries.npy", mmap_mode="r")
    results: list[CaseRetrievalResult] = []
    latencies_ms: list[float] = []
    passage_offset = 0
    answerable = 0
    with DEFAULT_INPUTS[benchmark_id].open("r", encoding="utf-8") as handle:
        for case_position, line in enumerate(handle):
            case = json.loads(line)
            count = len(case["documents"])
            started = perf_counter()
            scores = passages[passage_offset : passage_offset + count] @ queries[case_position]
            order = sorted(
                range(count),
                key=lambda index: (-float(scores[index]), case["documents"][index]["id"]),
            )[: max(ks)]
            latencies_ms.append((perf_counter() - started) * 1000)
            relevant = frozenset(
                evidence["document_id"] for evidence in case["supporting_evidence"]
            )
            answerable += bool(relevant)
            results.append(
                CaseRetrievalResult(
                    case["id"],
                    relevant,
                    tuple(case["documents"][index]["id"] for index in order),
                )
            )
            passage_offset += count
    if passage_offset != manifest["document_count"]:
        raise ValueError("dense scoring document offset drifted")
    return {
        "report_version": "1.0.0",
        "benchmark_id": benchmark_id,
        "retriever": {
            "name": "e5_small_v2",
            "model": EMBED_MODEL,
            "model_revision": EMBED_MODEL_REVISION,
            "max_tokens": 512,
        },
        "retrieval_scope": SCOPES[benchmark_id],
        "source": {
            "sha256": manifest["source_sha256"],
            "passages_sha256": manifest["passages_sha256"],
            "queries_sha256": manifest["queries_sha256"],
        },
        "cases": {
            "total": manifest["case_count"],
            "answerable_scored": answerable,
            "without_gold_evidence": manifest["case_count"] - answerable,
            "candidate_documents_ranked": manifest["document_count"],
        },
        "metrics": aggregate_results(results, ks),
        "ranking_latency_ms_per_query": {
            "median": round(float(np.median(latencies_ms)), 3),
            "p95": round(percentile(latencies_ms, 0.95), 3),
            "note": "Excludes offline embedding build time.",
        },
        "limitations": [
            "This is bounded candidate ranking, not open-corpus retrieval.",
            "Passages are truncated to the pinned model maximum of 512 tokens.",
            "Cases without gold evidence are excluded from retrieval quality metrics.",
            "End-to-end generation and answer correctness are not measured here.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=SUPPORTED)
    parser.add_argument("--k", default="1,2,4,5,10")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ks = sorted({int(value) for value in args.k.split(",")})
    root, manifest = build(args.benchmark)
    if args.build_only:
        print(f"dense candidate index ready: {root}")
        return
    report = score(args.benchmark, root, manifest, ks)
    output = args.output or (
        ROOT
        / "artifacts"
        / "benchmarks"
        / args.benchmark
        / "retrieval"
        / f"{args.benchmark}-e5-small-v2.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"dense retrieval complete: benchmark={args.benchmark}")


if __name__ == "__main__":
    main()
