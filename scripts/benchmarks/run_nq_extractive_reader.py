"""Run a pinned local extractive reader over retrieved NQ documents."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    canonical_sha256,
    dense_work_items,
    write_json,
    write_jsonl,
)

MODEL_ID = "deepset/deberta-v3-large-squad2"
MODEL_REVISION = "ff971b7ff7946434a93edc8271fa4de35eab29f7"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}"
RUNNER_VERSION = "1.0.0"


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("natural_questions", "hotpotqa"),
        default="natural_questions",
    )
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--offset", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-document-chars", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--max-answer-len", type=int, default=30)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-training-report", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "reports"
        / "public-benchmarks"
        / "nq-extractive-deberta-v3-large-offset200-300.json",
    )
    args = parser.parse_args()
    if min(
        args.sample,
        args.top_k,
        args.max_document_chars,
        args.batch_size,
        args.max_seq_len,
        args.doc_stride,
        args.max_answer_len,
    ) < 1 or args.offset < 0:
        raise SystemExit("sample/options must be positive and offset nonnegative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the local extractive reader")

    works = dense_work_items(
        args.benchmark,
        top_k=args.top_k,
        max_document_chars=args.max_document_chars,
        sample_per_benchmark=args.sample,
        sample_offset_per_benchmark=args.offset,
    )
    if len(works) != args.sample:
        raise RuntimeError("retrieval work count mismatch")

    model_path = Path(
        hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    )
    config_path = Path(
        hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
    )
    tokenizer_path = Path(
        hf_hub_download(MODEL_ID, "spm.model", revision=MODEL_REVISION)
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float16,
        use_safetensors=True,
    )
    adapter_artifacts = None
    if args.adapter:
        if not args.adapter_training_report:
            raise SystemExit("--adapter requires --adapter-training-report")
        model = PeftModel.from_pretrained(model, args.adapter)
        adapter_files = sorted(path for path in args.adapter.iterdir() if path.is_file())
        adapter_artifacts = {
            "directory": args.adapter.resolve().relative_to(ROOT).as_posix(),
            "files": {path.name: sha256_file(path) for path in adapter_files},
            "training_report_path": args.adapter_training_report.resolve()
            .relative_to(ROOT)
            .as_posix(),
            "training_report_sha256": sha256_file(args.adapter_training_report),
        }
    model = model.to("cuda")
    model.eval()
    reader = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=0,
    )

    flat_inputs = []
    flat_identity = []
    for work in works:
        for context in work["contexts"]:
            flat_inputs.append(
                {"question": work["query"], "context": context["text"]}
            )
            flat_identity.append(
                {
                    "case_id": work["case_id"],
                    "citation_id": context["citation_id"],
                    "document_id": context["document_id"],
                }
            )
    with torch.inference_mode():
        flat_outputs = reader(
            flat_inputs,
            batch_size=args.batch_size,
            handle_impossible_answer=True,
            max_seq_len=args.max_seq_len,
            doc_stride=args.doc_stride,
            max_answer_len=args.max_answer_len,
        )
    if len(flat_outputs) != len(flat_identity):
        raise RuntimeError("extractive output count mismatch")

    by_case: dict[str, list[dict]] = {work["case_id"]: [] for work in works}
    for identity, output in zip(flat_identity, flat_outputs, strict=True):
        by_case[identity["case_id"]].append(
            {
                "citation_id": identity["citation_id"],
                "document_id": identity["document_id"],
                "answer": str(output.get("answer") or "").strip(),
                "score": float(output.get("score") or 0.0),
                "start": int(output.get("start") or 0),
                "end": int(output.get("end") or 0),
            }
        )
    records = []
    for work in works:
        records.append(
            {
                "schema_version": "1.0.0",
                "case_id": work["case_id"],
                "benchmark_id": args.benchmark,
                "query": work["query"],
                "retrieved_document_ids": work["retrieved_document_ids"],
                "gold": work["gold"],
                "gold_document_ids": sorted(work["gold_document_ids"]),
                "gold_evidence_sets": work["gold_evidence_sets"],
                "candidates": by_case[work["case_id"]],
            }
        )

    records_path = args.out.with_suffix(".records.jsonl")
    write_jsonl(records_path, records)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "profile_id": (
            f"deberta-v3-large-squad2-{args.benchmark}-lora-retrieved-local-fp16"
            if args.adapter
            else "deberta-v3-large-squad2-retrieved-nq-local-fp16"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_url": MODEL_URL,
        "model_license": "CC-BY-4.0",
        "model_artifacts": {
            "model_safetensors_sha256": sha256_file(model_path),
            "config_sha256": sha256_file(config_path),
            "sentencepiece_sha256": sha256_file(tokenizer_path),
        },
        "adapter_artifacts": adapter_artifacts,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "benchmark": args.benchmark,
        "dtype": "float16",
        "sample_per_benchmark": args.sample,
        "sample_offset_per_benchmark": args.offset,
        "case_ids_sha256": canonical_sha256(sorted(by_case)),
        "top_k": args.top_k,
        "max_document_chars": args.max_document_chars,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "doc_stride": args.doc_stride,
        "max_answer_len": args.max_answer_len,
        "handle_impossible_answer": True,
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "raw_candidates_complete",
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "case_count": len(records),
        "candidate_count": sum(len(row["candidates"]) for row in records),
        "operation": {
            "invocation_wall_seconds": time.perf_counter() - started,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
        "artifacts": {
            "records_path": records_path.resolve().relative_to(ROOT).as_posix(),
            "records_sha256": sha256_file(records_path),
            "records_canonical_sha256": canonical_sha256(records),
        },
        "limitations": [
            "This raw artifact contains reader candidates and no selected score threshold.",
            f"The SQuAD-2.0 base reader is tested on retrieved {args.benchmark} documents.",
            "The offset-200 window is development data; policy selection must be confirmed on a disjoint window.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": report["case_count"],
                "candidates": report["candidate_count"],
                "operation": report["operation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
