"""Run a pinned local extractive-QA model on the SQuAD 2.0 oracle lane."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    SENTINEL,
    aggregate,
    canonical_sha256,
    case_results,
    write_json,
    write_jsonl,
)
from scripts.benchmarks.run_squad_oracle import (  # noqa: E402
    DEFAULT_INPUT,
    SAMPLE_SEED,
    load_cases,
    work_item,
)

MODEL_ID = "deepset/deberta-v3-large-squad2"
MODEL_REVISION = "ff971b7ff7946434a93edc8271fa4de35eab29f7"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}"
RUNNER_VERSION = "1.0.0"


def prediction_from_output(output: dict) -> tuple[str, list[str]]:
    answer = str(output.get("answer", "")).strip()
    if not answer:
        return SENTINEL, []
    return answer, ["D1"]


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "reports"
        / "public-benchmarks"
        / "squad-v2-oracle-context-deberta-v3-large.json",
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-context-chars", type=int, default=16000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--max-answer-len", type=int, default=30)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.max_context_chars,
        args.max_seq_len,
        args.doc_stride,
        args.max_answer_len,
    ) < 1 or args.sample < 0:
        raise SystemExit("sample must be nonnegative and other numeric options positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the promoted local extractive reader")

    cases = load_cases(args.input, args.sample)
    works = [work_item(case, args.max_context_chars) for case in cases]
    model_path = Path(
        hf_hub_download(
            MODEL_ID,
            "model.safetensors",
            revision=MODEL_REVISION,
        )
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
    ).to("cuda")
    model.eval()
    reader = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=0,
    )
    inputs = [
        {
            "question": work["query"],
            "context": work["contexts"][0]["text"],
        }
        for work in works
    ]
    with torch.inference_mode():
        outputs = reader(
            inputs,
            batch_size=args.batch_size,
            handle_impossible_answer=True,
            max_seq_len=args.max_seq_len,
            doc_stride=args.doc_stride,
            max_answer_len=args.max_answer_len,
        )
    if len(outputs) != len(works):
        raise RuntimeError("extractive reader output count mismatch")

    batches = []
    for start in range(0, len(works), args.batch_size):
        batch_works = works[start : start + args.batch_size]
        batch_outputs = outputs[start : start + args.batch_size]
        results = []
        for work, output in zip(batch_works, batch_outputs, strict=True):
            prediction, citation_ids = prediction_from_output(output)
            results.append(
                {
                    "case_id": work["case_id"],
                    "prediction": prediction,
                    "citation_ids": citation_ids,
                }
            )
        batches.append(
            {
                "schema_version": "1.0.0",
                "batch_id": f"squad_v2:{start // args.batch_size:05d}",
                "benchmark_id": "squad_v2",
                "case_ids": [work["case_id"] for work in batch_works],
                "valid": True,
                "results": results,
                "extractive_outputs": batch_outputs,
                "errors": [],
                "raw_contract_valid": True,
            }
        )

    work_by_id = {work["case_id"]: work for work in works}
    rows = case_results(batches, work_by_id)
    batch_path = args.out.with_suffix(".batches.jsonl")
    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(batch_path, batches)
    write_jsonl(case_path, rows)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "evaluation_lane": "oracle_context_extractive_reader",
        "retrieval_included": False,
        "profile_id": "deberta-v3-large-squad2-local-fp16",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_url": MODEL_URL,
        "model_license": "CC-BY-4.0",
        "model_artifacts": {
            "model_safetensors_sha256": sha256_file(model_path),
            "config_sha256": sha256_file(config_path),
            "sentencepiece_sha256": sha256_file(tokenizer_path),
        },
        "torch_version": torch.__version__,
        "transformers_model_class": type(model).__name__,
        "transformers_tokenizer_class": type(tokenizer).__name__,
        "device": torch.cuda.get_device_name(0),
        "dtype": "float16",
        "source_sha256": sha256_file(args.input),
        "case_ids_sha256": canonical_sha256([work["case_id"] for work in works]),
        "sample": args.sample,
        "sample_seed": SAMPLE_SEED if args.sample else None,
        "batch_size": args.batch_size,
        "max_context_chars": args.max_context_chars,
        "max_seq_len": args.max_seq_len,
        "doc_stride": args.doc_stride,
        "max_answer_len": args.max_answer_len,
        "handle_impossible_answer": True,
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "case_count": len(rows),
        "batch_count": len(batches),
        "valid_batch_count": len(batches),
        "fail_closed_case_count": sum(row["fail_closed"] for row in rows),
        "metrics": aggregate(rows),
        "operation": {
            "invocation_wall_seconds": time.perf_counter() - started,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
        "artifacts": {
            "batch_records_path": batch_path.resolve().relative_to(ROOT).as_posix(),
            "batch_records_sha256": sha256_file(batch_path),
            "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
            "case_records_sha256": sha256_file(case_path),
            "case_results_canonical_sha256": canonical_sha256(rows),
        },
        "provenance": {
            "public_annotations": "official_pinned_squad_v2_dev",
            "human_annotated": True,
        },
        "limitations": [
            "This is a SQuAD-2.0-fine-tuned extractive reader evaluated on the public SQuAD 2.0 development set.",
            "The official paragraph is supplied; retrieval is not evaluated.",
            "The public development set is not a blinded out-of-domain evaluation.",
            "Answer correctness uses official-style normalized exact match and token F1, not an LLM judge.",
        ],
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": report["case_count"],
                "metrics": report["metrics"],
                "operation": report["operation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
