"""Run a pinned FEVER-trained local NLI model over retrieved documents."""
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    canonical_sha256,
    fever_work_items,
    write_json,
    write_jsonl,
)

MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MODEL_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}"
RUNNER_VERSION = "1.0.0"


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--offset", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-document-chars", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-training-report", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "reports"
        / "public-benchmarks"
        / "fever-local-nli-top12-offset800-development-300.json",
    )
    args = parser.parse_args()
    if min(
        args.sample,
        args.top_k,
        args.max_document_chars,
        args.batch_size,
        args.max_seq_len,
    ) < 1 or args.offset < 0:
        raise SystemExit("sample/options must be positive and offset nonnegative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for local NLI")

    works = fever_work_items(
        top_k=args.top_k,
        max_document_chars=args.max_document_chars,
        sample_per_benchmark=args.sample,
        sample_offset_per_benchmark=args.offset,
    )
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
    model = AutoModelForSequenceClassification.from_pretrained(
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
    label_by_index = {
        int(index): label.casefold()
        for index, label in model.config.id2label.items()
    }
    required_labels = {"entailment", "neutral", "contradiction"}
    if set(label_by_index.values()) != required_labels:
        raise RuntimeError(f"unexpected NLI labels: {label_by_index}")

    pairs = []
    identities = []
    for work in works:
        for context in work["contexts"]:
            pairs.append((context["text"], work["query"]))
            identities.append(
                {
                    "case_id": work["case_id"],
                    "citation_id": context["citation_id"],
                    "document_id": context["document_id"],
                }
            )
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            encoded = tokenizer(
                [row[0] for row in batch],
                [row[1] for row in batch],
                padding=True,
                truncation="only_first",
                max_length=args.max_seq_len,
                return_tensors="pt",
            ).to("cuda")
            probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1)
            outputs.extend(probabilities.cpu().tolist())
    if len(outputs) != len(identities):
        raise RuntimeError("NLI output count mismatch")

    by_case: dict[str, list[dict]] = {work["case_id"]: [] for work in works}
    for identity, probabilities in zip(identities, outputs, strict=True):
        scores = {
            label_by_index[index]: float(score)
            for index, score in enumerate(probabilities)
        }
        by_case[identity["case_id"]].append({**identity, **scores})
    records = [
        {
            "schema_version": "1.0.0",
            "case_id": work["case_id"],
            "benchmark_id": "fever",
            "query": work["query"],
            "retrieved_document_ids": work["retrieved_document_ids"],
            "gold": work["gold"],
            "gold_document_ids": sorted(work["gold_document_ids"]),
            "gold_evidence_sets": work["gold_evidence_sets"],
            "candidates": by_case[work["case_id"]],
        }
        for work in works
    ]
    records_path = args.out.with_suffix(".records.jsonl")
    write_jsonl(records_path, records)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "profile_id": (
            "deberta-v3-large-fever-nli-lora-local-fp16"
            if args.adapter
            else "deberta-v3-large-fever-nli-local-fp16"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_url": MODEL_URL,
        "model_license": "MIT",
        "model_artifacts": {
            "model_safetensors_sha256": sha256_file(model_path),
            "config_sha256": sha256_file(config_path),
            "sentencepiece_sha256": sha256_file(tokenizer_path),
        },
        "adapter_artifacts": adapter_artifacts,
        "model_labels": label_by_index,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "dtype": "float16",
        "sample_per_benchmark": args.sample,
        "sample_offset_per_benchmark": args.offset,
        "case_ids_sha256": canonical_sha256(sorted(by_case)),
        "top_k": args.top_k,
        "max_document_chars": args.max_document_chars,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "pair_order": "premise_document_then_hypothesis_claim",
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
            "Raw document-level NLI probabilities are stored without a selected aggregation policy.",
            "The offset-800 window is development data and requires disjoint confirmation.",
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
