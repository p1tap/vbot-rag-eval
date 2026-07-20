"""LoRA-adapt the pinned NQ extractive reader on a declared development window."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import peft
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    canonical_sha256,
    dense_work_items,
    write_json,
)
from scripts.benchmarks.run_nq_extractive_reader import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
)

RUNNER_VERSION = "1.0.0"


def find_answer(context: str, answers: list[str]) -> tuple[int, str] | None:
    folded = context.casefold()
    matches = []
    for answer in answers:
        position = folded.find(answer.casefold())
        if position >= 0:
            matches.append((position, context[position : position + len(answer)]))
    return min(matches, key=lambda row: (row[0], len(row[1]))) if matches else None


def select_training_context(work: dict) -> tuple[str, tuple[int, str] | None]:
    evidence_documents = {
        document_id
        for evidence_set in work["gold_evidence_sets"]
        for document_id in evidence_set
    }
    for context in work["contexts"]:
        if context["document_id"] not in evidence_documents:
            continue
        match = find_answer(context["text"], work["gold"]["answers"])
        if match:
            return context["text"], match
    return (work["contexts"][0]["text"], None)


def tokenize_training(works: list[dict], tokenizer, max_length: int, stride: int):
    features = []
    positive_cases = 0
    for work in works:
        context, match = select_training_context(work)
        positive_cases += match is not None
        encoded = tokenizer(
            work["query"],
            context,
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        for feature_index in range(len(encoded["input_ids"])):
            input_ids = encoded["input_ids"][feature_index]
            sequence_ids = encoded.sequence_ids(feature_index)
            offsets = encoded["offset_mapping"][feature_index]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            start_position = cls_index
            end_position = cls_index
            if match:
                answer_start, answer = match
                answer_end = answer_start + len(answer)
                context_tokens = [
                    index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
                ]
                if context_tokens:
                    first = context_tokens[0]
                    last = context_tokens[-1]
                    if offsets[first][0] <= answer_start and offsets[last][1] >= answer_end:
                        while first <= last and offsets[first][0] <= answer_start:
                            first += 1
                        start_position = first - 1
                        while last >= start_position and offsets[last][1] >= answer_end:
                            last -= 1
                        end_position = last + 1
            features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": encoded["attention_mask"][feature_index],
                    "start_positions": start_position,
                    "end_positions": end_position,
                }
            )
    return features, positive_cases


def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor([row[key] for row in rows], dtype=torch.long)
        for key in rows[0]
    }


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("natural_questions", "hotpotqa"),
        default="natural_questions",
    )
    parser.add_argument("--sample", type=int, default=1200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-document-chars", type=int, default=6000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--adapter-out",
        type=Path,
        default=ROOT / "artifacts" / "models" / "nq-deberta-v3-large-lora-offset0-1200",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT
        / "reports"
        / "public-benchmarks"
        / "nq-deberta-v3-large-lora-training-offset0-1200.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for LoRA training")
    if min(
        args.sample,
        args.top_k,
        args.max_document_chars,
        args.max_seq_len,
        args.stride,
        args.batch_size,
        args.gradient_accumulation,
        args.epochs,
        args.lora_r,
        args.lora_alpha,
    ) < 1 or args.offset < 0 or args.learning_rate <= 0:
        raise SystemExit("training options must be positive and offset nonnegative")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    works = dense_work_items(
        args.benchmark,
        top_k=args.top_k,
        max_document_chars=args.max_document_chars,
        sample_per_benchmark=args.sample,
        sample_offset_per_benchmark=args.offset,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    features, positive_cases = tokenize_training(
        works, tokenizer, args.max_seq_len, args.stride
    )
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        features,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        use_safetensors=True,
    )
    config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["query_proj", "key_proj", "value_proj"],
        modules_to_save=["qa_outputs"],
        bias="none",
    )
    model = get_peft_model(model, config).to("cuda")
    model.train()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    loss_values = []
    optimizer_steps = 0
    for _epoch in range(args.epochs):
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            loss_values.append(float(loss.detach().cpu()) * args.gradient_accumulation)
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

    args.adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_out, safe_serialization=True)
    tokenizer.save_pretrained(args.adapter_out)
    adapter_files = sorted(
        path for path in args.adapter_out.iterdir() if path.is_file()
    )
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "base_model_id": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "peft_version": peft.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "benchmark": args.benchmark,
        "seed": 42,
        "sample_offset": args.offset,
        "sample_count": args.sample,
        "case_ids_sha256": canonical_sha256(sorted(work["case_id"] for work in works)),
        "top_k": args.top_k,
        "max_document_chars": args.max_document_chars,
        "max_seq_len": args.max_seq_len,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_config": json.loads(
            json.dumps(
                config.to_dict(),
                default=lambda value: (
                    sorted(value) if isinstance(value, set) else str(value)
                ),
            )
        ),
    }
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "training_complete",
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "training": {
            "case_count": len(works),
            "positive_span_case_count": positive_cases,
            "tokenized_feature_count": len(features),
            "trainable_parameter_count": trainable_parameters,
            "optimizer_step_count": optimizer_steps,
            "mean_loss": sum(loss_values) / len(loss_values),
            "final_loss": loss_values[-1],
            "wall_seconds": time.perf_counter() - started,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
        "artifacts": {
            "adapter_directory": args.adapter_out.resolve().relative_to(ROOT).as_posix(),
            "files": {
                path.name: sha256_file(path) for path in adapter_files
            },
        },
        "limitations": [
            "Only the declared offset-0 development cases provide training labels.",
            "Promotion requires evaluation on a disjoint case window.",
            "One evidence context per case is selected before token-window expansion.",
        ],
    }
    write_json(args.report_out, report)
    print(json.dumps({"status": report["status"], **report["training"]}, indent=2))


if __name__ == "__main__":
    main()
