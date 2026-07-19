"""Apply a resumable evidence-verification cascade to a frozen E2E report."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import llm  # noqa: E402
from rag.llm import chat_with_metadata  # noqa: E402
from rag.provenance import sha256_file  # noqa: E402
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    PROFILES,
    SYSTEM,
    aggregate,
    append_jsonl,
    canonical_sha256,
    case_results,
    complete_hotpot_citations,
    contract_batch,
    dense_work_items,
    fever_work_items,
    load_jsonl,
    parse_provider_output,
    response_format,
    validate_output,
    write_json,
    write_jsonl,
)

RUNNER_VERSION = "1.2.0"


def verification_prompt(
    batch: list[dict], results: list[dict], policy: str = "evidence_recheck"
) -> str:
    benchmark_id = batch[0]["benchmark_id"]
    rows = []
    for item, result in zip(batch, results, strict=True):
        rows.append(
            {
                "case_id": item["case_id"],
                "query": item["query"],
                "first_pass_prediction": result["prediction"],
                "first_pass_citation_ids": result["citation_ids"],
                "retrieved_evidence": [
                    {
                        "citation_id": context["citation_id"],
                        "title": context["title"],
                        "text": context["text"],
                    }
                    for context in item["contexts"]
                ],
            }
        )
    if benchmark_id == "natural_questions":
        rule = (
            "Natural Questions passages may be topically related without stating the "
            "requested fact. If the evidence directly and unambiguously states it, return "
            "the shortest exact answer span; otherwise return __UNANSWERABLE__. Cite every "
            "retrieved passage needed for the answer and every retrieved passage that "
            "independently supplies the same answer."
        )
    elif benchmark_id == "hotpotqa" and policy == "hotpot_answer_recheck":
        rule = (
            "For each HotpotQA question, independently reconstruct the complete entity "
            "or comparison chain before deciding whether to retain the candidate answer. "
            "Check that the final answer has the type explicitly requested by the question "
            "(person, place, date, number, yes/no, or other attribute), and do not return an "
            "intermediate bridge entity. For comparison questions, establish the requested "
            "attribute for both subjects before comparing them. Prefer a directly supported "
            "minimal answer over abstention; use __UNANSWERABLE__ only when the retrieved "
            "evidence truly cannot establish the chain. Cite every document in the final "
            "supporting chain and no unrelated document."
        )
    elif benchmark_id == "hotpotqa" and policy == "hotpot_citation_reselection":
        rule = (
            "Treat the first-pass prediction as frozen and do not alter or reinterpret it. "
            "Independently rebuild the minimal complete evidence chain for that frozen answer. "
            "You may replace incorrect first-pass citations. Cite every bridge and final-answer "
            "document required by the chain, and cite no merely related document. Return the "
            "frozen prediction verbatim with the newly selected citations."
        )
    elif benchmark_id == "hotpotqa":
        rule = (
            "HotpotQA may require a multi-hop chain. Re-solve the question and return a "
            "minimal exact answer only if the retrieved evidence establishes every hop. "
            "Cite the complete chain, including bridge and final-answer documents. If any "
            "required hop is missing, return __UNANSWERABLE__."
        )
    elif benchmark_id == "fever":
        rule = (
            "For FEVER, prediction must be exactly supports, refutes, or "
            "not_enough_info. Use supports or refutes only when a complete retrieved "
            "evidence set directly establishes that label; cite that complete set. Use "
            "not_enough_info with no citations when retrieved evidence is insufficient."
        )
    else:
        raise ValueError(f"unsupported verification benchmark: {benchmark_id}")
    return (
        "Independently verify each first-pass result using only its retrieved evidence. "
        "Ignore background knowledge and do not defer to the first-pass conclusion. "
        f"{rule} Do not cite merely related passages. Return exactly one result per case "
        "in input order with only case_id, prediction, and citation_ids.\n\n"
        + json.dumps(rows, ensure_ascii=False)
    )


def recheck_batch(
    batch: list[dict],
    original_results: list[dict],
    profile_id: str,
    policy: str = "evidence_recheck",
) -> dict:
    profile = PROFILES[profile_id]
    wire_batch = contract_batch(batch)
    source_to_wire = {
        source["case_id"]: wire["case_id"]
        for source, wire in zip(batch, wire_batch, strict=True)
    }
    wire_to_source = {wire: source for source, wire in source_to_wire.items()}
    original_by_id = {result["case_id"]: result for result in original_results}
    wire_results = [
        {
            **original_by_id[item["case_id"]],
            "case_id": source_to_wire[item["case_id"]],
        }
        for item in batch
    ]
    if policy in {
        "hotpot_citation_completion",
        "hotpot_single_citation_completion",
        "frozen_citation_completion",
    }:
        if (
            policy
            in {"hotpot_citation_completion", "hotpot_single_citation_completion"}
            and batch[0]["benchmark_id"] != "hotpotqa"
        ):
            raise ValueError("Hotpot citation completion cannot run on another benchmark")
        completed, completion_record, calls, raw_outputs = complete_hotpot_citations(
            wire_batch,
            wire_results,
            profile,
            single_citation_only=policy == "hotpot_single_citation_completion",
        )
        mapped = [
            {**result, "case_id": wire_to_source[result["case_id"]]}
            for result in completed
        ]
        mapped_by_id = {result["case_id"]: result for result in mapped}
        ordered = [mapped_by_id[item["case_id"]] for item in batch]
        changes = [
            {
                "case_id": before["case_id"],
                "before_prediction": before["prediction"],
                "after_prediction": after["prediction"],
                "before_citation_ids": before["citation_ids"],
                "after_citation_ids": after["citation_ids"],
            }
            for before, after in zip(original_results, ordered, strict=True)
            if before["citation_ids"] != after["citation_ids"]
        ]
        return {
            "valid": (
                completion_record.get("valid", True)
                if completion_record.get("attempted")
                else True
            ),
            "results": ordered,
            "changes": changes,
            "errors": completion_record.get("errors", []),
            "calls": calls,
            "raw_outputs": raw_outputs,
            "normalization_attempts": [],
            "completion_record": completion_record,
        }
    if policy not in {
        "evidence_recheck",
        "hotpot_answer_recheck",
        "hotpot_citation_reselection",
    }:
        raise ValueError(f"unknown verifier cascade policy: {policy}")
    if policy in {
        "hotpot_answer_recheck",
        "hotpot_citation_reselection",
    } and batch[0]["benchmark_id"] != "hotpotqa":
        raise ValueError("Hotpot recheck cannot run on another benchmark")
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": verification_prompt(wire_batch, wire_results, policy),
        },
    ]
    calls = []
    raw_outputs = []
    normalization_attempts = []
    errors = []
    accepted = None
    for _ in range(3):
        try:
            call = chat_with_metadata(
                profile["model"],
                messages,
                max_tokens=profile["max_tokens"],
                temperature=profile["temperature"],
                retries=6,
                response_format=response_format(wire_batch, profile["strict_schema"]),
                request_options=profile["request_options"],
                timeout_seconds=profile.get("timeout_seconds"),
            )
            call_record = call.to_record()
            call_record["execution_scope"] = "benchmark_evidence_verification"
            call_record["wire_case_ids"] = [item["case_id"] for item in wire_batch]
            calls.append(call_record)
            raw_outputs.append(call.content)
            candidate, events, parse_error = parse_provider_output(call.content, wire_batch)
            normalization_attempts.append(events)
            if parse_error:
                errors = [parse_error]
                continue
            errors = validate_output(candidate, wire_batch)
            if errors:
                continue
            accepted = candidate["results"]
            break
        except Exception as exc:  # noqa: BLE001
            errors = [f"{type(exc).__name__}: {exc}"]
    if accepted is None:
        return {
            "valid": False,
            "results": original_results,
            "changes": [],
            "errors": errors,
            "calls": calls,
            "raw_outputs": raw_outputs,
            "normalization_attempts": normalization_attempts,
            "fallback": "preserved_frozen_first_pass",
        }
    mapped = [{**result, "case_id": wire_to_source[result["case_id"]]} for result in accepted]
    mapped_by_id = {result["case_id"]: result for result in mapped}
    ordered = [mapped_by_id[item["case_id"]] for item in batch]
    if policy == "hotpot_citation_reselection":
        ordered = [
            {**after, "prediction": before["prediction"]}
            for before, after in zip(original_results, ordered, strict=True)
        ]
    changes = [
        {
            "case_id": before["case_id"],
            "before_prediction": before["prediction"],
            "after_prediction": after["prediction"],
            "before_citation_ids": before["citation_ids"],
            "after_citation_ids": after["citation_ids"],
        }
        for before, after in zip(original_results, ordered, strict=True)
        if before["prediction"] != after["prediction"]
        or before["citation_ids"] != after["citation_ids"]
    ]
    return {
        "valid": True,
        "results": ordered,
        "changes": changes,
        "errors": [],
        "calls": calls,
        "raw_outputs": raw_outputs,
        "normalization_attempts": normalization_attempts,
        "messages_sha256": canonical_sha256(messages),
    }


def reconstruct_work(report: dict) -> list[dict]:
    identity = report["identity"]
    sample = identity["sample_per_benchmark"]
    sample_offset = identity.get("sample_offset_per_benchmark", 0)
    top_k = identity["top_k_by_benchmark"]
    retrieval_mode = identity["retrieval"]["bounded_candidate_mode"]
    compression_mode = identity["compression"]["mode"]
    max_chars = identity["compression"]["max_document_chars"]
    works = []
    for benchmark_id in identity["benchmarks"]:
        if benchmark_id == "fever":
            works.extend(
                fever_work_items(
                    top_k=top_k[benchmark_id],
                    max_document_chars=max_chars,
                    sample_per_benchmark=sample,
                    sample_offset_per_benchmark=sample_offset,
                    compression_mode=compression_mode,
                )
            )
        else:
            works.extend(
                dense_work_items(
                    benchmark_id,
                    top_k=top_k[benchmark_id],
                    max_document_chars=max_chars,
                    sample_per_benchmark=sample,
                    sample_offset_per_benchmark=sample_offset,
                    retrieval_mode=retrieval_mode,
                    compression_mode=compression_mode,
                )
            )
    return works


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--benchmarks", default="hotpotqa,natural_questions,fever")
    parser.add_argument(
        "--policy",
        choices=(
            "evidence_recheck",
            "hotpot_citation_completion",
            "hotpot_single_citation_completion",
            "frozen_citation_completion",
            "hotpot_answer_recheck",
            "hotpot_citation_reselection",
        ),
        default="evidence_recheck",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--batch-ids",
        default="",
        help="Optional comma-separated batch IDs to verify; all selected batches by default.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    profile_id = args.profile or baseline["identity"]["profile_id"]
    profile = PROFILES[profile_id]
    expected_endpoint = profile.get("endpoint")
    if expected_endpoint and llm.BASE_URL != expected_endpoint:
        raise SystemExit(f"profile {profile_id} requires RAG_LLM_BASE_URL={expected_endpoint}")
    selected = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    if not selected or selected - set(baseline["identity"]["benchmarks"]):
        raise SystemExit("unknown or empty benchmark selection")
    if args.policy in {
        "hotpot_citation_completion",
        "hotpot_single_citation_completion",
        "hotpot_answer_recheck",
        "hotpot_citation_reselection",
    } and selected != {"hotpotqa"}:
        raise SystemExit("Hotpot-only policies require --benchmarks hotpotqa")
    if args.policy == "frozen_citation_completion" and len(selected) != 1:
        raise SystemExit("frozen_citation_completion requires exactly one benchmark")
    selected_batch_ids = {
        item.strip() for item in args.batch_ids.split(",") if item.strip()
    }

    works = reconstruct_work(baseline)
    work_by_id = {item["case_id"]: item for item in works}
    baseline_batch_path = ROOT / baseline["artifacts"]["batch_records_path"]
    baseline_batches = load_jsonl(baseline_batch_path)
    identity = {
        **baseline["identity"],
        "generator_runner_version": baseline["identity"]["runner_version"],
        "generator_runner_source_sha256": baseline["identity"]["runner_source_sha256"],
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "frozen_baseline_report_sha256": sha256_file(args.baseline),
        "frozen_baseline_case_results_sha256": baseline["artifacts"]["case_results_canonical_sha256"],
        "verification_profile_id": profile_id,
        "verification_profile": profile,
        "verification_benchmarks": sorted(selected),
        "verification_policy": args.policy,
        "verification_batch_ids": sorted(selected_batch_ids),
    }
    identity_sha256 = canonical_sha256(identity)
    checkpoint = args.out.with_suffix(".verification.jsonl")
    history = {row["batch_id"]: row for row in load_jsonl(checkpoint)}
    tasks = []
    for batch in baseline_batches:
        if batch["benchmark_id"] not in selected:
            continue
        if selected_batch_ids and batch["batch_id"] not in selected_batch_ids:
            continue
        old = history.get(batch["batch_id"])
        if old and old.get("run_identity_sha256") == identity_sha256:
            continue
        batch_work = [work_by_id[case_id] for case_id in batch["case_ids"]]
        tasks.append((batch, batch_work))
    if selected_batch_ids:
        available_batch_ids = {batch["batch_id"] for batch, _ in tasks}
        missing_batch_ids = selected_batch_ids - available_batch_ids
        if missing_batch_ids:
            raise SystemExit(
                f"unknown or already completed batch IDs: {sorted(missing_batch_ids)}"
            )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                recheck_batch,
                batch_work,
                batch["results"],
                profile_id,
                args.policy,
            ): batch
            for batch, batch_work in tasks
        }
        completed = len(history)
        for future in as_completed(futures):
            batch = futures[future]
            result = future.result()
            record = {
                "schema_version": "1.0.0",
                "batch_id": batch["batch_id"],
                "benchmark_id": batch["benchmark_id"],
                "case_ids": batch["case_ids"],
                "run_identity_sha256": identity_sha256,
                **result,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(checkpoint, record)
            history[batch["batch_id"]] = record
            completed += 1
            print(f"[{completed}/{len(tasks)}] {batch['batch_id']}: {'valid' if record['valid'] else 'FALLBACK'}", flush=True)

    candidate_batches = copy.deepcopy(baseline_batches)
    for batch in candidate_batches:
        verification = history.get(batch["batch_id"])
        if verification:
            batch["frozen_first_pass_results_sha256"] = canonical_sha256(batch["results"])
            batch["results"] = verification["results"]
            batch["benchmark_verification"] = {
                key: value
                for key, value in verification.items()
                if key not in {"results", "calls", "raw_outputs"}
            }
            batch["calls"] = [*batch.get("calls", []), *verification.get("calls", [])]
            batch["raw_outputs"] = [
                *batch.get("raw_outputs", []),
                *verification.get("raw_outputs", []),
            ]

    rows = case_results(candidate_batches, work_by_id)
    batch_path = args.out.with_suffix(".batches.jsonl")
    case_path = args.out.with_suffix(".cases.jsonl")
    write_jsonl(batch_path, candidate_batches)
    write_jsonl(case_path, rows)
    verification_rows = [history[batch["batch_id"]] for batch in baseline_batches if batch["batch_id"] in history]
    verification_calls = [call for row in verification_rows for call in row.get("calls", [])]
    usage = sum(
        (
            Counter({key: value for key, value in call.get("usage", {}).items() if isinstance(value, (int, float))})
            for call in verification_calls
        ),
        Counter(),
    )
    report = copy.deepcopy(baseline)
    report.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity_sha256": identity_sha256,
            "identity": identity,
            "metrics": aggregate(rows),
            "fail_closed_case_count": sum(row["fail_closed"] for row in rows),
            "verification": {
                "batch_count": len(verification_rows),
                "valid_batch_count": sum(row["valid"] for row in verification_rows),
                "fallback_batch_count": sum(not row["valid"] for row in verification_rows),
                "changed_case_count": sum(len(row["changes"]) for row in verification_rows),
                "call_count": len(verification_calls),
                "wall_seconds": time.perf_counter() - started,
                "usage": dict(sorted(usage.items())),
            },
            "artifacts": {
                "baseline_report_path": args.baseline.resolve().relative_to(ROOT).as_posix(),
                "baseline_report_sha256": sha256_file(args.baseline),
                "verification_records_path": checkpoint.resolve().relative_to(ROOT).as_posix(),
                "verification_records_sha256": sha256_file(checkpoint),
                "batch_records_path": batch_path.resolve().relative_to(ROOT).as_posix(),
                "batch_records_sha256": sha256_file(batch_path),
                "case_records_path": case_path.resolve().relative_to(ROOT).as_posix(),
                "case_records_sha256": sha256_file(case_path),
                "case_results_canonical_sha256": canonical_sha256(rows),
            },
            "limitations": [
                *baseline.get("limitations", []),
                "The verifier sees only frozen first-pass predictions and the exact same retrieved evidence; it has no access to gold answers or labels.",
                "A verifier contract failure preserves the frozen first-pass result and is counted as a fallback batch.",
            ],
        }
    )
    write_json(args.out, report)
    print(json.dumps({"case_count": len(rows), "metrics": report["metrics"], "verification": report["verification"]}, indent=2))


if __name__ == "__main__":
    main()
