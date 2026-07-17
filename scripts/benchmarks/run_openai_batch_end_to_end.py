"""Run the public RAG generator through direct OpenAI Batch/Responses.

Lifecycle:
  prepare  -> deterministic local JSONL shards (no API key, no cost)
  submit   -> upload and create paid 24-hour batches (explicit confirmation)
  sync     -> refresh metadata and download provider output/error files
  finalize -> validate by custom_id, fail closed, and publish scored artifacts
  retry    -> append shards containing only missing/failed/invalid custom_ids

One logical benchmark case is sent per API request. That makes failures and
retries case-isolated while preserving the frozen public prompt and scoring
contract used by the historical synchronous runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.openai_batch import (  # noqa: E402
    RESPONSES_ENDPOINT,
    TERMINAL_BATCH_STATUSES,
    OpenAIBatchClient,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    extract_responses_output_text,
    index_batch_results,
    make_responses_request,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    shard_requests,
)
from rag.provenance import sha256_file as provenance_sha256_file  # noqa: E402
from scripts.benchmarks.audit_suite import (  # noqa: E402
    DEFAULT_INPUTS,
    MAX_NORMALIZED_BATCH_RATE,
    PUBLIC_E2E_RESPONSE_CONTRACT,
    audit_suite,
)
from scripts.benchmarks.run_end_to_end import (  # noqa: E402
    SYSTEM,
    aggregate,
    case_results,
    contract_batch,
    dense_work_items,
    fever_work_items,
    parse_provider_output,
    prompt,
    response_format,
    validate_output,
    write_json,
    write_jsonl,
)


RUNNER_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "1.0.0"
DEFAULT_OUT = (
    ROOT
    / "reports"
    / "public-benchmarks"
    / "end-to-end-gpt56-sol-openai-batch.json"
)
DEFAULT_RUN_ROOT = ROOT / "artifacts" / "openai-batch"
PROFILES = {
    "gpt-5.6-sol-standard-high": {
        "model": "gpt-5.6-sol",
        "reasoning_mode": "standard",
        "reasoning_effort": "high",
        "max_output_tokens": 8000,
        "strict_schema": True,
        "api": "openai_responses_batch",
        "pricing_snapshot_utc": "2026-07-18",
        "batch_input_usd_per_million_tokens": 2.50,
        "batch_output_usd_per_million_tokens": 15.00,
    },
    "gpt-5.6-sol-standard-xhigh": {
        "model": "gpt-5.6-sol",
        "reasoning_mode": "standard",
        "reasoning_effort": "xhigh",
        "max_output_tokens": 8000,
        "strict_schema": True,
        "api": "openai_responses_batch",
        "pricing_snapshot_utc": "2026-07-18",
        "batch_input_usd_per_million_tokens": 2.50,
        "batch_output_usd_per_million_tokens": 15.00,
    },
    "gpt-5.6-sol-pro-high": {
        "model": "gpt-5.6-sol",
        "reasoning_mode": "pro",
        "reasoning_effort": "high",
        "max_output_tokens": 8000,
        "strict_schema": True,
        "api": "openai_responses_batch",
        "pricing_snapshot_utc": "2026-07-18",
        "batch_input_usd_per_million_tokens": 2.50,
        "batch_output_usd_per_million_tokens": 15.00,
    },
    "gpt-5.6-sol-pro-xhigh": {
        "model": "gpt-5.6-sol",
        "reasoning_mode": "pro",
        "reasoning_effort": "xhigh",
        "max_output_tokens": 8000,
        "strict_schema": True,
        "api": "openai_responses_batch",
        "pricing_snapshot_utc": "2026-07-18",
        "batch_input_usd_per_million_tokens": 2.50,
        "batch_output_usd_per_million_tokens": 15.00,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_custom_id(case_id: str) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:24]
    return f"rag-{digest}"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def load_manifest(run_dir: Path) -> dict:
    path = manifest_path(run_dir)
    if not path.exists():
        raise SystemExit(f"run manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SystemExit(f"unsupported manifest schema: {manifest.get('schema_version')}")
    return manifest


def save_manifest(run_dir: Path, manifest: dict) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(manifest_path(run_dir), manifest)


def verify_local_contract_sources(manifest: dict) -> None:
    identity = manifest["identity"]
    checks = {
        "runner_source_sha256": provenance_sha256_file(Path(__file__)),
        "transport_source_sha256": provenance_sha256_file(
            ROOT / "rag" / "openai_batch.py"
        ),
    }
    for field, observed in checks.items():
        if identity.get(field) != observed:
            raise SystemExit(
                f"local {field} changed after prepare; create a new frozen run"
            )


def relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def prepare_works(args: argparse.Namespace) -> tuple[dict, list[dict], list[str]]:
    benchmark_ids = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    if not benchmark_ids or set(benchmark_ids) - set(DEFAULT_INPUTS):
        raise SystemExit("unknown or empty benchmark selection")
    suite = audit_suite()
    works = []
    retrieval_sample = (
        args.sample_per_benchmark + args.sample_offset_per_benchmark
        if args.sample_per_benchmark
        else 0
    )
    for benchmark_id in benchmark_ids:
        if benchmark_id == "fever":
            selected = fever_work_items(
                top_k=args.top_k,
                max_document_chars=args.max_document_chars,
                sample_per_benchmark=retrieval_sample,
            )
        else:
            selected = dense_work_items(
                benchmark_id,
                top_k=args.top_k,
                max_document_chars=args.max_document_chars,
                sample_per_benchmark=retrieval_sample,
            )
        if args.sample_per_benchmark:
            selected = sorted(
                selected,
                key=lambda row: (
                    hashlib.sha256(
                        f"public-e2e-pilot-v1:{row['case_id']}".encode("utf-8")
                    ).hexdigest(),
                    row["case_id"],
                ),
            )[
                args.sample_offset_per_benchmark :
                args.sample_offset_per_benchmark + args.sample_per_benchmark
            ]
        works.extend(selected)
        print(f"prepared {benchmark_id}: {len(selected)}", flush=True)
    if len({row["case_id"] for row in works}) != len(works):
        raise SystemExit("prepared case IDs are not unique")
    return suite, works, benchmark_ids


def prepare(args: argparse.Namespace) -> None:
    if min(
        args.top_k,
        args.max_document_chars,
        args.max_requests_per_shard,
        args.max_file_bytes,
        args.max_estimated_prompt_tokens,
    ) < 1:
        raise SystemExit("all numeric preparation limits must be positive")
    if args.sample_offset_per_benchmark < 0:
        raise SystemExit("sample-offset-per-benchmark cannot be negative")
    if args.sample_per_benchmark < 0:
        raise SystemExit("sample-per-benchmark cannot be negative")
    if not args.sample_per_benchmark and args.sample_offset_per_benchmark:
        raise SystemExit("sample offset requires sample-per-benchmark")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.out}")
    profile = PROFILES[args.profile]
    suite, works, benchmark_ids = prepare_works(args)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": provenance_sha256_file(Path(__file__)),
        "transport_source_sha256": provenance_sha256_file(
            ROOT / "rag" / "openai_batch.py"
        ),
        "response_contract_version": PUBLIC_E2E_RESPONSE_CONTRACT,
        "transport": "openai_responses_batch",
        "profile_id": args.profile,
        "profile": profile,
        "benchmarks": benchmark_ids,
        "source_sha256": {
            item: provenance_sha256_file(DEFAULT_INPUTS[item]) for item in benchmark_ids
        },
        "suite_report_sha256": canonical_sha256(suite),
        "top_k": args.top_k,
        "logical_cases_per_request": 1,
        "max_document_chars": args.max_document_chars,
        "sample_per_benchmark": args.sample_per_benchmark,
        "sample_offset_per_benchmark": args.sample_offset_per_benchmark,
        "system_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "case_ids_sha256": canonical_sha256([row["case_id"] for row in works]),
        "shard_limits": {
            "max_requests": args.max_requests_per_shard,
            "max_file_bytes": args.max_file_bytes,
            "max_estimated_prompt_tokens": args.max_estimated_prompt_tokens,
            "token_estimate": "ceil((utf8_prompt_bytes+utf8_schema_bytes)/3)+64",
        },
    }
    identity_sha256 = canonical_sha256(identity)
    run_dir = args.run_dir or DEFAULT_RUN_ROOT / identity_sha256[:16]
    existing_path = manifest_path(run_dir)
    if existing_path.exists():
        existing = load_manifest(run_dir)
        if existing.get("identity_sha256") != identity_sha256:
            raise SystemExit(f"incompatible existing run: {existing_path}")
        print(f"already prepared: {existing_path}")
        return

    requests_ = []
    request_index = {}
    for work in works:
        wire = contract_batch([work])
        custom_id = stable_custom_id(work["case_id"])
        if custom_id in request_index:
            raise SystemExit(f"custom_id collision: {custom_id}")
        request = make_responses_request(
            custom_id=custom_id,
            model=profile["model"],
            system=SYSTEM,
            user=prompt(wire),
            chat_response_format=response_format(wire, True),
            reasoning_effort=profile["reasoning_effort"],
            reasoning_mode=profile["reasoning_mode"],
            max_output_tokens=profile["max_output_tokens"],
        )
        requests_.append(request)
        request_index[custom_id] = {
            "case_id": work["case_id"],
            "benchmark_id": work["benchmark_id"],
            "work_sha256": canonical_sha256(work),
            "request_sha256": canonical_sha256(request),
        }
    planned_shards = shard_requests(
        requests_,
        max_requests=args.max_requests_per_shard,
        max_file_bytes=args.max_file_bytes,
        max_estimated_prompt_tokens=args.max_estimated_prompt_tokens,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    works_path = run_dir / "work-items.jsonl"
    atomic_write_jsonl(works_path, works)
    shard_records = []
    for index, planned in enumerate(planned_shards):
        request_path = run_dir / "requests" / f"generation-00-shard-{index:04d}.jsonl"
        atomic_write_jsonl(request_path, planned["requests"])
        shard_records.append(
            {
                "generation": 0,
                "shard_index": index,
                "state": "prepared",
                "request_path": relative_to_root(request_path),
                "request_sha256": sha256_file(request_path),
                "request_count": planned["request_count"],
                "jsonl_bytes": planned["jsonl_bytes"],
                "estimated_prompt_tokens": planned["estimated_prompt_tokens"],
                "first_custom_id": planned["first_custom_id"],
                "last_custom_id": planned["last_custom_id"],
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "state": "prepared",
        "identity_sha256": identity_sha256,
        "identity": identity,
        "output_report_path": relative_to_root(args.out),
        "work_items_path": relative_to_root(works_path),
        "work_items_sha256": sha256_file(works_path),
        "case_count": len(works),
        "request_index": request_index,
        "shards": shard_records,
        "submission_requires": "--confirm-paid-submit",
    }
    save_manifest(run_dir, manifest)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "identity_sha256": identity_sha256,
                "cases": len(works),
                "shards": len(shard_records),
                "estimated_prompt_tokens": sum(
                    row["estimated_prompt_tokens"] for row in shard_records
                ),
                "paid_calls_made": False,
            },
            indent=2,
        )
    )


def verified_request_path(shard: dict) -> Path:
    path = resolve_recorded_path(shard["request_path"])
    observed = sha256_file(path)
    if observed != shard["request_sha256"]:
        raise SystemExit(f"request shard hash mismatch: {path}")
    return path


def submit(args: argparse.Namespace) -> None:
    if not args.confirm_paid_submit:
        raise SystemExit(
            "paid submission blocked; inspect the manifest, then pass "
            "--confirm-paid-submit"
        )
    manifest = load_manifest(args.run_dir)
    verify_local_contract_sources(manifest)
    if args.max_shards < 1:
        raise SystemExit("max-shards must be positive")
    client = OpenAIBatchClient()
    changed = False
    submitted_now = 0
    for shard in manifest["shards"]:
        if shard.get("batch_id"):
            continue
        if submitted_now >= args.max_shards:
            break
        request_path = verified_request_path(shard)
        if not shard.get("input_file_id"):
            uploaded = client.upload_batch_input(request_path)
            shard["input_file_id"] = uploaded["id"]
            shard["input_file"] = uploaded
            shard["state"] = "uploaded"
            save_manifest(args.run_dir, manifest)
        created = client.create_batch(
            shard["input_file_id"],
            endpoint=RESPONSES_ENDPOINT,
            metadata={
                "run": manifest["identity_sha256"][:32],
                "generation": str(shard["generation"]),
                "shard": str(shard["shard_index"]),
            },
            idempotency_key=canonical_sha256(
                {
                    "run": manifest["identity_sha256"],
                    "generation": shard["generation"],
                    "shard": shard["shard_index"],
                    "request_sha256": shard["request_sha256"],
                }
            ),
        )
        shard["batch_id"] = created["id"]
        shard["batch_status"] = created.get("status")
        shard["state"] = "submitted"
        shard["submitted_at_utc"] = utc_now()
        shard["batch_metadata"] = created
        save_manifest(args.run_dir, manifest)
        changed = True
        submitted_now += 1
        print(f"submitted {request_path.name} -> {created['id']}", flush=True)
    manifest["state"] = "submitted" if any(
        row.get("batch_id") for row in manifest["shards"]
    ) else manifest["state"]
    save_manifest(args.run_dir, manifest)
    if not changed:
        print("no unsubmitted shards")


def download_provider_file(
    client: OpenAIBatchClient,
    *,
    run_dir: Path,
    shard: dict,
    field: str,
    kind: str,
) -> None:
    file_id = shard.get("batch_metadata", {}).get(field)
    if not file_id:
        return
    path = run_dir / "provider" / (
        f"generation-{shard['generation']:02d}-shard-{shard['shard_index']:04d}.{kind}.jsonl"
    )
    existing_hash = shard.get(f"{kind}_sha256")
    if path.exists() and existing_hash and sha256_file(path) == existing_hash:
        return
    content = client.download_file(file_id)
    atomic_write_bytes(path, content)
    shard[f"{kind}_file_id"] = file_id
    shard[f"{kind}_path"] = relative_to_root(path)
    shard[f"{kind}_sha256"] = sha256_bytes(content)


def sync_once(run_dir: Path, client: OpenAIBatchClient) -> tuple[dict, bool]:
    manifest = load_manifest(run_dir)
    active = False
    for shard in manifest["shards"]:
        batch_id = shard.get("batch_id")
        if not batch_id:
            continue
        metadata = client.retrieve_batch(batch_id)
        shard["batch_metadata"] = metadata
        shard["batch_status"] = metadata.get("status")
        shard["last_synced_at_utc"] = utc_now()
        metadata_path = run_dir / "provider" / (
            f"generation-{shard['generation']:02d}-shard-{shard['shard_index']:04d}.batch.json"
        )
        atomic_write_json(metadata_path, metadata)
        shard["batch_metadata_path"] = relative_to_root(metadata_path)
        shard["batch_metadata_sha256"] = sha256_file(metadata_path)
        download_provider_file(
            client, run_dir=run_dir, shard=shard,
            field="output_file_id", kind="output"
        )
        download_provider_file(
            client, run_dir=run_dir, shard=shard,
            field="error_file_id", kind="error"
        )
        terminal = metadata.get("status") in TERMINAL_BATCH_STATUSES
        shard["state"] = "downloaded" if terminal else "submitted"
        active = active or not terminal
        save_manifest(run_dir, manifest)
        counts = metadata.get("request_counts") or {}
        print(
            f"{batch_id}: {metadata.get('status')} "
            f"{counts.get('completed', 0)}/{counts.get('total', '?')}",
            flush=True,
        )
    submitted = [row for row in manifest["shards"] if row.get("batch_id")]
    if submitted and all(
        row.get("batch_status") in TERMINAL_BATCH_STATUSES for row in submitted
    ):
        manifest["state"] = "downloaded"
    save_manifest(run_dir, manifest)
    return manifest, active


def sync(args: argparse.Namespace) -> None:
    if args.poll_seconds < 10:
        raise SystemExit("poll-seconds must be at least 10")
    client = OpenAIBatchClient()
    while True:
        _, active = sync_once(args.run_dir, client)
        if not args.watch or not active:
            break
        time.sleep(args.poll_seconds)


def load_provider_attempts(manifest: dict) -> dict[str, list[dict]]:
    attempts: dict[str, list[dict]] = {}
    for shard in manifest["shards"]:
        for kind in ("output", "error"):
            if shard.get(f"{kind}_path"):
                path = resolve_recorded_path(shard[f"{kind}_path"])
                if sha256_file(path) != shard.get(f"{kind}_sha256"):
                    raise SystemExit(f"provider {kind} hash mismatch: {path}")
        output_rows = (
            read_jsonl(resolve_recorded_path(shard["output_path"]))
            if shard.get("output_path")
            else []
        )
        error_rows = (
            read_jsonl(resolve_recorded_path(shard["error_path"]))
            if shard.get("error_path")
            else []
        )
        indexed = index_batch_results(output_rows, error_rows)
        for custom_id, entry in indexed.items():
            if custom_id not in manifest["request_index"]:
                raise SystemExit(f"provider returned unknown custom_id: {custom_id}")
            attempts.setdefault(custom_id, []).append(
                {
                    **entry,
                    "generation": shard["generation"],
                    "shard_index": shard["shard_index"],
                    "batch_id": shard.get("batch_id"),
                }
            )
    return attempts


def provider_error(entry: dict) -> str:
    row = entry["row"]
    if entry["source"] == "error":
        return canonical_sha256(row) + ": " + str(row.get("error"))
    response = row.get("response")
    if not isinstance(response, dict):
        return "provider output has no response object"
    return f"provider status_code={response.get('status_code')}: {row.get('error')}"


def evaluated_attempt(entry: dict, work: dict, profile_id: str) -> dict:
    wire = contract_batch([work])
    base = {
        "schema_version": "1.2.0",
        "batch_id": stable_custom_id(work["case_id"]),
        "benchmark_id": work["benchmark_id"],
        "case_ids": [work["case_id"]],
        "wire_case_id_mapping": [
            {"wire_case_id": "C1", "case_id": work["case_id"]}
        ],
        "input_sha256": canonical_sha256([work]),
        "wire_input_sha256": canonical_sha256(wire),
        "profile_id": profile_id,
        "profile_sha256": canonical_sha256(PROFILES[profile_id]),
        "provider_batch_id": entry.get("batch_id"),
        "provider_generation": entry.get("generation"),
        "provider_shard_index": entry.get("shard_index"),
        "completed_at_utc": utc_now(),
        "execution_events": [],
        "case_slot_retry_records": [],
    }
    row = entry["row"]
    response = row.get("response")
    if entry["source"] != "output" or not isinstance(response, dict) or not (
        200 <= int(response.get("status_code", 0)) < 300
    ):
        return {
            **base,
            "valid": False,
            "results": [],
            "errors": [provider_error(entry)],
            "calls": [],
            "raw_outputs": [],
            "normalization_attempts": [],
            "normalization_events": [],
            "raw_contract_valid": False,
        }
    body = response.get("body")
    try:
        content = extract_responses_output_text(body)
        candidate, events, parse_error = parse_provider_output(content, wire)
        errors = [parse_error] if parse_error else validate_output(candidate, wire)
    except Exception as exc:  # noqa: BLE001 - retain and fail closed
        content = ""
        candidate = None
        events = []
        errors = [f"{type(exc).__name__}: {exc}"]
    valid = not errors
    result = (
        [{**candidate["results"][0], "case_id": work["case_id"]}]
        if valid
        else []
    )
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    call = {
        "provider": "openai",
        "api": "responses_batch",
        "model": body.get("model") if isinstance(body, dict) else None,
        "response_id": body.get("id") if isinstance(body, dict) else None,
        "request_id": response.get("request_id"),
        "provider_batch_id": entry.get("batch_id"),
        "usage": usage if isinstance(usage, dict) else {},
        "cost": None,
        "execution_scope": "case",
        "wire_case_ids": ["C1"],
    }
    return {
        **base,
        "valid": valid,
        "results": result,
        "errors": errors,
        "calls": [call],
        "raw_outputs": [content],
        "normalization_attempts": [events],
        "normalization_events": events if valid else [],
        "raw_contract_valid": valid and not events,
    }


def missing_attempt(work: dict, profile_id: str) -> dict:
    return {
        "schema_version": "1.2.0",
        "batch_id": stable_custom_id(work["case_id"]),
        "benchmark_id": work["benchmark_id"],
        "case_ids": [work["case_id"]],
        "input_sha256": canonical_sha256([work]),
        "profile_id": profile_id,
        "profile_sha256": canonical_sha256(PROFILES[profile_id]),
        "valid": False,
        "results": [],
        "errors": ["missing provider result"],
        "calls": [],
        "raw_outputs": [],
        "normalization_attempts": [],
        "normalization_events": [],
        "execution_events": [],
        "case_slot_retry_records": [],
        "raw_contract_valid": False,
        "completed_at_utc": utc_now(),
    }


def numeric_usage(calls: list[dict]) -> Counter:
    total = Counter()
    for call in calls:
        usage = call.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                total[key] += value
    return total


def finalize(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_dir)
    verify_local_contract_sources(manifest)
    works_path = resolve_recorded_path(manifest["work_items_path"])
    if sha256_file(works_path) != manifest["work_items_sha256"]:
        raise SystemExit(f"work item hash mismatch: {works_path}")
    works = read_jsonl(works_path)
    work_by_id = {row["case_id"]: row for row in works}
    attempts = load_provider_attempts(manifest)
    profile_id = manifest["identity"]["profile_id"]
    selected_records = []
    retryable_custom_ids = []
    attempt_counts = {}
    for custom_id, request_record in manifest["request_index"].items():
        work = work_by_id[request_record["case_id"]]
        evaluated = [
            evaluated_attempt(entry, work, profile_id)
            for entry in attempts.get(custom_id, [])
        ]
        attempt_counts[custom_id] = len(evaluated)
        valid = [row for row in evaluated if row["valid"]]
        selected = valid[-1] if valid else evaluated[-1] if evaluated else missing_attempt(work, profile_id)
        selected["provider_attempt_count"] = len(evaluated)
        if len(evaluated) > 1:
            selected["prior_provider_attempts"] = evaluated[:-1]
        selected_records.append(selected)
        if not selected["valid"]:
            retryable_custom_ids.append(custom_id)
    record_by_case = {row["case_ids"][0]: row for row in selected_records}
    ordered_records = [record_by_case[work["case_id"]] for work in works]
    rows = case_results(ordered_records, work_by_id)
    out = resolve_recorded_path(manifest["output_report_path"])
    batch_records = out.with_suffix(".batches.jsonl")
    case_records = out.with_suffix(".cases.jsonl")
    write_jsonl(batch_records, ordered_records)
    write_jsonl(case_records, rows)
    calls = [call for record in ordered_records for call in record.get("calls", [])]
    usage = numeric_usage(calls)
    profile = PROFILES[profile_id]
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    estimated_cost = (
        input_tokens * profile["batch_input_usd_per_million_tokens"]
        + output_tokens * profile["batch_output_usd_per_million_tokens"]
    ) / 1_000_000
    suite = audit_suite()
    benchmark_ids = manifest["identity"]["benchmarks"]
    full_count = sum(suite["benchmarks"][item]["cases"] for item in benchmark_ids)
    valid_count = sum(row["valid"] for row in ordered_records)
    normalized_count = sum(
        bool(row.get("valid") and row.get("normalization_events"))
        for row in ordered_records
    )
    normalization_rate = normalized_count / len(ordered_records) if ordered_records else 0.0
    all_valid = valid_count == len(ordered_records)
    finalize_state_path = args.run_dir / "finalize-state.json"
    manifest["state"] = "finalized"
    manifest["last_finalize_state_path"] = relative_to_root(finalize_state_path)
    manifest["retryable_count"] = len(retryable_custom_ids)
    save_manifest(args.run_dir, manifest)
    report = {
        "schema_version": "1.1.0",
        "completed_at_utc": utc_now(),
        "status": (
            "complete" if len(rows) == full_count and all_valid
            else "complete_with_fail_closed_batches" if len(rows) == full_count
            else "sample_complete" if all_valid
            else "sample_complete_with_fail_closed_batches"
        ),
        "resume_claim_ready": (
            set(benchmark_ids) == set(DEFAULT_INPUTS)
            and len(rows) == suite["public_case_count"] == 10_000
            and suite["status"] == "machine_validated_ai_spot_audit_passed"
            and all_valid
            and normalization_rate <= MAX_NORMALIZED_BATCH_RATE
        ),
        "identity_sha256": manifest["identity_sha256"],
        "identity": manifest["identity"],
        "case_count": len(rows),
        "batch_count": len(ordered_records),
        "valid_batch_count": valid_count,
        "raw_contract_valid_batch_count": sum(
            bool(row.get("raw_contract_valid")) for row in ordered_records
        ),
        "deterministically_normalized_batch_count": normalized_count,
        "case_slot_retry_batch_count": 0,
        "deterministic_normalization_rate": normalization_rate,
        "case_slot_retry_batch_rate": 0.0,
        "contract_guardrails": {
            "max_deterministic_normalization_rate": MAX_NORMALIZED_BATCH_RATE,
            "normalization_rate_pass": normalization_rate <= MAX_NORMALIZED_BATCH_RATE,
            "case_isolation": True,
        },
        "fail_closed_case_count": sum(row["fail_closed"] for row in rows),
        "metrics": aggregate(rows),
        "operation": {
            "provider": "openai",
            "api": "responses_batch",
            "logical_cases_per_request": 1,
            "call_count": len(calls),
            "submitted_shard_count": sum(
                bool(row.get("batch_id")) for row in manifest["shards"]
            ),
            "usage": dict(sorted(usage.items())),
            "estimated_batch_cost_usd": estimated_cost,
            "pricing_snapshot_utc": profile["pricing_snapshot_utc"],
        },
        "artifacts": {
            "run_manifest_path": relative_to_root(manifest_path(args.run_dir)),
            "run_manifest_sha256": sha256_file(manifest_path(args.run_dir)),
            "batch_records_path": relative_to_root(batch_records),
            "batch_records_sha256": sha256_file(batch_records),
            "case_records_path": relative_to_root(case_records),
            "case_records_sha256": sha256_file(case_records),
            "case_results_canonical_sha256": canonical_sha256(rows),
        },
        "provenance": {
            "public_annotations": "inherited_from_pinned_publishers",
            "local_adapter_review": "ai_agent_not_human",
            "locally_human_reviewed_public_cases": 0,
        },
        "limitations": [
            "All public inputs are development sets and may be present in model training data.",
            "QA correctness uses deterministic exact match/token F1, not an LLM judge.",
            "Long documents are compressed with the frozen query-only lexical selector.",
            "Fail-closed missing, provider-error, or contract-invalid cases are scored as abstentions.",
            "Prompt-token shard counts are conservative tokenizer-free estimates; actual provider usage is reported separately.",
        ],
    }
    write_json(out, report)
    atomic_write_json(
        finalize_state_path,
        {
            "completed_at_utc": utc_now(),
            "identity_sha256": manifest["identity_sha256"],
            "retryable_custom_ids": retryable_custom_ids,
            "retryable_count": len(retryable_custom_ids),
            "provider_attempt_counts": attempt_counts,
            "report_path": relative_to_root(out),
            "report_sha256": sha256_file(out),
        },
    )
    print(
        json.dumps(
            {
                "report": str(out),
                "case_count": len(rows),
                "valid": valid_count,
                "fail_closed": len(rows) - valid_count,
                "retryable": len(retryable_custom_ids),
                "estimated_batch_cost_usd": estimated_cost,
            },
            indent=2,
        )
    )


def prepare_retry(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_dir)
    verify_local_contract_sources(manifest)
    state_path_value = manifest.get("last_finalize_state_path")
    if not state_path_value:
        raise SystemExit("finalize the run before preparing retries")
    state = json.loads(resolve_recorded_path(state_path_value).read_text(encoding="utf-8"))
    retry_ids = state.get("retryable_custom_ids", [])
    if not retry_ids:
        print("no retryable custom_ids")
        return
    original_requests = {}
    for shard in manifest["shards"]:
        if shard["generation"] != 0:
            continue
        for request in read_jsonl(verified_request_path(shard)):
            original_requests[request["custom_id"]] = request
    missing = set(retry_ids) - set(original_requests)
    if missing:
        raise SystemExit(f"cannot find original requests for {len(missing)} IDs")
    generation = max(row["generation"] for row in manifest["shards"]) + 1
    limits = manifest["identity"]["shard_limits"]
    planned = shard_requests(
        [original_requests[custom_id] for custom_id in retry_ids],
        max_requests=limits["max_requests"],
        max_file_bytes=limits["max_file_bytes"],
        max_estimated_prompt_tokens=limits["max_estimated_prompt_tokens"],
    )
    existing_count = len(manifest["shards"])
    for index, shard_plan in enumerate(planned):
        path = args.run_dir / "requests" / (
            f"generation-{generation:02d}-shard-{index:04d}.jsonl"
        )
        atomic_write_jsonl(path, shard_plan["requests"])
        manifest["shards"].append(
            {
                "generation": generation,
                "shard_index": index,
                "state": "prepared",
                "request_path": relative_to_root(path),
                "request_sha256": sha256_file(path),
                "request_count": shard_plan["request_count"],
                "jsonl_bytes": shard_plan["jsonl_bytes"],
                "estimated_prompt_tokens": shard_plan["estimated_prompt_tokens"],
                "first_custom_id": shard_plan["first_custom_id"],
                "last_custom_id": shard_plan["last_custom_id"],
                "retry_reason": "missing_provider_or_contract_invalid",
            }
        )
    manifest["state"] = "retry_prepared"
    save_manifest(args.run_dir, manifest)
    print(
        f"prepared generation {generation}: {len(retry_ids)} requests in "
        f"{len(manifest['shards']) - existing_count} shards"
    )


def status(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_dir)
    counts = Counter(row.get("batch_status") or row["state"] for row in manifest["shards"])
    print(
        json.dumps(
            {
                "state": manifest["state"],
                "identity_sha256": manifest["identity_sha256"],
                "case_count": manifest["case_count"],
                "shard_count": len(manifest["shards"]),
                "shards_by_state": dict(sorted(counts.items())),
                "retryable_count": manifest.get("retryable_count"),
                "output_report_path": manifest["output_report_path"],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--profile", choices=sorted(PROFILES), default="gpt-5.6-sol-pro-xhigh")
    prepare_parser.add_argument("--benchmarks", default="hotpotqa,natural_questions,fever")
    prepare_parser.add_argument("--top-k", type=int, default=8)
    prepare_parser.add_argument("--max-document-chars", type=int, default=6000)
    prepare_parser.add_argument("--sample-per-benchmark", type=int, default=0)
    prepare_parser.add_argument("--sample-offset-per-benchmark", type=int, default=0)
    prepare_parser.add_argument("--max-requests-per-shard", type=int, default=50_000)
    prepare_parser.add_argument("--max-file-bytes", type=int, default=190_000_000)
    prepare_parser.add_argument("--max-estimated-prompt-tokens", type=int, default=1_000_000)
    prepare_parser.add_argument("--run-dir", type=Path)
    prepare_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    prepare_parser.set_defaults(func=prepare)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--run-dir", type=Path, required=True)
    submit_parser.add_argument("--confirm-paid-submit", action="store_true")
    submit_parser.add_argument("--max-shards", type=int, default=1)
    submit_parser.set_defaults(func=submit)

    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("--run-dir", type=Path, required=True)
    sync_parser.add_argument("--watch", action="store_true")
    sync_parser.add_argument("--poll-seconds", type=int, default=60)
    sync_parser.set_defaults(func=sync)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-dir", type=Path, required=True)
    finalize_parser.set_defaults(func=finalize)

    retry_parser = subparsers.add_parser("retry")
    retry_parser.add_argument("--run-dir", type=Path, required=True)
    retry_parser.set_defaults(func=prepare_retry)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-dir", type=Path, required=True)
    status_parser.set_defaults(func=status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
