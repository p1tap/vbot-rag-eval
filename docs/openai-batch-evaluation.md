# Direct GPT-5.6 Sol Batch evaluation

## Purpose

This path replaces provider-dependent synchronous generation with OpenAI's
direct Batch API while preserving the frozen public prompt, retrieval inputs,
strict output schema, deterministic scoring, and publisher annotations. It
does not rewrite the historical Qwen or GPT-5.4 reports.

The public suite uses GPT-5.6 Sol as the RAG answer generator. Publisher labels
and evidence—not a second LLM—score all 10,000 public cases. GPT-5.6 Sol may be
tested separately as a Vbot support/coverage judge only against the frozen 175
human calibration labels; it must never be the sole judge of its own answers.

Official API references:

- [GPT-5.6 Sol model](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [Batch API guide](https://developers.openai.com/api/docs/guides/batch)
- [Reasoning models and pro mode](https://developers.openai.com/api/docs/guides/reasoning)
- [API pricing](https://developers.openai.com/api/docs/pricing)

## Safety and reproducibility properties

- `prepare`, `status`, `finalize`, and `retry` are local operations. Preparation
  requires no API key and makes no paid call.
- `submit` refuses to run without `--confirm-paid-submit`.
- The API key is read only from `OPENAI_API_KEY`; it is never written to an
  artifact.
- Each logical case is one `/v1/responses` request with a stable `custom_id`.
  Provider output order is ignored.
- Input files stay below 50,000 requests and 200 MB. The lower configurable
  prompt-token limit is a conservative queue guard.
- Request JSONL, work items, batch metadata, output JSONL, and error JSONL are
  retained with SHA-256 hashes.
- A run refuses changed runner/transport source code, changed work items, or
  changed provider files.
- Finalization fails missing, provider-error, and contract-invalid cases closed.
- `retry` creates a new generation containing only missing or invalid IDs.
- The default paid submit sends one shard at a time so a Tier 1 account does not
  accidentally exceed its 1.5M-token Batch queue. Higher tiers can deliberately
  raise `--max-shards`.

Prompt-token shard counts are deliberately conservative tokenizer-free
estimates. Provider-reported input and output usage remains the authoritative
usage record. The report's dollar figure is an estimate from the pricing
snapshot frozen in the profile.

## Reasoning profiles

Reasoning mode and reasoning effort are independent in GPT-5.6. The roster is:

| Profile | Mode | Effort | Role |
|---|---|---|---|
| `gpt-5.6-sol-standard-high` | standard | high | quality/cost control |
| `gpt-5.6-sol-standard-xhigh` | standard | xhigh | deep-effort control |
| `gpt-5.6-sol-pro-high` | pro | high | pro-mode control |
| `gpt-5.6-sol-pro-xhigh` | pro | xhigh | maximum-quality candidate |

Do not assume the most expensive profile wins. Select it only if the fixed
bakeoff demonstrates a meaningful quality or guardrail benefit.

## Lifecycle

The example below prepares 30 cases: 10 from each benchmark. Change the run and
report names for every profile so identities never collide.

```powershell
python scripts/benchmarks/run_openai_batch_end_to_end.py prepare `
  --profile gpt-5.6-sol-pro-xhigh `
  --sample-per-benchmark 10 `
  --run-dir artifacts/openai-batch/gpt56-pro-xhigh-smoke-30 `
  --out reports/public-benchmarks/end-to-end-gpt56-pro-xhigh-smoke-30.json

python scripts/benchmarks/run_openai_batch_end_to_end.py status `
  --run-dir artifacts/openai-batch/gpt56-pro-xhigh-smoke-30

# This is the only paid transition.
python scripts/benchmarks/run_openai_batch_end_to_end.py submit `
  --run-dir artifacts/openai-batch/gpt56-pro-xhigh-smoke-30 `
  --confirm-paid-submit

python scripts/benchmarks/run_openai_batch_end_to_end.py sync `
  --run-dir artifacts/openai-batch/gpt56-pro-xhigh-smoke-30 `
  --watch --poll-seconds 60

python scripts/benchmarks/run_openai_batch_end_to_end.py finalize `
  --run-dir artifacts/openai-batch/gpt56-pro-xhigh-smoke-30
```

If finalization reports failures:

```powershell
python scripts/benchmarks/run_openai_batch_end_to_end.py retry `
  --run-dir artifacts/openai-batch/gpt56-pro-xhigh-smoke-30
python scripts/benchmarks/run_openai_batch_end_to_end.py submit `
  --run-dir artifacts/openai-batch/gpt56-pro-xhigh-smoke-30 `
  --confirm-paid-submit
```

Then sync and finalize again. Valid completed IDs are never resubmitted.

## Staged experiment

1. Run one HotpotQA case on each of the four profiles as a paid schema smoke.
2. Expand passing profiles to the same 30 cases (10 per benchmark).
3. Run the same frozen 300 cases (100 per benchmark) for the profile bakeoff.
4. Select by macro joint correctness plus per-benchmark guardrails, not the
   macro average alone.
5. Confirm the winner on a disjoint 999-case set with
   `--sample-per-benchmark 333 --sample-offset-per-benchmark 100`.
6. Prepare the full 10,000 with both sample arguments at zero. Submit only as
   many token-aware shards as the account's active Batch queue can hold.

The 300-case comparison must explicitly reject a candidate that improves macro
quality while materially regressing Natural Questions, citation precision,
fail-closed rate, or any other frozen guardrail. This preserves the lesson from
the rejected GPT-5.4 pilot.

## Claim language

Accurate portfolio language after the full run passes is:

> Evaluated an evidence-citing RAG pipeline on 10,000 publisher-human-annotated
> public benchmark cases, with hash-bound resumability, case-isolated retries,
> deterministic scoring, and a separate frozen Vbot domain evaluation.

Do not say that the project owner personally reviewed 10,000 cases. The local
human-reviewed Vbot set contains 100 cases and the judge calibration contains
175 human decisions. Public annotations were inherited from the benchmark
publishers and are reported with that provenance.
