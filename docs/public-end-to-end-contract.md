# Public end-to-end generation contract

## Scope

The public runner evaluates three publisher-human-annotated development sets:
HotpotQA distractor, Natural Questions, and FEVER. It uses task-correct
retrieval scopes and a shared evidence-citing batch response. This track is
separate from the 100-case Vbot owner-reviewed domain set.

## Wire IDs and source identity

Benchmark IDs can be 40–60 characters long and have no semantic value to the
generator. Contract version `public-rag-batch-1.2.0-local-case-slots`
introduced ordered batch-local wire IDs (`C1` through `Cn`); current contract
`public-rag-batch-1.4.0-observed-serialization` retains them. Every batch
artifact records the exact slot→source-ID mapping. Before scoring, result IDs
are restored by position; the promotion audit replays and verifies the mapping,
result order, uniqueness, and complete 10K universe.

This design removed a measured long-ID transcription failure class and reduced
300-case pilot completion tokens from roughly 18.4K to 12.0K. It does not alter
predictions or citations.

## Accepted deterministic normalization

Raw provider output is retained. A batch is counted as normalized, not raw-
valid, if one of these explicitly enumerated transformations is required:

- remove a complete Markdown fence or stray trailing backtick;
- restore one missing array opener and/or the required `results` wrapper;
- wrap a single complete result;
- restore an omitted empty citation array only for an explicit abstention;
- canonicalize FEVER's exact QA abstention alias `__UNANSWERABLE__` to the
  semantically identical `not_enough_info` label;
- remove exactly one extra trailing closing delimiter after a complete JSON
  value;
- restore a batch-local case ID only when it is one edit from the frozen ID at
  that position and is not any other expected ID;
- regroup a root object only when its key sequence is exactly
  `case_id,prediction,citation_ids` repeated once per expected result.
- wrap comma-separated root result objects only when they parse into exactly
  one object per frozen batch slot; the ordinary field, ID/order, prediction,
  and citation validators still run afterward.
- restore one missing root-object closer only for the exact `{"results":[... ]`
  prefix/suffix shape and only when appending it produces valid JSON;
- split merged result objects only when their preserved duplicate-key sequence
  is exactly `case_id,prediction,citation_ids` repeated, expansion produces
  exactly the frozen batch count, and normal validation passes afterward;
- for Natural Questions only, serialize an observed prediction array to one
  semicolon-delimited string only when it contains 2–20 unique, already-trimmed,
  non-empty plain strings with no nested values, abstention/FEVER labels,
  newlines, or ambiguous semicolons.

No rule invents or edits answer content or citation values. Unknown events,
duplicate/swap ambiguity, incomplete triplets, arbitrary labels, or other
malformed shapes fail closed.

The v2.0.1 300-case pilot completed 75/75 valid batches with two normalized
batches (2.67%). Its first full attempt was stopped and retained after 1,812
cases when a repeated comma-separated-object shape fell outside that frozen
allowlist. Version 1.2.0 adds only the exact repair above. Runner v2.0.3 also
binds its own source-file SHA-256 into the checkpoint identity, so code drift
cannot resume an older run. Both changes require a fresh pilot and full run
identity. The confirmation guardrail remains a maximum 5%;
the full report is not promotion-eligible if any batch fails or if the
normalized rate exceeds that bound.

The fresh v2.0.3 pilot then passed 75/75 batches and 300/300 cases with zero
fail-closed cases. It used two allowed repairs (2.67%), reproduced 34.67% macro
joint correctness, and bound runner source
`9f52a0b95f735f8f1997c054c15208374405e2de3f275b0e733c32315c9b2dc0`.

Runner v2.1.0 then completed a full 10,000-case diagnostic with 2,497/2,500
valid batches, 20 isolated slot-retry batches, and 23 normalized batches. It
was rejected because three Natural Questions batches returned a prediction as
an array of non-empty strings, producing 12 fail-closed cases. The same run
also exposed four exact merged-result serialization cases in FEVER, one with a
missing root closer. Contract 1.4 adds only those fully observed structural
repairs. The diagnostic report, raw outputs, and hash-bound rejection record
remain preserved; runner v2.2.0 requires fresh probes, pilot, and full-run
identity.

## Resumability and audit

Every batch stores input, prompt, response-format, profile, and run-identity
hashes plus raw output, calls, usage, latency, validation errors, and
normalization events. Checkpoints reject incompatible run identities. Final
promotion re-hashes batch/case artifacts and verifies:

- all 10,000 source IDs appear exactly once;
- all batches are valid with zero fail-closed cases;
- reported raw/normalized counts match replayed batch records;
- the response-contract version and normalization allowlist are exact;
- normalized batch rate is at most 5%;
- public provenance declares zero locally human-reviewed public cases.

## Rejected diagnostics

Partial and failed experiments remain under `reports/public-benchmarks` rather
than being rewritten:

- the paid GPT-5.4 full attempt stopped when OpenRouter returned HTTP 402;
- a shorter task-specific Qwen system prompt destabilized structured output;
- the first local full attempts exposed long benchmark-ID transcription.
- the v2.0.1 local full attempt exposed an unlisted comma-separated-result
  serialization shape and was stopped rather than silently promoted.
- the v2.1.0 local full diagnostic completed all 10,000 cases but was rejected
  after three Natural Questions string-array predictions failed closed.

These artifacts are diagnostic and never satisfy `resume_claim_ready`.
