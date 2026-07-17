# Structured answer and claim-evaluation contract

Status: frozen human calibration and completed-profile bakeoff complete;
selected fail-closed cascade frozen

Version: 1.0.0

## Why the V1 judge is insufficient

V1 asks a model for one correctness score and one faithfulness score. That is
useful as a prototype signal, but it cannot identify the failed claim, prove
that a citation existed, or distinguish a missing required fact from an
unsupported extra fact. V2 keeps V1 reproducible and introduces a separate,
fail-closed path.

## Answer boundary

The generator must return exact JSON conforming to
`evals/schema/structured-answer.schema.json`. An `answer` contains ordered
atomic claims (`c1`, `c2`, ...), each citing at least one prompt-local source
ID (`S1`, `S2`, ...). Its `response` field is null; the service constructs the
user-facing text by joining validated claim text in order. This prevents
uncaptured factual prose from bypassing claim checks. Non-answer actions have
no factual claims and carry a non-empty response explaining the limitation.

The report retains both forms of source identity:

- Prompt-local IDs prevent the model from pretending it saw an arbitrary
  corpus record.
- Stable chunk IDs connect each cited source back to the index and corpus.

Before any judge call, deterministic validation rejects malformed JSON,
unknown fields, missing/duplicate/out-of-order claim IDs, uncited claims,
duplicate citations, nonexistent source IDs, and claims attached to a
non-answer action. A nonexistent citation therefore cannot receive a passing
judge score.

Citation existence is not citation correctness. Existence and per-claim
coverage are deterministic; whether the cited text entails the claim is a
claim-level judgment.

## Claim judge boundary

For a valid answered case, independent judge lanes handle support and coverage.
The support lane receives the question, generated atomic claims, and only the
source text cited by each claim. The coverage lane receives the question,
generated response, and human-authored required claims, but no source text.
Generated claims use prompt-local `g*` IDs and required claims use `r*` IDs.
Missing, duplicate, invented, out-of-order, or malformed verdict IDs fail
closed. The evaluator derives the overall pass; neither judge self-awards it.

Every model call retains the exact raw output, input hash, requested and
provider-returned model IDs, response ID, latency, attempt/retry count, token
usage, and provider-reported cost when available. `null` cost means the
provider did not report one; it is never guessed.

## Running the evaluator

The default candidate file contains proposals, not approved gold labels, so a
normal run intentionally stops until reviewed cases are exported:

```powershell
python scripts/run_structured_eval.py --cases path/to/approved-cases.jsonl
```

For plumbing-only development, the override is explicit and permanently marks
the report ineligible for release claims:

```powershell
python scripts/run_structured_eval.py --allow-unreviewed --limit 5
```

`--skip-judge` checks generation, actions, and citation validity but does not
produce claim correctness. It also makes the report release-ineligible.

## Human calibration

Gold-case review and judge calibration are different jobs. The current
100-case review establishes the questions, expected actions, required claims,
and evidence. After the system produces answers, humans separately label
whether each generated claim is supported and whether each required claim is
covered. Those labels conform to
`evals/schema/judge-calibration-label.schema.json` and are bound to the exact
answer hash and eval run ID.

Agreement is reported with exact agreement, Cohen's kappa, confusion counts,
false-accept rates, and every disagreement row:

```powershell
python scripts/report_judge_agreement.py `
  --eval-report reports/v2/structured-eval.json `
  --human-labels path/to/final-human-labels.jsonl `
  --out reports/v2/judge-agreement.json
```

The first calibration batch covers diverse outputs across
lanes, with every suspected failure included instead of sampling only easy
passes. That is enough to expose large judge defects, not enough to establish
a tiny false-pass rate. If zero false accepts occur in 100 independent labels,
the approximate one-sided 95% upper bound is still about 3%. Demonstrating
roughly a 1% bound requires about 300 independent zero-failure labels. The owner
retired from future manual review after freezing this seed. Future calibration
expands through cross-family AI adjudication with explicit AI provenance;
unresolved critical disagreements fail closed and cannot be described as new
human ground truth.

The local review application should receive a dedicated calibration queue
after the first structured run exists; this avoids asking the owner to inspect
model outputs before there are stable output hashes to review.

## Judge calibration and bakeoff implementation plan

Plan approved: 2026-07-14

This is the canonical implementation plan for replacing the uncalibrated
single-judge development path. The four configurations below form a bakeoff
roster, not a four-vote production jury:

1. `deepseek/deepseek-v4-flash` with reasoning effort `high`.
2. `deepseek/deepseek-v4-flash` with reasoning effort `max`.
3. `google/gemini-3.5-flash` with a pinned reasoning configuration.
4. `openai/gpt-5.4` with a pinned reasoning configuration.

The expected steady-state path is a cascade: deterministic checks, an
economical primary judge, a cross-family second opinion for selected cases,
and a strong adjudicator or fail-closed decision for unresolved disagreements. Repeating the
same V4 configuration four times is not an independence strategy and will not
be used as the production ensemble. Same-model repeats are permitted only to
measure label stability.

### Frozen starting point

The first bakeoff reuses `reports/v2/structured-dev-current-k4.json` without
regenerating answers or retrievals. Its calibration workload is:

- 51 total cases and 42 answered cases.
- 80 generated-claim support labels.
- 95 required-claim coverage labels.
- 175 total human verdicts.
- 25 current end-to-end passes, which remain development-only evidence.

The current report, answer hashes, dataset fingerprint, source/index hashes,
prompt hashes, and case IDs must be frozen into the calibration manifest. A
judge experiment is invalid if it silently changes an answer, required claim,
retrieved source, prompt, schema, provider profile, or reasoning effort.

### Phase A: repair the judge contract

- [x] Split generated-claim support and required-claim coverage into distinct
  prompts, schemas, validation paths, metrics, and stored artifacts.
- [x] Give generated claims and required claims unambiguous namespaces such as
  `g1` and `r1` inside judge payloads and verdicts.
- [x] Compare support only against the source IDs cited by the generated claim.
- [x] Compare coverage only against the generated response; source text must
  not be allowed to fill a fact omitted by the response.
- [x] Require `covered` and `contradicted` coverage verdicts to link at least
  one generated claim. Require `missing` to link none.
- [x] Require support verdicts to return an evidence span or explicit
  no-support reason. Validate returned spans against the cited source text.
- [x] Add deterministic exact-value checks for identifiers, paths, URLs,
  versions, counts, percentages, ports, and explicit negation where the gold
  contract permits an exact comparison.
- [x] Add regression fixtures for the observed failure patterns: model size
  versus pair count, generic availability versus access controls, generic
  deployment versus a specific promotion path, and post-retry rate versus the
  retry rule itself.

Implemented 2026-07-14 in `rag/claim_verifier.py`,
`rag/coverage_judge.py`, `rag/judge_contract.py`, and the compatibility
orchestrator in `rag/claim_judge.py`. The evaluator reuses the runtime support
call, so the split contract adds one coverage call rather than duplicating the
support call. Frozen answers and source snapshots were not regenerated.

Phase A exits when malformed or internally inconsistent verdicts fail closed,
the observed empty-link coverage bug is impossible, and the split tasks have
unit tests covering positive, missing, unsupported, and contradicted cases.

### Phase B: make model calls reproducible

- [x] Extend the LLM client to accept and hash reasoning effort, thinking mode,
  provider preferences, and all other behavior-affecting request parameters.
- [x] Use OpenRouter provider filtering that requires strict structured-output
  support. Pin the provider profile used for each calibration run.
- [x] Record requested model, returned model, provider, endpoint or provider
  slug when available, quantization when known, reasoning configuration,
  schema hash, prompt hash, latency, retries, token classes, and cost.
- [x] Increase the judge output budget only after a capability smoke test;
  truncated JSON remains a failed verdict rather than a partial pass.
- [x] Add a preflight command that validates model availability and strict JSON
  Schema behavior without running generation or the full bakeoff.

Completed 2026-07-15: behavior-affecting request options are now included in
the call hash and stored call record. All judge calls set OpenRouter
`provider.require_parameters=true`, pin exactly one provider, and disable
fallbacks. Paid strict-schema probes rejected several endpoints whose catalog
metadata advertised support but whose responses violated the schema. The
frozen profiles are V4 high/max on `akashml/fp8`, Gemini on
`google-vertex/global`, and GPT-5.4 on `openai`. The final unified preflight is
`reports/v2/judge-preflight-pinned.json`; the repeated V4 evidence, including
failed providers, is `reports/v2/judge-v4-provider-probe.json`. GPT-5.4 also
exposed an unnecessary-temperature client bug; optional unsupported sampling
parameters are now omitted rather than defeating `require_parameters`.

Phase B exits when two invocations with the same declared profile have the same
input hash and complete provenance, and an undeclared provider/model fallback
cannot enter a release-eligible report.

### Phase C: build the human calibration set — complete

- [x] Generate an answer-hash-bound calibration queue for all 175 verdicts.
- [x] Add the queue to the existing review application with separate support
  and coverage decisions and no model verdict visible before the human label.
- [x] Include every suspected current judge defect, not only easy agreements.
- [x] Capture reviewer, timestamp, decision, rationale, evidence span when
  applicable, and adjudication state under the existing calibration schema.
- [x] Export immutable JSONL labels and validate every answer hash and claim ID
  before a judge comparison is allowed.
- [x] Create a judge-development slice and a hidden confirmation slice where
  the available sample size permits it.

Completed 2026-07-17: `scripts/build_judge_calibration_queue.py` froze all
175 tasks into `evals/v2/judge-calibration/queue.jsonl` and wrote the source
report, dataset, queue, answer, prompt, model, retrieval, source/index, and task
identity hashes to `evals/v2/judge-calibration/manifest.json`. The queue has 80
support tasks and 95 coverage tasks across 42 answered cases. It contains no
automated verdicts or retrieval scores. Coverage tasks contain no source text.
Cases, rather than individual claims, are deterministically divided into 34
development and 8 hidden-confirmation cases to prevent answer leakage across
slices. The separate `/judge-calibration` workspace now imports the frozen
queue, stores labels in D1, validates exact source quotes and linked claim IDs,
auto-advances, preserves audit history, and exposes a JSONL export. It does not
contain model verdicts or retrieval scores. All 175 human labels are final and
were exported as 42 answer-hash-bound case records in
`evals/v2/judge-calibration/human-labels.jsonl`. The scrubbed export omits the
reviewer's email and uses a stable pseudonym.

Support labels may retain multiple disconnected exact quotes, including spans
from different cited sources. Each span is validated independently against its
citation. Supported and contradicted labels require at least one span;
unsupported labels require none. This prevents compound claims from being
calibrated against a single cherry-picked fragment while preserving older
single-span labels as one-element span arrays.

The 175 verdicts are the complete initial workload, but they do not prove a
tiny population error rate. Confidence bounds must be reported alongside point
estimates, and the calibration set must expand adaptively around observed
failure families.

### Phase D: run the judge bakeoff — complete with one declared roster omission

Completed 2026-07-17. The frozen summary contains the three profiles that
finished their complete 82-group checkpoint universe: DeepSeek V4 Flash high,
Gemini 3.5 Flash high, and GPT-5.4 high. OpenRouter returned HTTP 402 after
DeepSeek V4 Flash max completed 39 of 82 groups, so max is omitted from the
decision summary and retained only as an incomplete diagnostic checkpoint.
The summary records the configured, included, and omitted rosters explicitly.

No single profile met every aspirational eligibility target. Against the
original human labels, final valid-task rates were 77.7% for V4 high, 89.1%
for Gemini, and 93.1% for GPT. Exact agreement on valid tasks was 77.9%, 82.7%,
and 76.1%, respectively. GPT had zero false accepts but was conservative; V4
and Gemini each had one. Invalid groups and unresolved disagreements therefore
cannot be accepted by a single-judge policy.

Each configuration judges the same frozen inputs once. The two leading
configurations then run repeated trials on a fixed judge-killer slice to
measure label instability; repeated same-model votes are not combined as
independent evidence.

Report these metrics separately for support and coverage, then by lane and
severity:

- Exact agreement and Cohen's kappa against human labels.
- False-accept and false-reject counts and rates.
- Confusion matrix, including unsupported versus contradicted confusion.
- Structured-output validity before and after retry.
- Label stability across repeated trials.
- Latency, input/output/reasoning tokens, retry count, and reported cost.
- Every disagreement row with the raw model output and human rationale.

Initial judge-eligibility targets are:

- 100% valid final structured output after bounded retries.
- Zero observed false accepts on critical and high-severity calibration items.
- At most 2% observed false accepts overall in each task.
- At least 90% exact agreement and 0.80 Cohen's kappa where label prevalence
  makes kappa meaningful.
- No hidden provider or model fallback and no missing provenance fields.

These are selection targets, not statistical claims about unseen traffic. A
candidate that misses them may still provide triage evidence but cannot become
the sole release judge.

### Phase E: select and simulate the cascade — complete

Evaluate at least these policies offline against the human labels:

1. Best single judge.
2. V4 `high` primary with Gemini second opinion on critical/high cases and a
   stratified sample of routine passes.
3. V4 `high` primary with Gemini on all primary failures and ambiguous cases.
4. V4/Gemini disagreement routed to GPT-5.4.
5. Any critical disagreement or unresolved adjudication fails closed after
   cross-family GPT adjudication; no human-review queue is created.

The selected policy minimizes false accepts first, then false rejects,
fail-closed volume, latency, and cost. V4 `max` replaces V4 `high` only if the human
comparison demonstrates a material reliability gain. GPT-5.4 is an
adjudicator candidate, not an assumed oracle.

The initial production policy must double-judge every critical case and at
least 10% of routine agreements. Reducing the AI audit rate requires measured
evidence, a documented error bound, and an explicit contract update.

The development-only selection chose `v4-primary-failure-cascade`: V4 high is
the primary; Gemini is consulted on primary failures; GPT adjudicates
unresolved disagreement; missing/invalid automated evidence fails closed. No
future human queue exists. Against the untouched human reference it observed
zero false accepts, 73.2% development agreement, 86.5% held-back confirmation
agreement, and 76.0% overall agreement. Its 12 invalid/unresolved tasks failed
closed.

When paid providers are unavailable, the explicitly selected
`qwen3.5-9b-local-diagnostic` profile can score the frozen 175 tasks or a new
local generation report. It is outside the four-profile bakeoff roster, uses
the pinned local Ollama image, and fails invalid groups closed. Because Qwen
then judges Qwen-family generation, the resulting report is marked
release-ineligible and cannot replace the selected cross-family cascade. See
`docs/local-model-provenance.md`.

The completed local diagnostic confirmed that boundary: only 104/175 tasks
were valid (59.43%), exact agreement was 74.04%, and five high/critical false
accepts occurred. The profile is rejected as an operational judge and remains
diagnostic-only.

A separate three-family consensus audit preserved every human label and wrote
22 explicit AI-only overrides (18 `covered->missing`, two
`supported->unsupported`, and two `contradicted->missing`). Against that
partially circular AI-adjusted reference the same preselected policy reached
86.2% development, 97.3% confirmation, and 88.6% overall agreement with zero
false accepts. Both reports are retained; the adjusted result is diagnostic,
not human-equivalent ground truth.

### Phase F: promotion and continued calibration

- [x] Freeze the winning judge profiles, prompts, schemas, routing policy, and
  frozen-human-label dataset fingerprints.
- [x] Emit a machine-readable bakeoff report and a short Markdown decision
  record explaining why the selected policy won.
- [x] Add CI checks for the frozen judge contract and prohibit silent model,
  provider, prompt, schema, or reasoning changes.
- [ ] Recalibrate after any judge, prompt, schema, corpus, answer format, or
  lane-distribution change.
- [ ] Run the stronger-generator comparison only after the judge calibration
  gate is satisfied.
- [x] Publish the public 10,000-case end-to-end scale claim only after the
  separate runner contract audit passed with zero fail-closed cases.

### Planned artifacts and commands

Implementation should add these artifacts or equivalent names chosen in the
same change. They do not exist merely because they are listed here:

- `evals/v2/judge-calibration/manifest.json`
- `evals/v2/judge-calibration/human-labels.jsonl`
- `reports/v2/judge-bakeoff/<run-id>/`
- `reports/v2/judge-bakeoff-summary.json`
- `reports/v2/judge-bakeoff-decision.md`
- `scripts/build_judge_calibration_queue.py`
- `scripts/run_judge_bakeoff.py`
- `scripts/simulate_judge_cascade.py`

The planned operator flow is:

```powershell
python scripts/build_judge_calibration_queue.py `
  --eval-report reports/v2/structured-dev-current-k4.json

python scripts/run_judge_bakeoff.py `
  --manifest evals/v2/judge-calibration/manifest.json `
  --human-labels evals/v2/judge-calibration/human-labels.jsonl

python scripts/simulate_judge_cascade.py `
  --bakeoff reports/v2/judge-bakeoff-summary.json
```

### Explicit non-goals

- Do not regenerate answers while comparing judges.
- Do not treat four identical same-model calls as four independent judges.
- Do not select a judge from general-purpose benchmark reputation alone.
- Do not tune against the hidden confirmation slice.
- Do not authorize release from an uncalibrated automated score or call an
  AI-only expansion human-reviewed.
- Do not run the expensive public end-to-end evaluation before the judge gate
  and AI-only audit policy are operational.
