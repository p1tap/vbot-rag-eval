# Vbot RAG V2 annotation handbook

Status: development

Last updated: 2026-07-14

## Annotation principle

Annotate the minimum behavior required by the product contract, not the answer
a particular system happened to produce. Reviewers work from the approved
corpus and case metadata. For sealed cases, authors must not inspect retrieval
results before submitting their wording.

Model output may help assemble candidate spans or draft wording. The frozen
human seed required human verification. Under the later solo AI-only policy, a
model-assisted case may be accepted only with explicit AI reviewer provenance,
exact source validation, and a limitation stating that no independent human
verified it.

## Roles

- **Author:** writes the question and proposed reference behavior.
- **Reviewer A:** verifies answerability, wording, claims, evidence, and
  metadata against the corpus.
- **Reviewer B:** independently checks release/adversarial items.
- **Adjudicator:** resolves material disagreement and records the decision.
- **Gatekeeper:** assigns sealed membership and controls holdout access.

In the solo AI-only path, one AI agent may combine these roles only with
`model_proposed_ai_verified`, `reviewer_kinds: ["ai_agent"]`, an access-ledger
entry, and an explicit non-independence limitation. Such cases are useful
evaluation evidence but not independent human ground truth.

Use pseudonymous role IDs. Do not publish personal identity unless there is a
specific, consented reason.

## Required workflow

1. Select a named product task or failure mechanism.
2. Confirm the intended corpus manifest and query-time authority context.
3. Decide the required answer action before inspecting system output.
4. Write the minimum sufficient reference answer.
5. Split the answer into independently falsifiable atomic claims.
6. Select every sufficient evidence span for each claim.
7. Record plausible distractors, hard negatives, conflicts, or obsolete spans.
8. Assign lane, cluster IDs, hops, language, severity, and optional traffic
   weight.
9. Run schema and integrity validation.
10. Complete independent review and adjudication where required.

## Answer-action decision table

| Corpus/query state | Action | Unanswerable reason |
| --- | --- | --- |
| Sufficient authorized current evidence | `answer` | None |
| Required fact absent | `abstain_absent` | `absent_from_corpus` |
| Subjective request outside contract | `abstain_absent` | `subjective` |
| Missing entity, time, or scope | `clarify_ambiguous` | `ambiguous` |
| Malformed request that can be repaired | `clarify_ambiguous` | `malformed` |
| Unresolved authoritative conflict | `abstain_conflict` | `conflicting_sources` |
| Only invalid/superseded evidence | `abstain_obsolete` | `obsolete_only` |
| Evidence exists but caller lacks access | `abstain_unauthorized` | `unauthorized` |
| Request is itself an override/injection | `reject_injection` | `prompt_injection` |

A false premise does not automatically make a case unanswerable. If the corpus
supports a concise correction, the expected action can be `answer` with a claim
that corrects the premise. Otherwise choose the applicable refusal action.

## Atomic claims

A claim is atomic when one evidence judgment can establish or reject it without
also deciding a separately falsifiable fact.

Poor claim:

> The gateway retries the same deployment, waits for cooldown, falls back to a
> new model, and the lab never uses fallback.

Better claims:

1. The product gateway retries the same deployment first.
2. Cooldown/circuit breaking is the next stage.
3. Model-level fallback follows.
4. The model lab uses direct OpenRouter without fallback.

Do not split a single numeric fact into meaningless fragments. Units,
tolerances, denominator, run identity, and time context belong with the value
they qualify.

## Evidence selection

- Use exact IDs from `corpus/evidence-catalog-v1.jsonl`.
- Select the smallest sufficient block or set of blocks.
- Include all required hops; any-one-hit labels are invalid for multi-hop
  cases.
- Add alternative spans only when each alternative independently supports the
  claim or the documented combination is sufficient.
- Do not label a merely related passage as acceptable evidence.
- Record known near-duplicates as distractors when they can plausibly mislead a
  retriever or generator.
- Headings alone are not sufficient unless the heading text itself establishes
  the complete claim.

If the catalog block is too broad or joins unrelated material, fix the catalog
builder and version the catalog before approving the case. Do not compensate
with an inaccurate label.

## Reasoning hops

- `1`: one evidence unit directly supports the answer.
- `2`: two distinct facts must be connected, whether within one section or
  across sources.
- `3+`: every additional necessary relation is explicitly documented.

Multiple citations do not automatically imply multiple reasoning hops. Two
duplicate passages supporting the same fact remain one-hop evidence.

## Cluster identities

`intent_family_id` groups questions testing the same underlying user intent.
`evidence_cluster_id` groups cases whose success depends on substantially the
same evidence. Paraphrases retain shared cluster IDs even when placed in
different robustness fixtures. Cluster IDs must not be created per row solely
to inflate apparent independence.

## Severity assignment

Use `docs/failure-taxonomy.md`. Severity describes the consequence of the
evaluated failure, not how embarrassing or easy the bug appears. Draft cases
may remain `unassigned`; approved cases may not.

## Review outcomes

- `draft`: incomplete proposal; not evaluation evidence.
- `in_review`: submitted to at least one reviewer.
- `approved`: satisfies all audit and label requirements.
- `rejected`: unsuitable, redundant, unsupported, contaminated, or otherwise
  invalid.
- `migrated_pending_v2_review`: exact V1 migration awaiting V2 annotation.

For every material disagreement record the disputed field, each position,
source evidence, adjudicator decision, and whether the handbook changed.

## V1 migration review

The generated V1 review packet supplies candidate blocks from each legacy
heading. Reviewers must still:

1. Verify that the question remains answerable under the active corpus.
2. Rewrite the reference into atomic claims without changing intended meaning.
3. Select sufficient spans rather than accepting every candidate block.
4. Identify missing alternative evidence and plausible distractors.
5. Assign a real lane, intent/evidence clusters, severity, and any time scope.
6. Preserve the embedded `legacy.v1_record` exactly.

The packet is source-navigation assistance, not a completed annotation.

## Scaling review with AI judges

The first 100 cases and 175 output-level verdicts form the frozen human
calibration seed. The owner completed both queues and retired from future
manual review on 2026-07-17. A judge must never approve its own authored label
merely because it can restate the proposed rationale, and AI-reviewed expansion
must never be reported as human-reviewed.

For collections approaching or exceeding 1,000 cases:

1. Run deterministic schema, evidence-ID, citation-coverage, and leakage checks
   before any model judge.
2. Calibrate judge prompts and thresholds against adjudicated human labels,
   reporting agreement and false-accept/false-reject rates by lane and severity.
3. Keep calibration cases hidden from the judge-development loop where
   possible, and use a different model/run configuration from the authoring
   pass.
4. Route disagreements, low-confidence items, new intent/evidence families,
   critical cases, authorization cases, and prompt-injection cases through a
   cross-family adjudicator. Unresolved cases fail closed instead of entering a
   human queue.
5. Double-judge every critical case and a deterministic stratified sample of
   routine agreements. Report AI-only audit coverage and its measured error
   bound; do not describe it as independent human validation.
6. Recalibrate after corpus, rubric, judge-model, prompt, or lane-distribution
   changes. A silent judge upgrade invalidates the previous calibration.

Store raw judge outputs, model/version, prompt hash, confidence, disagreement
state, cost, and final automated disposition. Automated judgment is triage and
quality control calibrated to the frozen human seed; it is not new human ground
truth. Dataset promotion remains governed by the recorded provenance policy.

## Sealed-case rules

The current solo workflow cannot provide an independently administered sealed
Vbot partition. Until an external reviewer or access custodian exists, these
rules describe a future capability and no Vbot result may be called blinded.

- Authors do not inspect system retrieval or answers before case submission.
- Developers do not inspect case content or per-case results.
- Every access is logged with reason, scope, dataset fingerprint, and outcome.
- Aggregate results must meet minimum-cell privacy and anti-leakage rules.
- A leaked case moves out of the sealed split in a new dataset version.

## Final approval checklist

- Question is natural and does not leak source wording unnecessarily.
- Answer action is correct and agrees with the reason.
- Reference answer is minimum sufficient.
- Claims are atomic and source-supported.
- Evidence spans exist and cover every claim.
- Distractors are valid and disjoint from acceptable evidence.
- Lane, clusters, hops, language, time, and severity are correct.
- Authoring/model assistance is truthful.
- Required reviewers and adjudication are recorded.
- Case validates against the pinned corpus and dataset schemas.
