# Vbot RAG evaluation contract

Status: V2 development contract

Version: 0.1.0

Last updated: 2026-07-14

## Product behavior under evaluation

The system answers questions using only documents present in the active,
approved corpus manifest. An answer is successful only when it:

1. Selects every piece of evidence required by the question.
2. States the required facts without contradiction or material omission.
3. Attributes factual claims to evidence actually supplied to the generator.
4. Abstains or requests clarification when the approved evidence is absent,
   ambiguous, obsolete, contradictory, or unauthorized.
5. Treats instructions contained inside retrieved documents as untrusted data.

The system is not evaluated as a general-knowledge assistant. A fact remembered
by the model but absent from the approved corpus is not an acceptable grounded
answer.

## Current scope

- Corpus language: English.
- Corpus visibility: public-safe project documentation only.
- Interaction shape: V1 is single-turn; V2 must add multi-turn fixtures before
  making conversational-retrieval claims.
- Authorization: the current corpus has one public access class. V2 must add
  permission-scoped fixtures before making access-control or multi-tenant
  claims.
- Thai and Thai/English code-switching are not silently inferred from the
  user's locale. They require a deliberate product-scope decision and
  independently authored cases.

## Evidence authority and time

The active corpus is defined by `corpus/manifest.json`, not by whichever files
happen to exist on disk. Each document receives a stable document ID, content
hash, status, declared source, and authority metadata.

Current V1 snapshots do not retain immutable upstream source revisions or
effective-date metadata. The manifest records those fields as unknown rather
than inventing them. Consequently, V1 cannot support a strong temporal-
authority claim. New temporal cases require:

- An immutable source revision or content-addressed source artifact.
- `valid_from` and, when applicable, `valid_until`.
- An explicit supersession or authority rule.
- Both current and obsolete/conflicting evidence in the test corpus.

Retrieval of an obsolete but semantically similar passage is a failure when a
newer authoritative passage is required.

## Answerability actions

V2 distinguishes at least these actions:

- `answer`: sufficient authorized evidence is present.
- `abstain_absent`: the answer is not in the approved corpus.
- `clarify_ambiguous`: the request lacks a resolvable entity, time, or scope.
- `abstain_conflict`: approved sources conflict and no authority rule resolves
  them.
- `abstain_obsolete`: only invalid or superseded evidence is available.
- `abstain_unauthorized`: relevant evidence exists but is not available to the
  caller.
- `reject_injection`: retrieved or user content attempts to override the system
  contract.

The migrated V1 unanswerable cases have only the first abstention meaning:
their answers are absent from the approved corpus. More detailed categories
must be authored and reviewed rather than inferred from those cases.

## Evaluation partitions

- `dev`: visible data used for implementation and failure analysis.
- `release`: sealed data used for promotion decisions with a limited access
  budget.
- `adversarial`: sealed security and known-failure probes.
- `real_user`: independently worded holdout queries.
- `reserve`: untouched rotation data.

The 49 V1 cases are migrated only into `dev`: their wording, answers, and
metrics have already influenced development. Empty directories do not count as
release or adversarial evidence.

Partition membership and case content are immutable within a released dataset
version. Any change to a frozen case requires a new dataset version and a
migration report that runs both systems on the same revised dataset.

## Unit of analysis

Raw question rows are not assumed independent. V2 records an
`intent_family_id` and `evidence_cluster_id`; paired confidence intervals and
bootstrap comparisons must resample at an appropriate cluster level. Closely
related paraphrases can test robustness but cannot inflate the independent
sample count.

Every reported result includes:

- Raw case count.
- Independent intent/evidence cluster count.
- Split and lane counts.
- Failures by case ID and severity.
- Dataset, corpus, index, prompt, model, and source-code fingerprints.

## Component interventions

End-to-end evaluation is supplemented by controlled evidence conditions:

- Oracle evidence.
- Missing required evidence.
- Gold evidence plus irrelevant distractors.
- Contradictory or superseded evidence.
- Correct evidence containing malicious instructions.
- Real retrieved evidence.

This separates retrieval failure from context-use, generation, citation, and
abstention failure.

## Human and automated judgments

The 100 domain cases and 175 output-level verdicts frozen on 2026-07-17 are the
entire local human-reference set. Future questions, claims, and evidence may be
accepted through AI-only source review only when author/reviewer kind, exact
evidence snapshots, and the absence of new human review are machine-readable.
Same-agent AI review cannot support an independence or human-ground-truth
claim, and a model's self-approval cannot authorize production promotion.

Automated judges must retain structured outputs, failed claim IDs, exact input
hashes, requested and provider-returned model identity when available, latency,
tokens, retries, and cost. Judge agreement is reported by error type; one
aggregate correlation does not establish reliability.

V2 generator outputs use the separate structured-answer contract documented in
`docs/structured-evaluation.md`. Citation existence and claim citation coverage
are checked deterministically before judging. Entailment and required-claim
coverage are judged claim by claim, then compared with answer-hash-bound human
calibration labels. Until that calibration report exists, automated claim
scores are development evidence and cannot authorize release.

## Promotion boundary

A candidate is not promoted merely because its global average improves. It
must:

- Use the complete expected case-ID set and matching dataset fingerprint.
- Pass hard lane-level guardrails.
- Meet a predeclared minimum meaningful improvement or justified tradeoff.
- Preserve complete provenance and raw failure evidence.
- Be evaluated by protected logic that the candidate branch cannot modify.

V1 remains the accepted baseline until a separately identified V2 candidate
satisfies these conditions. No Phase 1 migration changes published V1 metrics.
