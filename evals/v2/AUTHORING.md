# V2 case authoring and review guide

Status: development

This guide applies to new V2 cases. The 49 migrated V1 records are a review
queue, not examples of completed V2 annotation.

## Before writing a case

1. Select a real product task or named failure mechanism.
2. Verify that the active corpus manifest can support the intended case.
3. Choose exact source blocks from `corpus/evidence-catalog-v1.jsonl`.
4. Check the current review inventory so the new case adds a new intent,
   evidence cluster, document structure, or failure mechanism rather than a
   cosmetic paraphrase.

Model-generated candidates remain proposals until their source, wording,
claims, and labels are verified. The frozen 100-case seed used human review.
After the owner retired from manual annotation, an AI agent may perform that
verification only with `model_proposed_ai_verified` authoring provenance,
`reviewer_kinds: ["ai_agent"]`, and an explicit note that no new human review
occurred. Such cases never increase the locally human-reviewed count.

## Required annotation

- A single primary lane and any orthogonal tags.
- Answer action and, for non-answer actions, a precise reason.
- Minimum sufficient reference answer.
- Atomic required claims. One claim must not combine separately falsifiable
  facts merely because V1 did so.
- Every acceptable supporting span for each claim that the annotators know
  about.
- Known hard negatives, obsolete/conflicting evidence, or injected evidence
  when the lane requires them.
- Intent and evidence cluster IDs.
- Language, time/authority scope, severity, and optional traffic weight.
- Authoring provenance without unnecessary personal identity.

An answerable case cannot be approved with `legacy_heading` evidence or a
`legacy_unatomized` claim. Select `span` IDs from the evidence catalog and mark
claims `atomic_verified` only after source review.

## Question wording

- Do not copy distinctive source wording unless exact-string retrieval is the
  behavior being tested.
- Preserve genuine real-user wording, including typos and ambiguity, after
  privacy scrubbing.
- Do not add trivial paraphrases to increase the count. Related robustness
  variants share an `intent_family_id` and are not counted as independent.
- State time or authoritative version when the requested fact can change.
- Do not leak the expected evidence ID, retrieval keywords, or answer in test
  metadata visible to the system under evaluation.

## Evidence review

Evidence labels are initially incomplete. Pool top results from materially
different retrievers and adjudicate plausible unlabelled spans before freezing
a release set. Record:

- Required supporting spans.
- Acceptable alternative spans.
- Distractors and hard negatives.
- Contradicted or superseded spans.
- Spans the caller is not authorized to retrieve.

For multi-hop cases, every required claim/hop must have evidence. Finding any
one relevant span is not success.

## Review workflow

1. Author submits a `draft` case without inspecting system retrieval output
   when the case is intended for a sealed partition.
2. Reviewer A checks wording, answer action, claims, evidence, and metadata
   against the corpus.
3. Reviewer B independently checks release/adversarial cases.
4. An adjudicator resolves disagreements; the final record lists pseudonymous
   reviewer IDs and an audit timestamp.
5. A gatekeeper assigns sealed partition membership. Routine developers do not
   receive the case content or per-case result.
6. Once frozen, any content or split change creates a new dataset version and
   migration report.

The same person may author and review development fixtures, but that fact must
be recorded and those fixtures cannot support an independence claim.

For the solo AI-only path, the author/reviewer separation above is unavailable.
The AI agent must disclose that limitation, validate exact evidence IDs and
schema mechanically, avoid relabelling synthetic wording as real-user data,
and use the resulting partition as AI-reviewed evaluation—not independent
human ground truth.

## Commands

```bash
python scripts/build_evidence_catalog.py --check
python scripts/migrate_v1_cases.py --check
python scripts/build_review_inventory.py --check
python scripts/validate_data.py
python -m unittest discover -s tests -p 'test_*.py'
```

`evals/v2/review-inventory.json` is deterministic and should change whenever
case content or review status changes. It is an implementation queue, not an
evaluation score.
