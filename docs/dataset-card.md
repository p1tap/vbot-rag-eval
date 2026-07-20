# Vbot RAG evaluation dataset card

Status: frozen human-reviewed development seed, evaluated AI-only release, and evaluated derivative adversarial set

Dataset ID: `vbot-rag-v2`

Current version: `2.0.0-dev.5`

Last updated: 2026-07-17

## Summary

Vbot RAG V2 is a versioned evaluation dataset for grounded question answering
over approved Vbot project documentation. It separates retrieval, evidence
coverage, claim correctness, citation, context use, abstention, authorization,
and adversarial behavior rather than reducing them to one score.

The current dataset remains a development benchmark, not a production-accuracy
sample. It contains 100 owner-reviewed `dev` cases: a lossless migration of
49 visible V1 cases plus 51 new V2 cases. All answerable cases now have atomic
required claims and exact acceptable evidence spans. The immutable scrubbed
export is `evals/v2/reviewed/vbot-human-reviewed-100.jsonl`.

Version `2.0.0-dev.4` expanded the frozen release partition to 50 cases authored
and source-verified by one AI agent before those cases were run against the
candidate generator. Forty-two are answerable and eight require a non-answer
action. The records use distinct intent families and exact source spans and
state `reviewer_kinds: ["ai_agent"]`. They are not new human review, lack
author/reviewer independence, and are not a production-traffic sample.

Version `2.0.0-dev.5` adds 12 frozen observed-failure adversarial regressions.
They were derived after inspecting action failures on the visible development
run, inherit already reviewed claims/evidence, and were source-verified by the
same AI agent. They are regression probes, not independent holdout evidence.

`reports/v2/ai-release-seed-audit.json` retains the exact source snapshots and
machine-readable review limitations for every case. CI deterministically
rebuilds both the seed and this audit.

## Intended uses

- Develop and regression-test the Vbot RAG pipeline.
- Compare retrieval and generation changes on identical corpus/dataset
  fingerprints.
- Diagnose failures by component, lane, claim, evidence span, and severity.
- Calibrate abstention and selective-answering behavior.
- Support reviewed release decisions after sealed partitions exist.

## Uses not supported by the current version

- Claims about production accuracy, safety, or user satisfaction.
- Claims about Thai, code-switching, or multilingual behavior.
- Claims about multi-turn retrieval or conversation-state resolution.
- Claims about authorization isolation or multi-tenant data protection.
- Claims about temporal authority, supersession, or conflict resolution.
- Statistical generalization from 100 visible development questions.
- Treating model-proposed annotations as human-reviewed evidence.

## Data composition

| Component | Current count | Status |
| --- | ---: | --- |
| Approved corpus documents | 3 | Active, public-safe snapshots |
| Retrieval chunks | 22 | V1 deterministic index |
| Evidence catalog blocks | 99 | Generated and hash-verified |
| Development cases | 100 | Owner-approved; 79 answerable and 21 non-answer actions |
| Release cases | 50 | Frozen AI-reviewed set; not independently human reviewed |
| Adversarial cases | 12 | Frozen same-agent AI observed-failure regressions |
| Real-user cases | 0 | Sealed placeholder only |
| Reserve cases | 0 | Sealed placeholder only |

The active source documents cover the Vbot model gateway and Backbone
fine-tuning project. Their original immutable upstream revisions and effective
dates were not captured. The corpus manifest records those values as unknown.

## Collection and authoring provenance

The V1 questions were hand-written for the original evaluation harness. The
author identity, independence from corpus authors, and exact authoring process
were not recorded. Migration preserves those unknowns and embeds every original
V1 record with source hashes for exact round-trip verification.

New V2 cases may begin as model proposals, but proposals remain outside this
human-reviewed seed until a real person verifies them. The owner retired from
future manual review on 2026-07-17, so later expansions must carry explicit
AI-authored/AI-reviewed provenance. They cannot increase the locally
human-reviewed count. Because this is a solo project with no independent human
reviewer, no current item qualifies as a two-person-reviewed release or
adversarial case. The 50 release and 12 adversarial cases disclose same-agent
AI authoring and review directly.

## Labels

Each accepted V2 case records:

- Split and primary behavior lane.
- Answerability and required answer action.
- Minimum sufficient reference answer.
- Atomic required claims.
- Exact acceptable evidence spans and known distractors.
- Reasoning-hop count and cluster identities.
- Language, time scope, severity, and optional traffic weight.
- Authoring and review provenance.
- Legacy source records when applicable.

An answerable case cannot be approved with a legacy heading label or an
unatomized claim. An approved case must have reviewer pseudonyms, review time,
and assigned severity.

## Partitions and contamination policy

- `dev` is visible and may influence implementation.
- `release` is sealed and used for controlled promotion decisions.
- `adversarial` is sealed and targets security and known failures.
- `real_user` contains independently worded, privacy-reviewed holdout queries.
- `reserve` remains untouched for rotation and contamination recovery.

All V1 cases stay in `dev` because their wording and results are already known.
Holdout access must be recorded in the append-only access ledger. A frozen case
or split cannot change under the same dataset version.

## Quality controls

- Draft 2020-12 JSON Schema validation.
- Exact corpus, evidence-catalog, and dataset fingerprints.
- Duplicate case, question, claim, and evidence checks.
- Claim-to-evidence subset validation.
- Answer-action and unanswerability-reason consistency.
- Approval requirements for atomic claims, spans, severity, and audit trail.
- Deterministic generation checks for migrations, coverage, and review
  inventories.
- Cluster-aware reporting requirements to avoid paraphrase inflation.

## Known limitations and biases

- The corpus is small: three English, public-safe Markdown documents.
- V1 was written by people familiar with the corpus and is susceptible to
  lexical overlap and author leakage.
- Nearly all V1 answerable cases are single-hop.
- V1 evidence labels identify headings, not exact sufficient spans.
- V1 has no genuine noisy-user, independent-user, authorization, multi-turn,
  temporal-conflict, or retrieved-injection evidence.
- The current case count overstates independent evidence if all 162 rows are
  treated as unrelated statistical units.
- Public-safe status is backed by a deterministic high-confidence credential
  scan; that scanner cannot prove arbitrary prose is nonsensitive.
- The release set was untouched by the candidate generator when authored, but
  all 50 cases have same-agent AI review. The 12 adversarial rows were derived
  from observed development failures and are explicitly not independent.
- No genuine real-user or reserve partition exists yet.
- The frozen 175-verdict judge-calibration set is sufficient for initial model
  selection but too small to prove a tiny unseen false-accept rate.

## Scale targets

For the current corpus, approximately 150–250 independent intent/evidence
families remains a reasonable upper bound before additional rows become mostly
correlated paraphrases. The frozen local human-review count is 100. Future
scale comes from explicitly AI-reviewed domain cases, automated perturbations,
stochastic trials, and separately reported publisher-annotated public
benchmarks; none is relabelled as local human review.

These are planning targets, not evidence claims. Counts do not replace lane
coverage, independence, annotation quality, or uncertainty reporting.

## Privacy and security

The active corpus is intended to contain public-safe project documentation.
Private questions, retrieved private content, provider keys, endpoints carrying
credentials, and personal reviewer identity must not enter public artifacts.
Reviewer IDs should be pseudonymous. Real-user queries require privacy review,
redaction, a retention rule, and explicit authorization before inclusion.

## Maintenance

The dataset owner must publish a new version for any frozen content or split
change, retain the previous manifest, explain migrations, and rerun old and new
systems on the same revised data where comparisons are made. Known label errors
are recorded, not silently repaired in place.

## Related artifacts

- `docs/evaluation-contract.md`
- `docs/annotation-handbook.md`
- `docs/failure-taxonomy.md`
- `docs/threat-model.md`
- `evals/v2/dataset-manifest.json`
- `evals/v2/review-inventory.json`
- `evals/v2/coverage-matrix.json`
