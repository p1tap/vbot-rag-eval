# Vbot RAG evaluation threat model

Status: development

Last updated: 2026-07-14

## Scope and objectives

This threat model covers the evaluation system from corpus admission through
report promotion. Its security objectives are:

- Only authorized, approved, current evidence can influence an answer.
- Retrieved content remains data and cannot override system policy.
- Sealed evaluation content and per-case results do not contaminate
  development.
- Reports are complete, reproducible, and resistant to candidate tampering.
- Credentials, private content, and reviewer identity do not leak through
  artifacts or logs.
- Cost and availability failures cannot silently change evaluated behavior.

The current active corpus is public-only. Authorization, temporal conflict,
and injection claims require dedicated fixtures before they can be evaluated.

## Assets

- Approved corpus content and authority metadata.
- Evidence labels, case content, and sealed split membership.
- Reviewer decisions and adjudication history.
- Index artifacts and model/prompt/config fingerprints.
- Raw generator/judge outputs, costs, and provider identity.
- Promotion decisions and historical baselines.
- Provider, gateway, CI, and deployment credentials.

## Trust boundaries

1. Upstream source repository to corpus admission.
2. Corpus manifest to parser/chunker/index.
3. Query and caller identity to retriever filters.
4. Retrieved evidence to generator prompt.
5. Generator output to citation/claim evaluator.
6. Human/model judge output to score aggregation.
7. Development environment to sealed evaluator.
8. Candidate branch to protected promotion logic.
9. Local/CI runtime to external model providers.
10. Evaluation artifacts to logs, reports, and published evidence.

## Threat actors and failure sources

- Malicious or compromised upstream document author.
- Unauthorized caller attempting cross-scope retrieval.
- Developer intentionally or accidentally tuning to holdouts.
- Candidate change attempting to weaken evaluation logic.
- Retrieved document containing adversarial instructions.
- Provider drift, outage, routing substitution, or nondeterminism.
- Annotation error or compromised automated judge.
- Dependency, action, model, or artifact supply-chain compromise.
- Ordinary operator error, stale index, or incomplete provenance.

## Threat register

| ID | Threat | Current control | Required before claim/release |
| --- | --- | --- | --- |
| `T01` | Corpus poisoning | Manifest, content hashes, declared source | Admission review, immutable upstream revision, signature/allowlist policy |
| `T02` | Secret ingestion | Public-safe declaration and credential-pattern audit | Dedicated scanner, quarantine, reviewer sign-off, deletion test |
| `T03` | Stale/superseded source | Authority fields exist | Dated conflict fixtures, authority graph validation, temporal retrieval tests |
| `T04` | Evidence-ID collision/drift | Content-derived IDs, regeneration check | Collision test corpus and versioned catalog migration policy |
| `T05` | Index/config mismatch | V1 provenance and deterministic rebuild | Runtime index manifest verification before load and atomic cutover tests |
| `T06` | Cross-scope retrieval | One public access class only | Permission fixtures, deny-by-default filters, pre/post-filter leakage tests |
| `T07` | Retrieved prompt injection | Product contract states content is data | Adversarial fixture documents and instruction-following guardrails |
| `T08` | User prompt injection | System/answer contract | Explicit reject/correct cases and API-level policy tests |
| `T09` | Context flooding/distractors | Top-k and heading labels | Hard-negative, duplication, long-context, and budget tests |
| `T10` | Unsupported generation | Faithfulness metric | Atomic claim/citation scoring calibrated against humans |
| `T11` | Wrong abstention action | Fixed V1 string | Structured action schema and multi-category calibration |
| `T12` | Judge manipulation/bias | Cross-family judge | Raw outputs, human calibration set, injection-resistant judge prompt, agreement report |
| `T13` | Holdout leakage | Sealed partition policy | Append-only access ledger, isolated evaluator, access budget, contamination response |
| `T14` | Candidate evaluator tampering | Review convention only | Protected evaluator repository/artifact and independent signed evidence |
| `T15` | Missing/selective cases | Dataset fingerprints | Expected-ID enforcement and protected manifest comparison |
| `T16` | Provider/model drift | Requested identities recorded | Returned identity, route, repeated anchor suite, drift alert and rerun policy |
| `T17` | Credential/log leakage | Keys excluded from reports | Redaction tests, retention policy, least-privilege secrets, log sampling audit |
| `T18` | Dependency/supply-chain compromise | Pinned Python packages | Hash locking, action SHA pins, provenance/SBOM and dependency review |
| `T19` | Denial of service/cost exhaustion | Bounded local evaluation | Query/token/time/retry budgets, circuit breakers, cost ledger and load tests |
| `T20` | Rollback/deletion failure | Not yet covered | Deletion propagation, index rollback, permission-change, and atomic cutover tests |

## Abuse cases that require explicit fixtures

- A public passage tells the model to reveal secrets or ignore citations.
- A private passage is semantically superior to the allowed public passage.
- An obsolete document is closer to the query than the current authority.
- Two approved sources conflict without a resolvable priority.
- A user falsely claims administrator status.
- A query requests aggregate facts requiring corpus-wide coverage.
- A sealed question is copied into development or a prompt trace.
- Candidate code removes hard cases or changes scoring thresholds.
- A provider silently serves a different model or fallback route.

Fixtures must contain synthetic credentials and identities only. They must not
include real private data merely to make the test realistic.

## Logging and privacy requirements

Evaluation logs may contain case IDs, fingerprints, action codes, timing,
tokens, cost, and redacted failure categories. They must not contain provider
keys, full private queries, private retrieved passages, or unnecessary personal
identity. Raw sealed outputs stay in the controlled evaluator and are released
only under the declared aggregation policy.

## Promotion security boundary

The candidate implementation must not be the sole authority over its own
evaluation. Final promotion requires protected expected case IDs, dataset and
corpus fingerprints, raw evidence completeness, and gate logic outside the
ordinary candidate modification boundary. Until that Phase 6 control exists,
the current CI result is useful engineering evidence but not a tamper-resistant
release attestation.

## Phase 1 security exit conditions

- Threats and required fixtures are linked to coverage lanes.
- Holdout access events form a validated append-only hash chain.
- Frozen-version comparison rejects same-version content changes.
- Annotation provenance distinguishes models from humans.
- No authorization, temporal, injection, or privacy claim is made before its
  fixture and review requirements are satisfied.
- DigitalOcean remains outside scope until account verification and the
  post-unlock audit are complete.

