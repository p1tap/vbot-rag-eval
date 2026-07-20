# Vbot RAG V2 system card

Status: development evidence complete where cited; no production-accuracy claim

Last updated: 2026-07-20

## System boundary

Vbot RAG V2 answers questions over three manifest-approved project documents.
It performs structure-aware ingestion, pinned E5 retrieval, strict structured
generation, deterministic claim/citation validation, and optional calibrated
claim-support/coverage judging. The same retrieval and generation functions
are used by offline evaluation and the FastAPI service.

The public benchmark runner reuses the evaluation interfaces but not the Vbot
corpus. HotpotQA, Natural Questions, and FEVER each retain their task-correct
retrieval scope and separate metrics.

## Selected components

| Component | Selected implementation | Evidence boundary |
| --- | --- | --- |
| Vbot retrieval | `intfloat/e5-small-v2` at revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`, k=4 | Deterministic local index |
| Public bounded retrieval | Pinned E5, k=8 for generation context | HotpotQA/NQ candidate scopes |
| FEVER retrieval | Two-stage global BM25 over 5,396,106 pinned pages | Dense global FEVER remains unimplemented |
| Local generator | Qwen3.5 9B derived Ollama image `a2845456d7ad` | Scalability fallback, not quality winner |
| Quality pilot generator | GPT-5.4 high, provider-pinned | 300-case pilot only; paid full run incomplete |
| Judge policy | `v4-primary-failure-cascade` | Calibrated on 175 frozen human tasks |
| Local judge fallback | Qwen3.5 9B diagnostic | Same-family, never release-promotable |

Exact local weight and Modelfile provenance is in
`docs/local-model-provenance.md`. Provider calls retain requested/returned
model, provider, reasoning options, hashes, retries, tokens, latency, and
reported cost where available.

## Verified evaluation evidence

- V1 accepted legacy baseline: recall@4 0.974, MRR 0.844, correctness 0.926,
  faithfulness 0.926, abstention 0.909 across 49 visible cases.
- V2 domain data: 100 owner-reviewed visible development cases across 15 lanes
  and all 19 active corpus headings; 79 answerable and 21 non-answer actions.
- Judge calibration: 175 human verdicts. The selected cascade achieved 76.0%
  overall exact agreement, 86.5% confirmation agreement, zero observed false
  accepts, and 12 fail-closed tasks.
- Public retrieval: pinned E5 improved complete-support recall over BM25 from
  59.30% to 76.17% on HotpotQA and 39.83% to 67.23% on Natural Questions.
  RRF was rejected where evidence completeness regressed.
- Public end-to-end: the promoted specialist report evaluated 10,000/10,000
  cases with zero fail-closed cases and 60.50% strict macro joint
  answer-and-citation correctness: 60.53% HotpotQA, 57.06% Natural Questions,
  and 63.90% FEVER. Task-specific selectors were gated on disjoint confirmation
  data before composition. The independent audit recomputed every metric,
  confirmed 10,000 unique IDs, and passed the declared 60% macro gate.
- Separate single-hop reader: a frozen 1,000-case SQuAD 2.0 oracle-context run
  with a pinned local SQuAD-2.0-fine-tuned extractive reader reached 91.08% F1
  and 87.50% exact match with zero fail-closed cases. Retrieval is excluded and
  the source is the public development set, so this is not comparable to the
  public end-to-end score or a blinded generalization result.
- Domain action policy: 100 development cases reached 100% contract validity,
  88% action accuracy, and zero false answers. The one-time 50-case AI-only
  release run reached 100%, 90%, and zero, respectively.
- Adversarial action policy: the 12 same-agent observed-failure regressions
  rejected the policy at 58.33% accuracy despite zero false answers. Prompt
  injection classification and conservative refusal remain weak.
- Local judge: rejected at 59.43% validity, 74.04% exact agreement, and five
  high/critical false accepts; it is diagnostic only.
- Real-model service: 20/20 Qwen requests succeeded at concurrency one with
  mean 3.31 s, p95 3.39 s, p99 3.44 s, and 0.302 requests/s. The separate final
  container mock test passed 100/100 at 10.25 requests/s and is not inference
  capacity evidence.

## Safety and failure behavior

- Retrieved content is untrusted data and never becomes instructions.
- Only source IDs supplied in the prompt may be cited.
- Invalid JSON, unsupported actions, empty answered claims, missing citations,
  nonexistent citations, and invalid judge groups fail closed.
- Corpus ingestion is denied if the manifest or high-confidence credential
  scan fails.
- The service does not expose raw provider output and does not use questions as
  metrics labels.

## Intended uses

- Regression evaluation for changes to the Vbot corpus, retrieval, structured
  answer contract, or judge policy.
- Reproducible public-benchmark retrieval/generation experiments.
- Local development and bounded service deployment over the approved corpus.
- Portfolio evidence when every number is linked to a retained report.

## Unsupported uses

- General web QA or searching public benchmark questions against the Vbot
  index.
- Treating public publisher annotations as local human review.
- Treating the local same-family judge as independent ground truth.
- Production safety, multilingual, real-user satisfaction, tenant-isolation,
  or long-duration reliability claims.
- Silent ingestion of private material, credentials, or sources outside the
  approved manifest.

## Known limitations

- The Vbot human-reviewed set is visible development data. The separate
  50-case release set was authored before candidate evaluation but was
  authored and reviewed by the same AI agent; it is not independent human
  validation or a production-traffic sample.
- AI-authored/AI-reviewed partitions carry that provenance and do not increase
  the human-review count.
- Public benchmark development sets may have appeared in model training data.
- The local 9B generator is weaker than the GPT-5.4 quality pilot.
- The selected action policy passed development/release safety gates but failed
  the derivative adversarial action gate; it is not promoted as production-safe.
- No genuine real-user partition exists; the final inventory remains blocked
  on `real_user_partition_has_cases` rather than fabricating user evidence.
- The selected cross-family judge depends on external providers. Comparative
  runs pin one provider and fail closed; the operational GLM route explicitly
  orders calibrated Baidu then StreamLake and records the provider actually used.
- Confidence intervals are wide for the 21-case non-answer domain lane.
- The deferred 100-hour endurance/load test is outside this RAG completion
  scope.
