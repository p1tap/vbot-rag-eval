# Retrieval modernization record

Status: experimental candidates evaluated; accepted E5 serving path unchanged

Last updated: 2026-07-26

## Why this work exists

The retrieval stack needed stronger baselines, but adding named architectures
without measuring their downstream effect would weaken the project. Each change
therefore lives behind an explicit experiment path. Component wins do not alter
the online index or earn promotion until answer and citation guardrails pass.

The public benchmark results below use publisher annotations. The Vbot index
comparison uses the reviewed development set. Neither is described as a new
local human annotation effort or a never-observed product holdout.

## Decisions

### Cross-encoder reranking

A pinned `cross-encoder/ms-marco-MiniLM-L6-v2` reranker now operates over a
bounded E5 candidate set. Model weights, revision, candidate depth, precision,
and device are recorded in the policy evidence.

Blanket reranking was rejected for HotpotQA. On 800 development cases at
`k=4`, it improved MRR by 0.30 percentage points but reduced complete-support
recall by 9.25 points and evidence recall by 4.69 points. A single-document
relevance model is not a safe replacement for multi-document chain retrieval.

Natural Questions was a better component match. A transparent confidence rule
was fitted on 800 development cases and frozen before a disjoint 600-case
confirmation run. The confirmation gate passed:

- reranked queries: 69.33%
- reranker pair reduction versus always reranking: 29.95%
- p95 retrieval latency on the recorded RTX 5080 run: 24.81 ms
- any-support recall: +1.67 percentage points versus E5
- complete-support recall: +0.28 points
- evidence recall: +1.06 points
- MRR: +1.76 points

The component evidence is recorded in
[`nq-adaptive-rerank-confirmation-600.json`](../reports/retrieval-modernization/nq-adaptive-rerank-confirmation-600.json).

The end-to-end gate nevertheless rejected promotion. On the same hashed
300-case Natural Questions window, with the same pinned local Qwen generator:

- complete-gold retrieval improved from 65.27% to 67.66%
- joint answer-and-citation correctness fell from 42.67% to 38.00%
- paired transitions were 10 wins, 24 losses, and 266 unchanged
- exact two-sided McNemar p-value: 0.0243
- fail-closed cases remained zero

This is a meaningful downstream regression, not an ambiguous tie. The accepted
E5 path remains unchanged. See
[`nq-adaptive-rerank-end-to-end-comparison-300.json`](../reports/retrieval-modernization/nq-adaptive-rerank-end-to-end-comparison-300.json).

### Provenance-contextual index

The contextual candidate prepends a deterministic header containing only
hash-bound manifest facts: document ID, repository, source path, status,
access class, and section. It does not ask an LLM to invent a summary, and it
keeps raw chunk text separate.

On the 79 answerable cases in the reviewed Vbot development set at unique
heading `k=4`, it improved complete-gold recall from 87.34% to 89.87% and gold
heading recall from 91.98% to 94.30%. MRR fell from 0.8228 to 0.8154. The index
is therefore a development candidate, not a promotion. No disjoint reviewed
confirmation set currently exists.

### Late chunking

Late chunking was tested with a pinned Jina long-context encoder and compared
with a matched pre-chunked Jina control. This avoids conflating encoder-family
quality with the chunking intervention.

At unique heading `k=4`, the late-chunked candidate reached 74.68%
complete-gold recall versus 89.87% for the matched pre-chunk control. MRR fell
from 0.8154 to 0.6762. The candidate is rejected. The current stable overlapping
windows create a poor assembled document view, and the corpus is too small and
short to justify redesigning it around this method.

The matched index evidence is summarized in
[`vbot-index-development-comparison.json`](../reports/retrieval-modernization/vbot-index-development-comparison.json).

### Corrective retrieval

The corrective controller permits one primary attempt and one approved-corpus
fallback. Invalid scores, duplicate documents, timeouts, low confidence, and
rewrite failures terminate in an abstention with no returned hits. It does not
fall back to the live web because that would violate the approved-corpus
product contract.

All eight injected operational scenarios passed. The deterministic report is
[`corrective-retrieval-fault-matrix.json`](../reports/validation/corrective-retrieval-fault-matrix.json).
The controller remains off the production path until confidence thresholds are
calibrated against reviewed answerability outcomes.

## Deferred branches

GraphRAG, RAPTOR, and ColPali are not implemented in the accepted path. The
current product corpus contains three short Markdown documents and 22 chunks;
it has no page images and no benchmark for corpus-wide thematic synthesis.
Adding those systems now would measure implementation effort rather than user
value.

Reconsider them only with an appropriate, licensed evaluation track:

- hierarchical retrieval: long reports or contracts with local-detail and
  global-summary questions
- graph retrieval: entity and cross-document relationship questions
- multimodal retrieval: page-image cases containing tables, charts, diagrams,
  and multi-column layouts

## Next promotion attempt

The next selector should optimize the end-to-end objective rather than
retrieval recall alone. Development labels should distinguish:

1. a reranker gain that changes an incorrect answer into a correct cited answer;
2. additional relevant context that distracts the generator;
3. retrieval changes that induce unnecessary abstention;
4. citation-recall gains that reduce citation precision.

A new policy then needs a disjoint reviewed confirmation partition. Until that
exists and passes both retrieval and end-to-end gates, none of these experiments
belongs in a production or resume claim.
