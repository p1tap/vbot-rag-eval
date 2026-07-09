# Vbot RAG Eval Harness

Eval-gated retrieval-augmented generation for the Vbot platform's internal
knowledge base. Retriever and prompt changes go through the **same
eval-gated promotion CI** as the [model gateway](https://github.com/p1tap/vbot-model-gateway)
and [backbone fine-tunes](https://github.com/p1tap/vbot-model-lab) — third
system under the gate.

**This is not another LangChain demo.** It's a RAG service whose config
changes (chunking, k, the answer prompt) are promotion-gated by frozen
eval metrics: retrieval recall and MRR (deterministic, free), plus
LLM-judged correctness, faithfulness, and abstention behavior. CI runs
the gate with no GPU, no API keys, no LLM calls — the same gate-on-artifact
pattern used across the Vbot MLOps platform.

## Architecture

```
corpus/*.md ──► ingest.py ──► section-aware chunking ──► e5-small-v2 embeddings ──► numpy index
                                  (heading-keyed)              (local, no API)

query ──► embed_queries() ──► cosine similarity ──► top-k chunks ──► LLM answer ──► judge
              (E5 "query:" prefix)                                    (Llama 3.1 8B     (Gemini Flash-Lite
                                                                       via OpenRouter)    different family)
```

**Key design decisions:**
- **Heading-keyed chunks** — golden set labels reference `doc::heading-slug`,
  not chunk indices, so re-chunking never invalidates relevance labels
- **Asymmetric embeddings** — E5's `query:`/`passage:` prefixes, mean-pooled
  + L2-normalized by hand (no sentence-transformers wrapper)
- **Cross-family judge** — generator is Llama, judge is Gemini; same-family
  judging inflates scores
- **Abstention contract** — out-of-corpus questions must be refused with a
  fixed string; the eval scores both false-negatives and false-positives

## Baseline Metrics

| Metric | Baseline | Config |
|--------|----------|--------|
| recall@k | 0.974 | chunk_max=180, overlap=40, k=4 |
| MRR | 0.844 | e5-small-v2 embeddings |
| answer_rate | 1.000 | Llama 3.1 8B (OpenRouter) |
| correctness | 0.926 | 49 questions (38 answerable, 11 abstention) |
| faithfulness | 0.926 | Judge: Gemini 2.5 Flash-Lite |
| abstention | 0.909 | — |

Golden set: 49 hand-written questions covering 3 corpus documents
(gateway README, backbone fine-tune plan, backbone findings). ~10
out-of-corpus questions that the system must refuse — the RAG equivalent
of the calibration lane in the fine-tune eval.

## Promotion Gate

The gate (`scripts/compare_gate.py`) compares a candidate eval-report
against the committed baseline:

| Exit Code | Decision | Meaning |
|-----------|----------|---------|
| 0 | **PROMOTE** | No metric regressed beyond tolerance |
| 1 | **NEEDS_REVIEW** | Soft metric slipped within review band |
| 2 | **REJECT** | Guardrail metric regressed |

**Guardrail metrics** (regression → REJECT): `correctness`, `abstention`
— did we answer right, and did we refuse the unanswerable. Same
safe-and-correct guarantee the fine-tuning gate applies to calibration +
general quality.

**Soft metrics** (small dip → REVIEW, large dip → REJECT): `recall@k`,
`MRR`, `faithfulness`, `answer_rate`.

**Noise floor**: LLM-judged metrics are not run-to-run deterministic even
at temperature 0 — the same config was measured flipping one abstention
item between runs (provider-side inference varies). On an 11-item lane a
single flip moves the metric by 0.091, so the effective tolerance is
`max(tol, 1.5 / lane_size)` per metric: a regression only counts when it
exceeds one item's worth of noise.

### Demo PRs

Two demo branches demonstrate the gate in action:

1. **`demo/improve-recall`** — Smaller chunks (90 words, overlap 25)
   reduce topic dilution. Fixes q13 (fact previously diluted in an
   oversized section), MRR 0.844→0.890, recall holds at 0.974. Gate
   decision: **PROMOTE**.

2. **`demo/reject-low-k`** — `TOP_K=1` looks like a latency win but
   collapses retrieval (recall 0.974→0.737) and with it the correctness
   guardrail (0.926→0.726). Notably, faithfulness *rises* to 0.979:
   starved of context the system abstains rather than hallucinates — a
   faithfulness-only eval would have promoted this change. Gate
   decision: **REJECT**.

## Quick Start

```bash
# 1. Build the embedded index (local E5, no API key needed)
python scripts/build_index.py

# 2. Query interactively
python scripts/query.py "What is the gateway's failover ladder?"

# 3. Run the full eval (needs OPENROUTER_API_KEY for generation + judge)
export OPENROUTER_API_KEY=sk-or-...
python scripts/run_eval.py

# 4. Retrieval metrics only (free, deterministic, no API key)
python scripts/run_eval.py --no-llm

# 5. Compare against baseline
python scripts/compare_gate.py
# exit 0 = PROMOTE, 1 = NEEDS_REVIEW, 2 = REJECT

# 6. After a successful eval, promote the baseline
python scripts/promote_baseline.py
git add baselines/baseline.json eval-report.json
git commit -m "promote baseline: <describe what improved>"
```

## Serving Path

The answer model is routed through an OpenAI-compatible endpoint. By
default it hits OpenRouter directly; set `RAG_LLM_BASE_URL` to route
through the [Vbot Model Gateway](https://github.com/p1tap/vbot-model-gateway)'s
budget-capped virtual key for per-key spend tracking:

```bash
export RAG_LLM_BASE_URL=http://127.0.0.1:4000/v1
export RAG_LLM_KEY=sk-vbot-rag-...   # gateway virtual key
```

This gives the RAG service its own spend ledger — the same billing/abuse
primitive the gateway provides for any client.

## CI Workflow

`.github/workflows/rag-gate.yml` — triggered on PRs touching `config.py`,
`rag/`, `evals/`, `scripts/`, `corpus/`, or the eval artifacts. Two halves:

1. **Gate** (stdlib, seconds) — compare the *committed* `eval-report.json`
   against the committed baseline across **all** metrics, including the
   LLM-judged ones measured locally before the PR. Exit code gates merge.
2. **Reproducibility check** (CPU torch, ~2 min) — rebuild the index from
   this PR's corpus + config and recompute the deterministic retrieval
   metrics (recall@k, MRR). They must match the committed report: a
   hand-edited report, or a config change whose report was never
   regenerated, fails here (`scripts/check_repro.py`).

The expensive LLM-judge pass runs locally before merging; CI enforces the
gate and verifies the artifact — it never holds API keys or calls an LLM.

## Corpus

| Document | Sections | Source |
|----------|----------|--------|
| `gateway-readme.md` | Gateway architecture, deploy runbook, k8s, GitOps, findings | [vbot-model-gateway](https://github.com/p1tap/vbot-model-gateway) |
| `backbone-plan.md` | Fine-tune plan, dataset spec, training, evaluation | [vbot-model-lab](https://github.com/p1tap/vbot-model-lab) |
| `backbone-findings.md` | SFT sweep findings, DPO ablation results | [vbot-model-lab](https://github.com/p1tap/vbot-model-lab) |

All content is public-safe (scrubbed of secrets, no IPs beyond the
already-public dashboard).

## Project Structure

```
├── config.py                  # all tunable knobs (the file a promotion PR edits)
├── corpus/                    # source documents (markdown)
├── evals/
│   └── golden.jsonl           # frozen golden set (49 questions)
├── baselines/
│   └── baseline.json          # committed baseline metrics
├── rag/
│   ├── embed.py               # E5 embeddings (local, no API)
│   ├── ingest.py              # heading-aware chunking → index
│   ├── retrieve.py            # cosine similarity ranking
│   ├── generate.py            # context-stuffed LLM answer + abstention
│   └── llm.py                 # OpenAI-compatible chat client
├── scripts/
│   ├── build_index.py         # rebuild the numpy index
│   ├── query.py               # interactive retrieval
│   ├── run_eval.py            # full eval → eval-report.json
│   ├── compare_gate.py        # baseline comparison → exit code
│   ├── check_repro.py         # CI: committed report must reproduce
│   └── promote_baseline.py    # adopt current report as new baseline
├── index/                     # (gitignored) built embeddings + chunk metadata
└── .github/workflows/
    └── rag-gate.yml           # CI promotion gate
```
