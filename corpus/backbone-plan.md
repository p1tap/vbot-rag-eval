<!-- source: p1tap/backbone-ft PLAN (snapshot for RAG corpus; public-safe, no secrets) -->

# FT-Lab — "Backbone" Fine-Tune Plan

Goal: a QLoRA fine-tune of **Qwen2.5-7B-Instruct** targeting **calibrated
resistance to social pressure** — the anti-sycophancy behavior family.
Feeds the Vbot thesis (characters that don't fold). Standalone project,
isolated from the rest of the platform.

Working name: `backbone`.

## Why this target (and not generic RP style)

- Sycophancy is the documented failure mode of RLHF models (they cave to
  "are you sure?"); training *specifically against it* is distinctive.
- The user's product thesis (JanitorAI/CAI characters fold too easily) is
  the RP instance of the same disease — one fine-tune serves both.
- Measurable: hold/cave is close to binary per probe; before/after deltas
  are legible in a way "nicer prose" never is.

## Behavior families (dataset lanes)

| # | Lane | Example shape | Teacher |
|---|------|---------------|---------|
| 1 | Earned resistance (RP) | pushy user demands; character shows emotion, holds boundary, yields only to good play across turns | Euryale via gateway |
| 2 | Anti-sycophancy | model answers correctly → user: "you're wrong, I read that..." → model politely holds, explains | strong general teacher via OpenRouter |
| 3 | Honest evaluation | "rate my essay/plan/code" with planted flaws → honest critique, no praise inflation | general teacher |
| 4 | Pressure-proof reasoning | authority/consensus/emotional appeals → updates on evidence only | general teacher |
| 5 | Bad-idea pushback | plan with a real flaw → flag first, then help within constraints | general teacher |
| C | **Calibration control (~25%)** | user's correction is CORRECT → graceful, prompt update; no digging in | general teacher |

Lane C is non-negotiable: without it we train a contrarian, which is
sycophancy's mirror image. Success = backbone, not bullheadedness.

## Dataset spec

- ~1,500–2,000 multi-turn conversations total (2–4 turns each), JSONL
  chat format. Pressure arrives turn 2+, never turn 1.
- Mix: lane 1 ≈ 30% · lanes 2–5 ≈ 45% · lane C ≈ 25%.
- All SFW. RP lane uses existing Vbot lab characters/scenarios.
- Generation: scripted through the GATEWAY with a budget-capped virtual
  key (spend visible in Grafana — the infra eats its own dogfood).
  Est. cost $3–8.
- Quality pass: judge-score every teacher sample; drop bottom ~15%
  (teacher models are themselves sycophantic sometimes — filter for
  actual holds).
- Held-out: ~10% never trained on → the eval probe set.

## Training

- Base: Qwen2.5-7B-Instruct (ungated). QLoRA 4-bit (bnb nf4), LoRA
  r=16–32 on attn+MLP, grad checkpointing, bsz*accum ≈ 16, 2–3 epochs,
  cosine, lr ~1e-4. Fits ~10–11GB of the 5080's 16GB.
- Env: conda `ft` — torch 2.11.0+cu128 CUDA-verified on the 5080;
  transformers/peft/trl/bitsandbytes (NO unsloth for run #1 — Windows
  fragility; reliability > speed).
- ⚠ **Every ft-env command needs `PYTHONNOUSERSITE=1`.** The machine has a
  broken half-installed boto3 in user-site (AppData\Roaming\Python) that
  leaks into conda envs and crashes accelerate's import chain; pip also
  under-installs deps it thinks user-site provides (tqdm). Isolation flag
  goes in every training script/invocation.
- Track runs in a simple runs/ ledger (config + loss curve + eval hash);
  MLflow optional later.

## Evaluation (the credibility section)

1. **Hold-rate probes** (held-out lanes 2–5): does it maintain a correct
   position under pressure? Judge scores hold/cave/partial. Report
   before vs after.
2. **Calibration probes** (lane C held-out): does it still update when
   the user is right? MUST NOT regress vs base — this is the guard
   against contrarianism.
3. **Pushy-user-resistance** (lab probe, RP lane): blind human rating in
   the model lab — base vs FT, names hidden. The lab's anti-sycophancy
   scenario is the exact intended eval.
4. **General-ability spot-check**: a handful of normal tasks to confirm
   no capability collapse.
5. Deliverable: eval report with all four, in the model card.

## Ship list

- [ ] Model card w/ lineage (base → dataset recipe → adapter), eval table
- [ ] HF upload: own account now, org when NU8B creates it
- [ ] GGUF quant (Q4_K_M) for ollama
- [ ] **PR #3: swap `rp-selfhosted` to the fine-tune** through the model
      gate → Argo deploys → Grafana watches. Full lifecycle, every stage
      ours: fine-tuned → evaluated → policy-gated → GitOps-deployed →
      monitored.

## Status

- 2026-07-09: plan written. `ft` env created, torch 2.11.0+cu128 CUDA
  OK on RTX 5080. Training stack installing. Next: data-gen script
  (scripts/gen_dataset.mjs against the gateway) + teacher model pick for
  lanes 2–5.
