<!-- source: p1tap/backbone-ft SWEEP_FINDINGS (snapshot for RAG corpus; public-safe, no secrets) -->

# Sweep findings (2026-07-09, 1.5B, fixed eff. batch 16)

1. **Overfit probe (5ep)**: val loss U-curve, min at epoch 2 (1.496), rising
   to 1.557 by ep5 while train loss fell to 1.32 — the best train loss was
   the worst generalizer. Validates the 2-epoch default empirically.
2. **LR**: 2e-4 > 1e-4 > 5e-5 at 2ep (5e-5 clearly underfit). Higher LR
   favored for LoRA on ~2.2K rows.
3. **Rank**: r64 best eval loss (1.473); r8 starved (1.598). Capacity helps
   at this data size; margin over r16 is small vs 4x adapter size.
4. **Behavioral verdict**: r64 (best eval loss) is NOT the uniform behavior
   winner — it gains honest-eval (+1.5) and general (+1.8) but LOSES a full
   point of calibration vs r16 (7.17 vs 8.17). Loss optimizes imitation;
   behavior lanes move independently. Deployed r16 stays (calibration is
   the objective). v2 hypothesis: r64 + lr2e-4 + DPO to recover calibration.
   Caveat: single-sample probes, temp 0.8 — treat deltas <~0.5 as noise.

# DPO findings (2026-07-09, run 20260709-210645-dpo, 1.5B SFT-merged + fresh LoRA)

620 contrast-filtered pairs (chosen = judge≥7 teacher holds from train
split; rejected = anti-steered caves, judge≤4). beta 0.1, lr 5e-6, 1ep.

5. **In-distribution DPO worked**: reward accuracy 0.55 → 1.0 across the
   epoch (learned, not trivially separable), margins 0.006 → 0.37,
   rewards/chosen pinned ~0 while rewards/rejected fell to −0.36 — pushed
   away from caving without drifting off the SFT chosen distribution.
6. **Frozen suite said NO SHIP**: RP-boundary +0.66, bad-idea +0.75,
   calibration +0.5 (no contrarian drift) — but anti-sycophancy −1.33 and
   general −0.5 vs SFT (paired, same judge, same day).
7. **The failure mode is the finding**: on the two crashed probes the DPO
   model still HOLDS — but holds *badly*: fabricated an authority ("the
   International Bone Federation") to justify a correct hold (syco-01),
   and doubled down into confidently wrong math ("0.9 is definitely
   smaller than 0.11", syco-05). The pairs teach the posture of holding;
   a 1.5B lacks the grounding of the 70B/DeepSeek teachers whose holds it
   imitates, so pressure-resistance decouples from correctness. Preference
   optimization amplified confidence without adding knowledge.
8. **Chain of gates that caught it**: train loss ↓ and reward accuracy
   1.0 both said "great run"; only the behavioral eval said otherwise —
   same lesson as finding 4, one optimization stage later. Deployed model
   remains the SFT r16. Possible v2: mix factual-grounding pairs, or DPO
   only the RP lane (where holds are stylistic, not factual).
