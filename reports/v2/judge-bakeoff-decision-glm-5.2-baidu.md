# Judge bakeoff and cascade decision

The calibration reference is the original frozen human-label artifact.

Decision date: 2026-07-19

## Decision

Selected policy: `glm-5.2-xhigh-baidu-primary-failure-cascade`.

The policy was selected using only the frozen development partition. Ranking minimizes high/critical false accepts, total false accepts, false rejects, categorical disagreements, fail-closed decisions, cost, and latency—in that order. The confirmation partition was opened only after the policy identity was fixed.

No future human-review queue is part of this policy. Unresolved or invalid automated adjudication fails closed and is recorded as AI review, never relabeled as human review.

## Policy comparison

| Policy | Dev agreement | Dev false accepts | Confirmation agreement | Confirmation false accepts | Calls | Reported cost |
|---|---:|---:|---:|---:|---:|---:|
| `single-gemini-3.5-flash-high` | 74.6% | 1 | 86.5% | 0 | 82 | $1.094448 |
| `single-glm-5.2-high-baidu` | 77.5% | 1 | 75.7% | 0 | 82 | $0.033574 |
| `single-glm-5.2-xhigh-baidu` | 74.6% | 0 | 86.5% | 0 | 82 | $0.054788 |
| `single-gpt-5.4-high` | 69.6% | 0 | 83.8% | 0 | 82 | $0.922275 |
| `single-v4-flash-high` | 65.2% | 1 | 75.7% | 0 | 82 | $0.022016 |
| `glm-5.2-xhigh-baidu-primary-severity-sample` | 76.1% | 0 | 86.5% | 0 | 173 | $1.311201 |
| `glm-5.2-xhigh-baidu-primary-failure-cascade` | 76.1% | 0 | 86.5% | 0 | 145 | $0.917395 |
| `glm-5.2-xhigh-baidu-secondary-disagreement-adjudicator` | 76.1% | 0 | 86.5% | 0 | 178 | $1.348074 |
| `glm-5.2-xhigh-baidu-conservative-full-panel` | 76.1% | 0 | 86.5% | 0 | 246 | $2.071511 |

## Selected-policy result

- Development: 76.1% agreement; 0 false accepts.
- Hidden confirmation: 86.5% agreement; 0 false accepts.
- Overall: 78.3% agreement; 0 false accepts and 34 false rejects.
- One-sided 95% upper bound on the observed overall false-accept rate: 1.5%.

This calibration set is deliberately failure-rich and small. Its point metrics select an operating policy; they do not prove the same error rate on unseen production traffic.

## Provider fallback qualification

The same xhigh profile was rerun with StreamLake pinned. Both providers
completed 156/175 valid tasks with zero false accepts. On the 153 tasks valid
for both providers, 152 decisions matched (99.35%). StreamLake's mean latency
was 15.99 seconds versus Baidu's 10.12 seconds, so the operational order is
Baidu then StreamLake. Comparative evaluation keeps fallback disabled; the
ordered route is for operational availability and records the actual provider.

## Generator and case-authoring follow-up

GLM 5.2 xhigh reached 49.0% strict joint correctness on the frozen 300-case
public RAG pilot, below the 51.67% task-aware DeepSeek ensemble, so it was not
promoted as the answer generator. A separate high-reasoning authoring run
produced 20 multi-hop proposals. Cross-model exact-source review approved 8 for
future AI-reviewed adversarial integration and rejected 12; this added zero
human-reviewed cases. One authoring batch exercised the StreamLake fallback.

## Frozen evidence

- Bakeoff summary SHA-256: `b98194f76073b137218c8e2df8fe3595985b50102966d0fd89577466bef6a8fb`
- Human labels SHA-256: `f411576104b635977bb21fa993adfc5103a7e03369598c0b027ade0ca0f7a5ed`
- Calibration manifest SHA-256: `30ccd2700da9deb9dd2da8d411d5f04547c26d06dea1cab78c747030a04b4c29`
- Provider parity report SHA-256: `22bdf1fcecf50478add79a9bab328d11e2da421833333f365edaa32258bfeef9`
- GLM generator pilot SHA-256: `66410ffacd6f596e25aad490fd9fa274b5271f8859b183e4876dd2f08e2c4990`
- Cross-model candidate review SHA-256: `0c47937bc0b5bb555661f1022942f3c6f840200a682e662cb2fa51adfa4d17c4`
