# Judge bakeoff and cascade decision

The calibration reference includes separately recorded unanimous three-family AI overrides; the original human labels remain frozen.

Decision date: 2026-07-17

## Decision

Selected policy: `v4-primary-failure-cascade`.

The policy was selected using only the frozen development partition. Ranking minimizes high/critical false accepts, total false accepts, false rejects, categorical disagreements, fail-closed decisions, cost, and latency—in that order. The confirmation partition was opened only after the policy identity was fixed.

No future human-review queue is part of this policy. Unresolved or invalid automated adjudication fails closed and is recorded as AI review, never relabeled as human review.

## Policy comparison

| Policy | Dev agreement | Dev false accepts | Confirmation agreement | Confirmation false accepts | Calls | Reported cost |
|---|---:|---:|---:|---:|---:|---:|
| `single-gemini-3.5-flash-high` | 87.7% | 1 | 97.3% | 0 | 82 | $1.094448 |
| `single-gpt-5.4-high` | 82.6% | 0 | 94.6% | 0 | 82 | $0.922275 |
| `single-v4-flash-high` | 78.3% | 1 | 86.5% | 0 | 82 | $0.022016 |
| `v4-primary-severity-sample` | 85.5% | 0 | 97.3% | 0 | 180 | $1.400716 |
| `v4-primary-failure-cascade` | 86.2% | 0 | 97.3% | 0 | 158 | $1.107654 |
| `v4-gemini-disagreement-gpt` | 86.2% | 0 | 97.3% | 0 | 187 | $1.455694 |
| `conservative-full-panel` | 86.2% | 0 | 97.3% | 0 | 246 | $2.038739 |

## Selected-policy result

- Development: 86.2% agreement; 0 false accepts.
- Hidden confirmation: 97.3% agreement; 0 false accepts.
- Overall: 88.6% agreement; 0 false accepts and 17 false rejects.
- One-sided 95% upper bound on the observed overall false-accept rate: 1.5%.

This calibration set is deliberately failure-rich and small. Its point metrics select an operating policy; they do not prove the same error rate on unseen production traffic.

## Frozen evidence

- Bakeoff summary SHA-256: `b98194f76073b137218c8e2df8fe3595985b50102966d0fd89577466bef6a8fb`
- Human labels SHA-256: `f411576104b635977bb21fa993adfc5103a7e03369598c0b027ade0ca0f7a5ed`
- Calibration manifest SHA-256: `30ccd2700da9deb9dd2da8d411d5f04547c26d06dea1cab78c747030a04b4c29`
- AI adjudication overrides SHA-256: `2e5a6a8eb25fb8fd0f630bf0f1ccdeb9218e2f9a0086a184d40fa2076647bcde`
