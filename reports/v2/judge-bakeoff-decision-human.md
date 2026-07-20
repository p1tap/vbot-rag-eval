# Judge bakeoff and cascade decision

The calibration reference is the original frozen human-label artifact.

Decision date: 2026-07-17

## Decision

Selected policy: `v4-primary-failure-cascade`.

The policy was selected using only the frozen development partition. Ranking minimizes high/critical false accepts, total false accepts, false rejects, categorical disagreements, fail-closed decisions, cost, and latency—in that order. The confirmation partition was opened only after the policy identity was fixed.

No future human-review queue is part of this policy. Unresolved or invalid automated adjudication fails closed and is recorded as AI review, never relabeled as human review.

## Policy comparison

| Policy | Dev agreement | Dev false accepts | Confirmation agreement | Confirmation false accepts | Calls | Reported cost |
|---|---:|---:|---:|---:|---:|---:|
| `single-gemini-3.5-flash-high` | 74.6% | 1 | 86.5% | 0 | 82 | $1.094448 |
| `single-gpt-5.4-high` | 69.6% | 0 | 83.8% | 0 | 82 | $0.922275 |
| `single-v4-flash-high` | 65.2% | 1 | 75.7% | 0 | 82 | $0.022016 |
| `v4-primary-severity-sample` | 72.5% | 0 | 86.5% | 0 | 180 | $1.400716 |
| `v4-primary-failure-cascade` | 73.2% | 0 | 86.5% | 0 | 158 | $1.107654 |
| `v4-gemini-disagreement-gpt` | 73.2% | 0 | 86.5% | 0 | 187 | $1.455694 |
| `conservative-full-panel` | 73.2% | 0 | 86.5% | 0 | 246 | $2.038739 |

## Selected-policy result

- Development: 73.2% agreement; 0 false accepts.
- Hidden confirmation: 86.5% agreement; 0 false accepts.
- Overall: 76.0% agreement; 0 false accepts and 37 false rejects.
- One-sided 95% upper bound on the observed overall false-accept rate: 1.5%.

This calibration set is deliberately failure-rich and small. Its point metrics select an operating policy; they do not prove the same error rate on unseen production traffic.

## Frozen evidence

- Bakeoff summary SHA-256: `b98194f76073b137218c8e2df8fe3595985b50102966d0fd89577466bef6a8fb`
- Human labels SHA-256: `f411576104b635977bb21fa993adfc5103a7e03369598c0b027ade0ca0f7a5ed`
- Calibration manifest SHA-256: `30ccd2700da9deb9dd2da8d411d5f04547c26d06dea1cab78c747030a04b4c29`
