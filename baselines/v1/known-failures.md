# V1 known failures and adjudication

Status: frozen retrospective triage of the accepted 49-case report. These
results are evidence, not a to-do list to edit inside the accepted artifact.
Any behavior fix must produce a separate candidate report.

The accepted report contains one retrieval miss, one false answer on the
unanswerable lane, several partial generation errors, and several likely judge
instrument errors. The original judge returned only an integer, so rationales
and provider revisions are unavailable; classifications below are a manual
source-to-answer review performed during Phase 0.

| Case | Recorded result | Phase 0 adjudication |
| --- | --- | --- |
| `q02` | correctness 8/10 | Answer contains the required single-host/second-rung fact. No material error found; likely judge granularity/variance. |
| `q13` | retrieval miss; correctness 0/10; faithfulness 1/10 | Confirmed end-to-end failure. The required host/precision span was not retrieved; the answer omitted NextBit and hallucinated “70B precision” instead of bf16. |
| `q17` | correctness 9/10 | Minor incomplete answer: it states the uid/chown requirement but omits `chmod 400`. |
| `q18` | correctness 9/10 | Minor incomplete answer: it explains the committed-files-only archive but omits that fresh production secrets travel separately via stdin. |
| `q20` | faithfulness 7/10 | All claims appear in supplied sections, including the k3s install and secret creation. Likely judge under-score, though the answer adds irrelevant deployment detail. |
| `q21` | correctness 7/10; faithfulness 7/10 | Confirmed numeric generation error: Grafana is `30301`, not `3001`. The correct span was retrieved. |
| `q29` | correctness 9/10 | Required size and pressure timing are present. No material error found; likely judge granularity. |
| `q34` | correctness 7/10; faithfulness 7/10 | Confirmed attribution error: rejected examples were anti-steered caves scored at most 4, not low-scoring teacher holds. |
| `q35` | correctness 3/10 | Materially incomplete/misframed: the answer omits the frozen-suite regressions and says pressure resistance failed, although the source says holding posture persisted while grounding/correctness failed. |
| `q36` | faithfulness 0/10 | Confirmed judge false negative. `Q4_K_M` exactly matches both the reference and retrieved source. |
| `u07` | did not abstain | Confirmed refusal failure. The answer gives nearby Euryale facts but never answers the absent context-window question. |

Primary V1 failure IDs for regression tracking:

- Retrieval: `q13`.
- Generation after successful retrieval: `q17`, `q18`, `q21`, `q34`, `q35`.
- Abstention: `u07`.
- Judge reliability review: `q02`, `q20`, `q29`, `q36`.

This triage must not be converted into corrected V1 scores. Human calibration
labels and structured judge rationales belong in V2.
