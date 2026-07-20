# V2 development reports

`structured-dev-51.json` is the first complete structured-evaluation diagnostic
run over the 51 Phase 1 model-proposed cases. It is intentionally retained as
failure evidence, not a baseline or resume result.

Important boundaries:

- The source cases were unreviewed proposals, so `release_eligible` is false.
- The run predates the derived-response contract. Its prompt hash records the
  exact earlier contract; current answered outputs use `response: null` and
  render user-facing prose only from validated claim text.
- It used the existing V1 Vbot E5 index at `k=4`, not the separate public
  benchmark indexes.
- It is not judge-calibrated against human output labels.

The report is still useful because it exposed real failure modes: only 45.1%
end-to-end case pass, 78.43% exact action accuracy, 83.91% generated-claim
support, 67.05% required-claim coverage, and 94.74% valid judge output. All
citations existed, proving that citation existence alone is not grounding.
Provider-reported cost was $0.010414 across 89 calls.

## Current controlled diagnostics

- `structured-dev-current-k4.json` is the current derived-response diagnostic:
  49.02% end to end, 84.31% action accuracy, 81.69% claim support, and 67.03%
  required-claim coverage. It remains release-ineligible and unreviewed.
- `structured-dev-current-k6.json` and `structured-dev-current-k8.json` are
  rejected depth experiments. Better retrieval/support did not compensate for
  action and safety regressions; see the matching comparison reports.
- `structured-dev-safety-verifier-v2-k4.json` is a 15-case gold-free runtime
  verifier diagnostic, not a release result.
- `abstention-calibration-dev.json` shows that no single scalar threshold met
  the predeclared false-answer and false-refusal limits.

## Judge calibration preflight

- `judge-preflight-catalog.json` is the zero-inference catalog check.
- `judge-v4-provider-probe.json` retains repeated pinned-provider successes and
  failures. Advertised structured-output support was not treated as proof.
- `judge-preflight-pinned.json` is the final 4/4 strict-schema smoke artifact:
  V4 high/max use Baidu FP8, Gemini uses Google Vertex global, and GPT-5.4 uses
  OpenAI. Every profile disables provider fallbacks.
